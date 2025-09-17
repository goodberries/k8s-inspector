import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import re

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    hints: Dict[str, Any]
    command: Optional[List[str]]
    output: Optional[str]
    error: Optional[str]
    summary: Optional[str]
    suggestions: Optional[List[str]]
    is_final: bool = False

# --- Bedrock/LLM Configuration ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
ENABLE_SUMMARIZE = os.getenv("ENABLE_SUMMARIZE", "true").lower() == "true"
MAX_SUMMARY_INPUT_CHARS = int(os.getenv("MAX_SUMMARY_INPUT_CHARS", "6000"))
MAX_AGENT_LOOPS = int(os.getenv("MAX_AGENT_LOOPS", 3))

try:
    import boto3
except Exception:
    boto3 = None

# --- FastAPI App Setup ---
app = FastAPI(title="K8s NL Interface Service", version="0.3.0-langgraph")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# --- Pydantic Models ---
class NLQuery(BaseModel):
    question: str
    namespace: Optional[str] = None
    name: Optional[str] = None
    resource: Optional[str] = None
    container: Optional[str] = None
    tail_lines: Optional[int] = None
    explain: Optional[bool] = True

class ExecResult(BaseModel):
    action: str
    kubectl: List[str]
    stdout: Any
    stderr: str
    exit_code: int
    summary: Optional[str] = None
    suggestions: Optional[List[str]] = None

# --- LangGraph Nodes ---
def _bedrock_client() -> Optional[Any]:
    if not boto3: return None
    try: return boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception: return None

def generate_command(state: AgentState) -> Dict[str, Any]:
    br = _bedrock_client()
    if not br:
        return {"error": "Bedrock client not available"}
    try:
        system = (
            "You are an assistant that translates a user's Kubernetes question into a single, read-only kubectl command. "
            "Rules: "
            "- Only output a single command in STRICT JSON as {\"kubectl\": [\"kubectl\", ...]}. No prose. "
            "- Use only read-only subcommands: get, describe, logs, top, api-resources, api-versions, cluster-info, version. "
            "- Never use write/unsafe ops (delete, apply, create, replace, patch, edit, exec, attach, drain, cordon, taint, annotate, label). "
            "- Prefer including '-o json' for 'get' commands. "
            "- If the user hints namespace/name/container/tail_lines, incorporate them. "
            "- If ambiguous, choose a safe default that best answers the question."
        )
        hints = state.get("hints", {})
        hint_lines = [f"{k}={v}" for k, v in hints.items() if v]
        user = f"Question: {state['question']}\nHints: {', '.join(hint_lines) if hint_lines else 'none'}"
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 200, "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = br.invoke_model(
            modelId=BEDROCK_MODEL_ID, body=json.dumps(body).encode("utf-8"),
            accept="application/json", contentType="application/json"
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        text_resp = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
        data = json.loads(text_resp)
        args = data.get("kubectl")
        if isinstance(args, list) and args and args[0] == "kubectl":
            return {"command": [str(a) for a in args]}
    except Exception as e:
        return {"error": f"LLM command generation failed: {e}"}
    return {"error": "Failed to generate a valid command"}

READ_ONLY_SUBCMDS = {"get", "describe", "logs", "top", "api-resources", "api-versions", "cluster-info", "version"}
FORBIDDEN_TOKENS = {";", "|", "&&", "||", ">", "<", "`", "$(", "${"}
FORBIDDEN_SUBCMDS = {"delete", "apply", "create", "replace", "patch", "edit", "exec", "attach", "drain", "cordon", "taint", "annotate", "label", "scale", "rollout", "expose", "set"}

def validate_command(state: AgentState) -> Dict[str, Any]:
    args = state.get("command")
    if not args or args[0] != "kubectl":
        return {"error": "Invalid command from LLM: must start with kubectl"}
    for t in args:
        for bad in FORBIDDEN_TOKENS:
            if bad in t: return {"error": "Unsafe token in command"}
    if len(args) >= 2:
        sub = args[1]
        if sub in FORBIDDEN_SUBCMDS or sub not in READ_ONLY_SUBCMDS:
            return {"error": f"Unsupported kubectl subcommand: {sub}"}
    else:
        return {"error": "Incomplete kubectl command"}
    return {}

def execute_command(state: AgentState) -> Dict[str, Any]:
    cmd = state["command"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.getenv("KCTL_TIMEOUT", "20")))
        stdout = proc.stdout
        stderr = proc.stderr
        parsed = stdout
        if stdout:
            try: parsed = json.loads(stdout)
            except Exception: pass
        return {"output": parsed, "error": stderr or None}
    except Exception as e:
        return {"error": str(e)}

def summarize_output(state: AgentState) -> Dict[str, Any]:
    if not ENABLE_SUMMARIZE: return {}
    br = _bedrock_client()
    if not br: return {}
    text = ""
    if state.get("output"):
        text = json.dumps(state["output"], indent=2) if isinstance(state["output"], (dict, list)) else str(state["output"])
    elif state.get("error"):
        text = state["error"]
    if not text: return {}
    text = text[-MAX_SUMMARY_INPUT_CHARS:]
    try:
        system = (
            "You are a Kubernetes assistant. Given kubectl output, write: "
            "1) a concise, plain-English summary (1-3 sentences). "
            "2) 3-6 actionable suggestions to investigate or fix potential issues. "
            "Be specific but safe; only suggest read-only checks unless the issue is obvious. "
            "Output STRICT JSON with keys: summary (string), suggestions (array of strings)."
        )
        user = f"Kubectl output snippet:\n{text}"
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 500, "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = br.invoke_model(
            modelId=BEDROCK_MODEL_ID, body=json.dumps(body).encode("utf-8"),
            accept="application/json", contentType="application/json"
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        text_resp = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
        data = json.loads(text_resp)
        return {"summary": data.get("summary"), "suggestions": data.get("suggestions")}
    except Exception:
        return {}

# --- LangGraph Conditional Edges ---
def should_execute(state: AgentState) -> str:
    return "execute_command" if not state.get("error") else "handle_error"

def should_summarize(state: AgentState) -> str:
    return "summarize_output" if state.get("output") or state.get("error") else END

def handle_error(state: AgentState) -> Dict[str, Any]:
    # For now, just end. Could add retry logic here.
    return {"is_final": True}

# --- Build and Compile Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("generate_command", generate_command)
workflow.add_node("validate_command", validate_command)
workflow.add_node("execute_command", execute_command)
workflow.add_node("summarize_output", summarize_output)
workflow.add_node("handle_error", handle_error)

workflow.set_entry_point("generate_command")
workflow.add_edge("generate_command", "validate_command")
workflow.add_conditional_edges("validate_command", should_execute)
workflow.add_edge("execute_command", "summarize_output")
workflow.add_edge("summarize_output", END)
workflow.add_edge("handle_error", END)

app_graph = workflow.compile()

# --- API Endpoint ---
@app.post("/ask", response_model=ExecResult)
def ask(query: NLQuery):
    initial_state = AgentState(
        question=query.question,
        hints={
            "namespace": query.namespace, "name": query.name, "resource": query.resource,
            "container": query.container, "tail_lines": query.tail_lines,
        }
    )
    final_state = app_graph.invoke(initial_state)

    return ExecResult(
        action="langgraph",
        kubectl=final_state.get("command") or [],
        stdout=final_state.get("output") or "",
        stderr=final_state.get("error") or "",
        exit_code=0 if not final_state.get("error") else 1,
        summary=final_state.get("summary"),
        suggestions=final_state.get("suggestions"),
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
