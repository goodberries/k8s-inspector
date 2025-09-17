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
    # New fields for planning
    plan: Optional[str]  # "direct" or "get_then_filter"
    filter_criteria: Optional[Dict[str, Any]]
    # Existing fields
    command: Optional[List[str]]
    output: Optional[Any]
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
app = FastAPI(title="K8s NL Interface Service", version="0.4.0-agent")
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

def planner(state: AgentState) -> Dict[str, Any]:
    br = _bedrock_client()
    if not br: return {"error": "Bedrock client not available"}
    
    system = (
        "You are a Kubernetes query planner. Your job is to decide how to answer a user's question. "
        "You have two plans: 'direct' for simple questions, and 'get_then_filter' for questions that require searching or filtering. "
        "Output STRICT JSON only. "
        "Schema for 'direct': {\"plan\": \"direct\", \"command\": [\"kubectl\", \"...\"]}. "
        "Schema for 'get_then_filter': {\"plan\": \"get_then_filter\", \"resource\": \"<pods|services|nodes|...etc>\", \"filter\": {\"field\": \"<e.g., metadata.name>\", \"operator\": \"<startswith|contains|equals>\", \"value\": \"<string>\"}}. "
        "For 'get_then_filter', the resource should be what you need to list broadly. "
        "Always use read-only commands (get, describe, logs, top). Never use write commands."
    )
    hints = state.get("hints", {})
    hint_lines = [f"{k}={v}" for k, v in hints.items() if v]
    user = f"Question: {state['question']}\nHints: {', '.join(hint_lines) if hint_lines else 'none'}"
    
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 400, "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = br.invoke_model(
            modelId=BEDROCK_MODEL_ID, body=json.dumps(body).encode("utf-8"),
            accept="application/json", contentType="application/json"
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        text_resp = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
        plan_data = json.loads(text_resp)

        if plan_data.get("plan") == "get_then_filter":
            resource = plan_data["resource"]
            return {
                "plan": "get_then_filter",
                "command": ["kubectl", "get", resource, "-A", "-o", "json"],
                "filter_criteria": plan_data["filter"],
            }
        elif plan_data.get("plan") == "direct":
            return {"plan": "direct", "command": plan_data["command"]}
        else:
            return {"error": "LLM returned an invalid plan."}
    except Exception as e:
        return {"error": f"Planner failed: {e}"}

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
        
        # Check return code first. A non-zero exit code is a definitive error.
        if proc.returncode != 0:
            return {"error": f"Kubectl command failed with exit code {proc.returncode}:\n{proc.stderr}"}
        stdout = proc.stdout
        stderr = proc.stderr # Capture stderr even on success for warnings
        parsed = stdout

        print("--- Raw stdout (first 5 lines) ---")
        for line in stdout.splitlines()[:5]:
            print(line)
        print("--- End of raw stdout ---")
        
        if stderr:
            print(f"--- Stderr --- \n{stderr}\n--- End of stderr ---")

        if stdout:
            try:
                parsed = json.loads(stdout)
                print("--- Parsed JSON (first 5 items) ---")
                if isinstance(parsed, dict) and "items" in parsed:
                    for item in parsed.get("items", [])[:5]:
                        # Print just the name for brevity
                        print(f"- {item.get('kind', 'Item')} named '{item.get('metadata', {}).get('name')}'")
                else:
                    print("Parsed JSON is not in the expected 'items' format.")
                print("--- End of parsed JSON ---")
            except json.JSONDecodeError:
                # Not JSON, treat as plain text. This is expected for commands like 'describe' or 'logs'.
                pass
        
        # Pass stderr to the error field to make it visible in the UI, but don't treat it as a failure.
        return {"output": parsed, "error": stderr or None}
        
    except Exception as e:
        return {"error": str(e)}

def filter_results(state: AgentState) -> Dict[str, Any]:
    criteria = state.get("filter_criteria")
    output = state.get("output")
    if not criteria or not output or not isinstance(output, dict) or "items" not in output:
        return {"error": "Cannot filter: invalid criteria or non-list output."}

    field_path = criteria["field"].split('.')
    op = criteria["operator"]
    value = criteria["value"]
    
    filtered_items = []
    for item in output.get("items", []):
        try:
            current_val = item
            for key in field_path:
                current_val = current_val[key]
            
            match = False
            if op == "startswith" and str(current_val).startswith(value): match = True
            elif op == "contains" and value in str(current_val): match = True
            elif op == "equals" and str(current_val) == value: match = True
            
            if match:
                filtered_items.append(item)
        except (KeyError, TypeError):
            continue
            
    return {"output": {"items": filtered_items}}

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
def route_after_validation(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "execute_command"

def route_after_execution(state: AgentState) -> str:
    if state.get("error"):
        return "summarize_output" # Summarize the error
    if state.get("plan") == "get_then_filter":
        return "filter_results"
    return "summarize_output"

def handle_error(state: AgentState) -> Dict[str, Any]:
    return {"is_final": True}

# --- Build and Compile Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("validate_command", validate_command)
workflow.add_node("execute_command", execute_command)
workflow.add_node("filter_results", filter_results)
workflow.add_node("summarize_output", summarize_output)
workflow.add_node("handle_error", handle_error)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "validate_command")
workflow.add_conditional_edges("validate_command", route_after_validation)
workflow.add_conditional_edges("execute_command", route_after_execution)
workflow.add_edge("filter_results", "summarize_output")
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
        action=final_state.get("plan") or "agent",
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
