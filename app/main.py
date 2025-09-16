import os
import json
import subprocess
import shlex
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import re

# Bedrock configuration via env
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
ENABLE_SUMMARIZE = os.getenv("ENABLE_SUMMARIZE", "true").lower() == "true"
MAX_SUMMARY_INPUT_CHARS = int(os.getenv("MAX_SUMMARY_INPUT_CHARS", "6000"))

try:
    import boto3
except Exception:
    boto3 = None

app = FastAPI(title="K8s NL Interface Service", version="0.3.0")

# CORS: allow all (demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static UI
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

class NLQuery(BaseModel):
    question: str
    namespace: Optional[str] = None
    # Optional direct hints
    name: Optional[str] = None
    resource: Optional[str] = None  # pod|service|node|namespace
    container: Optional[str] = None
    tail_lines: Optional[int] = None  # for logs
    explain: Optional[bool] = True

class ExecResult(BaseModel):
    action: str
    kubectl: List[str]
    stdout: Any
    stderr: str
    exit_code: int
    summary: Optional[str] = None
    suggestions: Optional[List[str]] = None


def _bedrock_client() -> Optional[Any]:
    if not boto3:
        return None
    try:
        return boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception:
        return None


def _fallback_command(hints: Dict[str, Any]) -> List[str]:
    # Reasonable default if LLM unavailable
    ns = hints.get("namespace")
    cmd = ["kubectl", "get", "pods", "-o", "json"]
    if ns:
        cmd += ["-n", ns]
    else:
        cmd = ["kubectl", "get", "pods", "-A", "-o", "json"]
    return cmd


def generate_kubectl_command(question: str, hints: Dict[str, Any]) -> List[str]:
    br = _bedrock_client()
    if not br:
        return _fallback_command(hints)
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
        hint_lines = [f"{k}={v}" for k, v in hints.items() if v]
        user = f"Question: {question}\nHints: {', '.join(hint_lines) if hint_lines else 'none'}"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = br.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        text_resp = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text_resp += block.get("text", "")
        data = json.loads(text_resp)
        args = data.get("kubectl")
        if isinstance(args, list) and args and args[0] == "kubectl":
            return [str(a) for a in args]
    except Exception:
        pass
    return _fallback_command(hints)


READ_ONLY_SUBCMDS = {"get", "describe", "logs", "top", "api-resources", "api-versions", "cluster-info", "version"}
FORBIDDEN_TOKENS = {";", "|", "&&", "||", ">", "<", "`", "$(`, "${"}
FORBIDDEN_SUBCMDS = {"delete", "apply", "create", "replace", "patch", "edit", "exec", "attach", "drain", "cordon", "taint", "annotate", "label", "scale", "rollout", "expose", "set"}


def validate_kubectl(args: List[str]) -> List[str]:
    if not args or args[0] != "kubectl":
        raise HTTPException(status_code=400, detail="Invalid command from LLM: must start with kubectl")
    # Disallow shell control tokens (we pass as list, but be extra safe)
    for t in args:
        for bad in FORBIDDEN_TOKENS:
            if bad in t:
                raise HTTPException(status_code=400, detail="Unsafe token in command")
    # Enforce read-only high-level subcommand
    if len(args) >= 2:
        sub = args[1]
        if sub in FORBIDDEN_SUBCMDS or sub not in READ_ONLY_SUBCMDS:
            raise HTTPException(status_code=400, detail=f"Unsupported kubectl subcommand: {sub}")
    else:
        raise HTTPException(status_code=400, detail="Incomplete kubectl command")
    return args


def safe_execute(cmd: List[str]) -> ExecResult:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("KCTL_TIMEOUT", "20")),
        )
        stdout = proc.stdout
        stderr = proc.stderr
        code = proc.returncode
        parsed = None
        if stdout:
            try:
                parsed = json.loads(stdout)
            except Exception:
                parsed = stdout
        return ExecResult(action="llm", kubectl=cmd, stdout=parsed, stderr=stderr, exit_code=code)
    except subprocess.TimeoutExpired as e:
        return ExecResult(action="llm", kubectl=cmd, stdout=None, stderr=str(e), exit_code=124)
    except Exception as e:
        return ExecResult(action="llm", kubectl=cmd, stdout=None, stderr=str(e), exit_code=1)


def _truncate_for_summary(text: str) -> str:
    if len(text) <= MAX_SUMMARY_INPUT_CHARS:
        return text
    # Prefer last part for logs
    return text[-MAX_SUMMARY_INPUT_CHARS:]


def summarize_output(action: str, stdout: Any, stderr: str) -> Tuple[Optional[str], Optional[List[str]]]:
    if not ENABLE_SUMMARIZE:
        return None, None
    br = _bedrock_client()
    if not br:
        return None, None
    # Prepare input text
    if isinstance(stdout, (dict, list)):
        text = json.dumps(stdout, indent=2)[:MAX_SUMMARY_INPUT_CHARS]
    elif isinstance(stdout, str) and stdout:
        text = _truncate_for_summary(stdout)
    else:
        text = _truncate_for_summary(stderr or "")
    if not text:
        return None, None
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
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = br.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        text_resp = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text_resp += block.get("text", "")
        data = json.loads(text_resp)
        return data.get("summary"), data.get("suggestions")
    except Exception:
        return None, None


@app.post("/ask", response_model=ExecResult)
def ask(query: NLQuery):
    hints = {
        "namespace": query.namespace,
        "name": query.name,
        "resource": query.resource,
        "container": query.container,
        "tail_lines": query.tail_lines,
    }
    args = generate_kubectl_command(query.question, hints)
    args = validate_kubectl(args)
    result = safe_execute(args)

    # Summarize into plain English and proposed next steps
    if query.explain:
        summary, suggestions = summarize_output(result.action, result.stdout, result.stderr)
        result.summary = summary
        result.suggestions = suggestions

    return result

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
