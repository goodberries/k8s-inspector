import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import re

# Simple mapping and guardrail layer for allowed kubectl commands
ALLOWED_ACTIONS = {
    "list_pods": {
        "patterns": ["what are the pods", "list pods", "show pods", "pods running"],
        "command": ["kubectl", "get", "pods", "-A", "-o", "json"],
        "post": "pods",
    },
    "list_services": {
        "patterns": ["list services", "what services", "show services"],
        "command": ["kubectl", "get", "svc", "-A", "-o", "json"],
        "post": "services",
    },
    "list_nodes": {
        "patterns": ["list nodes", "what nodes", "show nodes"],
        "command": ["kubectl", "get", "nodes", "-o", "json"],
        "post": "nodes",
    },
    "list_namespaces": {
        "patterns": ["list namespaces", "what namespaces", "show namespaces"],
        "command": ["kubectl", "get", "ns", "-o", "json"],
        "post": "namespaces",
    },
    # New actions (read-only)
    "logs_pod": {
        "patterns": ["logs", "show logs", "pod logs", "crashloop logs"],
        "command": ["kubectl", "logs"],  # name/namespace/container/tail added dynamically
        "post": None,
    },
    "describe_pod": {
        "patterns": ["describe pod"],
        "command": ["kubectl", "describe", "pod"],  # name/ns added dynamically
        "post": None,
    },
    "describe_service": {
        "patterns": ["describe service", "describe svc"],
        "command": ["kubectl", "describe", "service"],
        "post": None,
    },
    "describe_node": {
        "patterns": ["describe node"],
        "command": ["kubectl", "describe", "node"],
        "post": None,
    },
    "describe_namespace": {
        "patterns": ["describe namespace"],
        "command": ["kubectl", "describe", "namespace"],
        "post": None,
    },
}

# Bedrock configuration via env
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
ENABLE_SUMMARIZE = os.getenv("ENABLE_SUMMARIZE", "true").lower() == "true"
MAX_SUMMARY_INPUT_CHARS = int(os.getenv("MAX_SUMMARY_INPUT_CHARS", "6000"))

try:
    import boto3
except Exception:
    boto3 = None

app = FastAPI(title="K8s NL Interface Service", version="0.2.0")

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
    # Optional direct hints (bypass some NL extraction)
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


def post_process(kind: str, payload: Dict[str, Any]) -> Any:
    if kind == "pods":
        items = payload.get("items", [])
        return [
            {
                "namespace": i.get("metadata", {}).get("namespace"),
                "name": i.get("metadata", {}).get("name"),
                "phase": i.get("status", {}).get("phase"),
                "nodeName": i.get("spec", {}).get("nodeName"),
                "containers": [c.get("name") for c in i.get("spec", {}).get("containers", [])],
            }
            for i in items
        ]
    if kind == "services":
        items = payload.get("items", [])
        return [
            {
                "namespace": i.get("metadata", {}).get("namespace"),
                "name": i.get("metadata", {}).get("name"),
                "type": i.get("spec", {}).get("type"),
                "clusterIP": i.get("spec", {}).get("clusterIP"),
                "ports": i.get("spec", {}).get("ports"),
            }
            for i in items
        ]
    if kind == "nodes":
        items = payload.get("items", [])
        return [
            {
                "name": i.get("metadata", {}).get("name"),
                "roles": i.get("metadata", {}).get("labels", {}).get("kubernetes.io/role"),
                "kubeletVersion": i.get("status", {}).get("nodeInfo", {}).get("kubeletVersion"),
                "conditions": i.get("status", {}).get("conditions", []),
            }
            for i in items
        ]
    if kind == "namespaces":
        items = payload.get("items", [])
        return [
            {
                "name": i.get("metadata", {}).get("name"),
                "status": i.get("status", {}).get("phase"),
            }
            for i in items
        ]
    return payload


def _bedrock_client():
    if not boto3:
        return None
    try:
        return boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception:
        return None


def guess_action_with_llm(question: str) -> Optional[str]:
    # Backward-compatible simple classifier for action only
    q = question.lower()
    for action, cfg in ALLOWED_ACTIONS.items():
        if any(p in q for p in cfg["patterns"]):
            return action
    br = _bedrock_client()
    if not br:
        return None
    try:
        system = (
            "You are a classifier that maps a user question about Kubernetes into one of these actions: "
            + ", ".join(ALLOWED_ACTIONS.keys())
            + ". If none match, answer 'none'. Output only the action keyword."
        )
        user = f"Question: {question}"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20,
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
        text = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        action = text.strip().split()[0].lower()
        if action in ALLOWED_ACTIONS:
            return action
    except Exception:
        return None
    return None


def extract_intent(question: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    q = question.lower()
    intent: Dict[str, Any] = {
        "action": None,
        "namespace": defaults.get("namespace"),
        "name": defaults.get("name"),
        "resource": defaults.get("resource"),
        "container": defaults.get("container"),
        "tail_lines": defaults.get("tail_lines") or 200,
    }

    # Heuristic extraction
    ns_m = re.search(r"in namespace\s+([a-z0-9-]+)", q)
    if ns_m:
        intent["namespace"] = ns_m.group(1)

    cont_m = re.search(r"container\s+([a-z0-9-]+)", q)
    if cont_m:
        intent["container"] = cont_m.group(1)

    tail_m = re.search(r"last\s+(\d+)\s+lines|tail\s+(\d+)", q)
    if tail_m:
        intent["tail_lines"] = int(next(g for g in tail_m.groups() if g))

    # logs
    if "log" in q:
        intent["action"] = "logs_pod"
        name_m = re.search(r"pod[s]?\s+([a-z0-9][-a-z0-9\.]+)", q)
        if not name_m:
            # try quoted name
            name_m = re.search(r"pod[s]?\s+\"([^\"]+)\"", q)
        if name_m:
            intent["name"] = name_m.group(1)
        intent["resource"] = "pod"
        return intent

    # describe
    desc_m = re.search(r"describe\s+(pod|service|svc|node|namespace)\s+([a-z0-9][-a-z0-9\.]+)", q)
    if desc_m:
        rsrc = desc_m.group(1)
        rsrc = "service" if rsrc == "svc" else rsrc
        intent["resource"] = rsrc
        intent["name"] = desc_m.group(2)
        intent["action"] = f"describe_{rsrc}"
        return intent

    # lists
    for action, cfg in ALLOWED_ACTIONS.items():
        if any(p in q for p in cfg.get("patterns", [])) and action.startswith("list_"):
            intent["action"] = action
            return intent

    # LLM fallback to structured JSON
    br = _bedrock_client()
    if not br:
        intent["action"] = guess_action_with_llm(question) or "list_pods"
        return intent
    try:
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ALLOWED_ACTIONS.keys())},
                "namespace": {"type": "string"},
                "name": {"type": "string"},
                "resource": {"type": "string", "enum": ["pod", "service", "node", "namespace", ""]},
                "container": {"type": "string"},
                "tail_lines": {"type": "integer"},
            },
            "required": ["action"],
            "additionalProperties": False,
        }
        system = (
            "Extract intent for a Kubernetes question. Return STRICT JSON ONLY matching this schema: "
            + json.dumps(schema)
            + ". If unsure, choose a reasonable default (list_pods)."
        )
        user = f"Question: {question}"
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
        text = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        parsed = json.loads(text)
        intent.update({k: v for k, v in parsed.items() if v})
    except Exception:
        intent["action"] = guess_action_with_llm(question) or "list_pods"
    return intent


def build_command(action: str, namespace: Optional[str], name: Optional[str], container: Optional[str], tail_lines: Optional[int]) -> List[str]:
    cfg = ALLOWED_ACTIONS[action]
    base = list(cfg["command"])

    if action.startswith("list_"):
        cmd = list(base)
        if namespace and "-A" in cmd:
            cmd = [c for c in cmd if c != "-A"] + ["-n", namespace]
        return cmd

    if action.startswith("describe_"):
        # base already includes resource kind
        if not name:
            raise HTTPException(status_code=400, detail="Resource name is required for describe")
        cmd = list(base) + [name]
        if namespace and action in ("describe_pod", "describe_service", "describe_namespace"):
            cmd = cmd + ["-n", namespace]
        return cmd

    if action == "logs_pod":
        if not name:
            raise HTTPException(status_code=400, detail="Pod name is required for logs")
        cmd = list(base) + [name]
        if namespace:
            cmd += ["-n", namespace]
        if container:
            cmd += ["-c", container]
        cmd += ["--tail", str(tail_lines or 200)]
        return cmd

    raise HTTPException(status_code=400, detail="Unsupported action")


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
        return ExecResult(action="", kubectl=cmd, stdout=parsed, stderr=stderr, exit_code=code)
    except subprocess.TimeoutExpired as e:
        return ExecResult(action="", kubectl=cmd, stdout=None, stderr=str(e), exit_code=124)
    except Exception as e:
        return ExecResult(action="", kubectl=cmd, stdout=None, stderr=str(e), exit_code=1)


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
        user = f"Action: {action}\nKubectl output snippet:\n{text}"
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
    # Extract intent (action + parameters)
    defaults = {
        "namespace": query.namespace,
        "name": query.name,
        "resource": query.resource,
        "container": query.container,
        "tail_lines": query.tail_lines,
    }
    intent = extract_intent(query.question, defaults)
    action = intent.get("action") or "list_pods"
    if action not in ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail="Unsupported action")

    cmd = build_command(
        action=action,
        namespace=intent.get("namespace"),
        name=intent.get("name"),
        container=intent.get("container"),
        tail_lines=intent.get("tail_lines"),
    )
    result = safe_execute(cmd)
    result.action = action

    # Post-process JSON into concise structure if applicable
    if isinstance(result.stdout, dict):
        kind = ALLOWED_ACTIONS[action].get("post")
        result.stdout = post_process(kind, result.stdout)

    # Summarize into plain English and proposed next steps
    if query.explain:
        summary, suggestions = summarize_output(action, result.stdout, result.stderr)
        result.summary = summary
        result.suggestions = suggestions

    return result

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
