"""GeniusPro Superintelligence v1 — Main Application

Routes chat completion requests to the best expert model via Macro-MoE.
"""

import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from superintelligence.agents.swarm import AgentSwarm
from superintelligence.auth import require_auth
from superintelligence.coding.language_classifier import CodingLanguage, detect_coding_language
from superintelligence.coding.memory_store import MemorySnippet, list_snippets, save_snippet
from superintelligence.coding.onboarding import build_onboarding_response
from superintelligence.coding.prompts import (
    CODING_FIRST_TURN_BOOTSTRAP_PROMPT,
    CODING_PROTOCOL_SYSTEM_PROMPT,
)
from superintelligence.coding.summarize import build_summarize_messages
from superintelligence.coding.tool_contract import (
    get_cursor_coding_tool_definitions,
    merge_client_tools,
)
from superintelligence.config import load_config, SuperintelligenceConfig
from superintelligence.http.completion_service import (
    CompletionContext,
    build_chat_request,
    complete_with_fallback,
    inject_system_message,
    merge_tools,
    stream_with_fallback,
)
from superintelligence.providers.openai_compat import OpenAICompatProvider
from superintelligence.providers.anthropic_provider import AnthropicProvider
from superintelligence.routing.classifier import TaskType
from superintelligence.routing.router import SuperintelligenceRouter
from superintelligence.tools.registry import ToolRegistry
from superintelligence.tools.web_search import WebSearchTool
from superintelligence.tools.url_fetch import URLFetchTool

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("superintelligence")

# ─── Constants ────────────────────────────────────────────────────────────────

# Public “model” identifiers (what clients send in requests).
MODEL_NAME = "GeniusPro-agi-1.2"
CODING_MODEL_NAME = "GeniusPro-coding-agi-1.2"

# Legacy aliases (accepted for backwards compatibility).

# Public base URLs (what clients configure as base_url).
API_PREFIX = "/superintelligence/v1"
CODING_API_PREFIX = "/coding-superintelligence/v1"

# Backwards-compatible legacy prefix (older deployments / docs).
LEGACY_API_PREFIX = "/super-intelligence/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are GeniusPro, a helpful AI assistant created by GeniusPro. "
    "Be concise, friendly, and helpful. Never identify as any other AI or company. "
    "If the user asks who you are, say you are GeniusPro."
)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GeniusPro Superintelligence v1",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State ────────────────────────────────────────────────────────────────────

_config: Optional[SuperintelligenceConfig] = None
_session: Optional[aiohttp.ClientSession] = None
_router: Optional[SuperintelligenceRouter] = None
_swarm: Optional[AgentSwarm] = None
_tool_registry: Optional[ToolRegistry] = None


@app.on_event("startup")
async def startup() -> None:
    global _config, _session, _router, _swarm, _tool_registry

    logger.info("Starting GeniusPro Superintelligence v1...")
    _config = load_config()
    _session = aiohttp.ClientSession()
    _router = SuperintelligenceRouter()

    # Register providers from config
    for pc in _config.providers:
        if not pc.enabled:
            continue

        if pc.format == "anthropic":
            provider = AnthropicProvider(
                name=pc.name,
                base_url=pc.base_url,
                api_key=pc.api_key,
                default_model=pc.model_id,
                session=_session,
            )
        else:
            # OpenAI-compatible (openai, deepseek, openrouter, mistral, google)
            provider = OpenAICompatProvider(
                name=pc.name,
                base_url=pc.base_url,
                api_key=pc.api_key,
                default_model=pc.model_id,
                session=_session,
            )

        _router.register_provider(provider)

    provider_count = len(_router.providers)

    # Initialize tool registry with built-in tools
    _tool_registry = ToolRegistry()
    _tool_registry.register(WebSearchTool(session=_session))
    _tool_registry.register(URLFetchTool(session=_session))

    # Start the Agent Swarm
    _swarm = AgentSwarm(_router)
    await _swarm.start()

    logger.info("GeniusPro Superintelligence v1 ready — %d providers, %d built-in tools", provider_count, len(_tool_registry.tools))


@app.on_event("shutdown")
async def shutdown() -> None:
    global _session, _swarm
    if _swarm:
        await _swarm.stop()
    if _session:
        await _session.close()
    logger.info("Superintelligence v1 shut down.")


# ─── Auth helper ──────────────────────────────────────────────────────────────

async def _auth(request: Request) -> dict:
    """Authenticate the request."""
    if not _config or not _session:
        raise HTTPException(status_code=503, detail="Service not ready")
    return await require_auth(
        request, _session, _config.supabase_url, _config.supabase_service_key
    )


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get(f"{API_PREFIX}/health")
@app.get(f"{LEGACY_API_PREFIX}/health")
async def health() -> dict:
    provider_names = list(_router.providers.keys()) if _router else []
    return {
        "status": "ok",
        "service": MODEL_NAME,
        "providers": len(provider_names),
    }

@app.get(f"{CODING_API_PREFIX}/health")
async def coding_health() -> dict:
    provider_names = list(_router.providers.keys()) if _router else []
    return {
        "status": "ok",
        "service": CODING_MODEL_NAME,
        "providers": len(provider_names),
    }


# ─── Models ───────────────────────────────────────────────────────────────────

@app.get(f"{API_PREFIX}/models")
@app.get(f"{LEGACY_API_PREFIX}/models")
async def list_models(request: Request) -> dict:
    await _auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "geniuspro",
            }
        ],
    }

@app.get(f"{CODING_API_PREFIX}/models")
async def list_coding_models(request: Request) -> dict:
    await _auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": CODING_MODEL_NAME,
                "object": "model",
                "owned_by": "geniuspro",
            }
        ],
    }


# ─── Chat Completions ────────────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/chat/completions", response_model=None)
@app.post(f"{LEGACY_API_PREFIX}/chat/completions", response_model=None)
async def chat_completions(request: Request):
    req_id = uuid.uuid4().hex[:8]
    key_info = await _auth(request)
    _enforce_api_key_profile(key_info, request.url.path)
    start = time.time()

    body = await request.json()
    messages = body.get("messages", [])
    client_tools = body.get("tools", [])  # Client-defined tools

    last_msg = messages[-1].get("content", "")[:80] if messages else ""
    logger.info("[%s] ── New request ── user=%s stream=%s msgs=%d last_msg='%s'",
                req_id, key_info.get("user_id", "?")[:8], bool(body.get("stream", False)), len(messages), last_msg)

    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Inject default system prompt if client didn't provide one
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    # Classify the task
    task_type = _router.classify(messages) if _router else TaskType.GENERAL
    logger.info("[%s] Task classified as: %s", req_id, task_type.value)

    return await _run_completion(
        body=body,
        messages=messages,
        client_tools=client_tools,
        task_type=task_type,
        key_info=key_info,
        start=start,
        req_id=req_id,
        endpoint=str(request.url.path),
    )


# ─── Coding-specific Superintelligence ───────────────────────────────────────

@app.post(f"{CODING_API_PREFIX}/chat/completions", response_model=None)
@app.post(f"{API_PREFIX}/coding/chat/completions", response_model=None)
@app.post(f"{LEGACY_API_PREFIX}/coding/chat/completions", response_model=None)
async def coding_chat_completions(request: Request):
    req_id = uuid.uuid4().hex[:8]
    key_info = await _auth(request)
    _enforce_api_key_profile(key_info, request.url.path)
    start = time.time()

    body = await request.json()
    messages = body.get("messages", [])
    request_tools = body.get("tools", [])

    # If the client starts a new session with no user message yet,
    # return the onboarding question immediately.
    if not messages:
        model_name = CODING_MODEL_NAME if str(request.url.path).startswith(CODING_API_PREFIX) else MODEL_NAME
        return JSONResponse(build_onboarding_response(model_name=model_name))

    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    inject_system_message(messages, CODING_PROTOCOL_SYSTEM_PROMPT)
    if _is_first_turn(messages):
        inject_system_message(messages, CODING_FIRST_TURN_BOOTSTRAP_PROMPT)

    enable_cursor_tools = body.get("enable_cursor_tools", True)
    injected_tools = get_cursor_coding_tool_definitions() if enable_cursor_tools else []
    client_tools = merge_client_tools(request_tools=request_tools, injected_tools=injected_tools)

    # Allow explicit override; otherwise detect from messages.
    requested_language = body.get("coding_language")
    detected_language = detect_coding_language(messages)
    coding_language = detected_language
    if isinstance(requested_language, str):
        try:
            coding_language = CodingLanguage(requested_language.lower())
        except Exception:
            coding_language = detected_language

    return await _run_completion(
        body=body,
        messages=messages,
        client_tools=client_tools,
        task_type=TaskType.CODE,
        code_language=coding_language,
        key_info=key_info,
        start=start,
        req_id=req_id,
        endpoint=str(request.url.path),
        model_name=CODING_MODEL_NAME,
    )


@app.post(f"{CODING_API_PREFIX}/summarize", response_model=None)
@app.post(f"{API_PREFIX}/coding/summarize", response_model=None)
@app.post(f"{LEGACY_API_PREFIX}/coding/summarize", response_model=None)
async def coding_summarize(request: Request):
    req_id = uuid.uuid4().hex[:8]
    key_info = await _auth(request)
    _enforce_api_key_profile(key_info, request.url.path)
    start = time.time()

    body = await request.json()
    summary_type = body.get("type")
    content = body.get("content")
    instructions = body.get("instructions")

    if summary_type not in {"diff", "session", "file_or_folder", "selection"}:
        raise HTTPException(
            status_code=400,
            detail="type is required and must be one of: diff, session, file_or_folder, selection",
        )
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="content is required")

    messages = build_summarize_messages(
        summary_type=summary_type,
        content=content,
        instructions=instructions if isinstance(instructions, str) else None,
    )

    client_tools = body.get("tools", [])
    return await _run_completion(
        body=body,
        messages=messages,
        client_tools=client_tools,
        task_type=TaskType.CODE,
        code_language=CodingLanguage.UNKNOWN,
        key_info=key_info,
        start=start,
        req_id=req_id,
        endpoint=str(request.url.path),
    )


# ─── Coding memory (user-approved snippets only) ─────────────────────────────

@app.post(f"{CODING_API_PREFIX}/memory/snippets", response_model=None)
@app.post(f"{API_PREFIX}/coding/memory/snippets", response_model=None)
@app.post(f"{LEGACY_API_PREFIX}/coding/memory/snippets", response_model=None)
async def coding_memory_save_snippet(request: Request):
    key_info = await _auth(request)
    _enforce_api_key_profile(key_info, request.url.path)
    if not _config or not _session:
        raise HTTPException(status_code=503, detail="Service not ready")

    body = await request.json()
    approved = body.get("approved")
    project_slug = body.get("project_slug")
    content = body.get("content")
    language = body.get("language")
    tags = body.get("tags")
    workspace_fingerprint = body.get("workspace_fingerprint")

    if approved is not True:
        raise HTTPException(status_code=400, detail="approved=true is required")
    if not isinstance(project_slug, str) or not project_slug.strip():
        raise HTTPException(status_code=400, detail="project_slug is required")
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    # Hard safety limit: we do not want to store large code blocks by accident.
    if len(content) > 8000:
        raise HTTPException(status_code=400, detail="content too large (max 8000 chars)")
    if tags is not None and not isinstance(tags, list):
        raise HTTPException(status_code=400, detail="tags must be an array of strings")

    snippet = MemorySnippet(
        user_id=key_info["user_id"],
        project_slug=project_slug.strip(),
        content=content,
        language=language if isinstance(language, str) else None,
        tags=[t[:64] for t in tags if isinstance(t, str) and t.strip()][:20] if isinstance(tags, list) else None,
        workspace_fingerprint=workspace_fingerprint if isinstance(workspace_fingerprint, str) else None,
    )
    try:
        row = await save_snippet(
            session=_session,
            supabase_url=_config.supabase_url,
            supabase_service_key=_config.supabase_service_key,
            snippet=snippet,
        )
        return JSONResponse({"ok": True, "snippet": row})
    except Exception as e:
        # Handle duplicate inserts cleanly.
        msg = str(e)
        if "coding_memory_snippets_dedupe_idx" in msg or "duplicate key value" in msg:
            return JSONResponse({"ok": True, "duplicate": True})
        raise HTTPException(status_code=502, detail=msg)


@app.get(f"{CODING_API_PREFIX}/memory/snippets", response_model=None)
@app.get(f"{API_PREFIX}/coding/memory/snippets", response_model=None)
@app.get(f"{LEGACY_API_PREFIX}/coding/memory/snippets", response_model=None)
async def coding_memory_list_snippets(request: Request):
    key_info = await _auth(request)
    _enforce_api_key_profile(key_info, request.url.path)
    if not _config or not _session:
        raise HTTPException(status_code=503, detail="Service not ready")

    project_slug = request.query_params.get("project_slug", "")
    limit_raw = request.query_params.get("limit", "20")
    try:
        limit = int(limit_raw)
    except Exception:
        limit = 20

    if not project_slug.strip():
        raise HTTPException(status_code=400, detail="project_slug is required")

    try:
        rows = await list_snippets(
            session=_session,
            supabase_url=_config.supabase_url,
            supabase_service_key=_config.supabase_service_key,
            user_id=key_info["user_id"],
            project_slug=project_slug.strip(),
            limit=limit,
        )
        return JSONResponse({"ok": True, "snippets": rows})
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _run_completion(
    *,
    body: dict,
    messages: list[dict],
    client_tools: list[dict],
    task_type: TaskType,
    code_language: CodingLanguage = CodingLanguage.UNKNOWN,
    key_info: dict,
    start: float,
    req_id: str,
    endpoint: str,
    model_name: str = MODEL_NAME,
):
    if not _config or not _session or not _router:
        raise HTTPException(status_code=503, detail="Service not ready")

    all_tools = merge_tools(client_tools=client_tools, tool_registry=_tool_registry)
    chain = _router.get_fallback_chain(task_type, code_language=code_language)
    if not chain:
        logger.error("[%s] No providers available for task_type=%s", req_id, task_type.value)
        raise HTTPException(status_code=503, detail="No providers available")

    chain_names = [p.name for p in chain]
    logger.info("[%s] Fallback chain: %s", req_id, " → ".join(chain_names))

    chat_request = build_chat_request(messages=messages, body=body, tools=all_tools)
    ctx = CompletionContext(
        config=_config,
        session=_session,
        router=_router,
        tool_registry=_tool_registry,
        model_name=model_name,
    )

    if chat_request.stream:
        return StreamingResponse(
            stream_with_fallback(
                ctx=ctx,
                chat_request=chat_request,
                chain=chain,
                key_info=key_info,
                task_type=task_type,
                start=start,
                client_tools=client_tools,
                req_id=req_id,
                endpoint=endpoint,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    response_data = await complete_with_fallback(
        ctx=ctx,
        chat_request=chat_request,
        chain=chain,
        key_info=key_info,
        task_type=task_type,
        start=start,
        client_tools=client_tools,
        req_id=req_id,
        endpoint=endpoint,
    )
    return JSONResponse(response_data)


def _is_first_turn(messages: list[dict]) -> bool:
    """
    Heuristic: treat as a brand-new session if we have no assistant/tool turns yet.
    Messages come in OpenAI-style dicts: {role, content}.
    """
    for msg in messages:
        role = msg.get("role")
        if role in ("assistant", "tool"):
            return False
    return True


def _enforce_api_key_profile(key_info: dict, path: str) -> None:
    """
    Enforce API key profile restrictions.
    JWT-authenticated users (id starts with 'jwt-') are not restricted here.
    """
    key_id = str(key_info.get("id", ""))
    if key_id.startswith("jwt-"):
        return

    profile = (key_info.get("profile") or "openai_compat").strip()

    # Legacy support: "universal" used to mean "works everywhere".
    # We no longer issue universal keys; allow but warn so users rotate.
    if profile == "universal":
        logger.warning("Legacy API key profile 'universal' used for %s — rotate this key", path)
        return

    # Gateway keys are only valid for the gateway (/v1) surface.
    if profile == "gateway":
        raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")

    # Regular Superintelligence keys (used with /superintelligence/v1/*).
    if profile == "openai_compat":
        # Allow only the regular surface (and its legacy prefix), not the coding surface.
        if path.startswith(f"{CODING_API_PREFIX}/"):
            raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")
        if path.startswith((f"{API_PREFIX}/", f"{LEGACY_API_PREFIX}/")):
            # Block legacy coding endpoints under the regular surface.
            if "/coding/" in path:
                raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")
            return
        raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")

    # Coding Superintelligence keys are restricted to coding surface + basic metadata.
    if profile == "coding_superintelligence":
        allowed_prefixes = (
            f"{CODING_API_PREFIX}/",
            f"{API_PREFIX}/coding/",
            f"{LEGACY_API_PREFIX}/coding/",
            f"{API_PREFIX}/health",
            f"{API_PREFIX}/models",
            f"{LEGACY_API_PREFIX}/health",
            f"{LEGACY_API_PREFIX}/models",
        )
        if not path.startswith(allowed_prefixes):
            raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")
        return

    # Unknown profile: deny by default.
    raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")


#
# Note: legacy fallback implementations were moved to
# `superintelligence/http/completion_service.py`.
