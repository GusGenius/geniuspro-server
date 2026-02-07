"""
GeniusPro Superintelligence v1 — Main Application

The Macro-MoE model server. Receives chat completion requests,
classifies the task, routes to the best expert model, and returns
a unified response under model name 'geniuspro-superintelligence-v1'.

Endpoints:
  POST /super-intelligence/v1/chat/completions
  GET  /super-intelligence/v1/models
  GET  /super-intelligence/v1/health
"""

import asyncio
import json
import time
from typing import Optional

import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from superintelligence.agents.swarm import AgentSwarm
from superintelligence.auth import require_auth
from superintelligence.config import load_config, SuperintelligenceConfig
from superintelligence.providers.base import ChatMessage, ChatRequest, StreamChunk
from superintelligence.providers.openai_compat import OpenAICompatProvider
from superintelligence.providers.anthropic_provider import AnthropicProvider
from superintelligence.routing.router import SuperintelligenceRouter
from superintelligence.usage import log_usage

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "geniuspro-superintelligence-v1"
API_PREFIX = "/super-intelligence/v1"

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


@app.on_event("startup")
async def startup() -> None:
    global _config, _session, _router, _swarm

    print("Starting GeniusPro Superintelligence v1...")
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

    # Start the Agent Swarm
    _swarm = AgentSwarm(_router)
    await _swarm.start()

    print(f"GeniusPro Superintelligence v1 ready — {provider_count} providers")


@app.on_event("shutdown")
async def shutdown() -> None:
    global _session, _swarm
    if _swarm:
        await _swarm.stop()
    if _session:
        await _session.close()
    print("Superintelligence v1 shut down.")


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
async def health() -> dict:
    provider_names = list(_router.providers.keys()) if _router else []
    return {
        "status": "ok",
        "service": "geniuspro-superintelligence-v1",
        "providers": len(provider_names),
    }


# ─── Models ───────────────────────────────────────────────────────────────────

@app.get(f"{API_PREFIX}/models")
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


# ─── Chat Completions ────────────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/chat/completions", response_model=None)
async def chat_completions(request: Request):
    key_info = await _auth(request)
    start = time.time()

    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Classify the task
    task_type = _router.classify(messages)

    # Get fallback chain (best provider first, then alternatives)
    chain = _router.get_fallback_chain(task_type)
    if not chain:
        raise HTTPException(status_code=503, detail="No providers available")

    # Build the normalized request
    chat_request = ChatRequest(
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_tokens"),
        stop=body.get("stop"),
        stream=stream,
    )

    if stream:
        return StreamingResponse(
            _stream_with_fallback(chat_request, chain, key_info, task_type, start),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming with fallback
    return await _complete_with_fallback(
        chat_request, chain, key_info, task_type, start
    )


# ─── Non-streaming with fallback ─────────────────────────────────────────────

async def _complete_with_fallback(
    chat_request: ChatRequest,
    chain: list,
    key_info: dict,
    task_type,
    start: float,
) -> JSONResponse:
    """Try providers in order until one succeeds."""
    last_error = ""

    for provider in chain:
        try:
            chat_request.model = provider.default_model
            result = await provider.chat_completion(chat_request)
            elapsed = int((time.time() - start) * 1000)

            # Log usage (fire-and-forget)
            asyncio.create_task(
                log_usage(
                    _session, _config.supabase_url, _config.supabase_service_key,
                    key_info["id"], key_info["user_id"],
                    MODEL_NAME,
                    f"{API_PREFIX}/chat/completions",
                    result.prompt_tokens, result.completion_tokens,
                    200, elapsed, provider.name,
                )
            )

            return JSONResponse({
                "id": f"si-{int(start * 1000)}",
                "object": "chat.completion",
                "created": int(start),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            })

        except Exception as e:
            last_error = str(e)
            print(f"Provider {provider.name} failed: {last_error}")
            if _router:
                _router.mark_unhealthy(provider.name)
            continue

    raise HTTPException(status_code=502, detail=f"All providers failed: {last_error}")


# ─── Streaming with fallback ─────────────────────────────────────────────────

async def _stream_with_fallback(
    chat_request: ChatRequest,
    chain: list,
    key_info: dict,
    task_type,
    start: float,
):
    """Stream from the first working provider. Fallback on connection error."""
    prompt_tokens = 0
    completion_tokens = 0
    provider_used = ""

    for provider in chain:
        try:
            chat_request.model = provider.default_model
            provider_used = provider.name

            async for chunk in provider.stream_chat_completion(chat_request):
                if chunk.prompt_tokens:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens:
                    completion_tokens = chunk.completion_tokens

                sse_data = {
                    "id": f"si-{int(start * 1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(start),
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.content} if chunk.content else {},
                        "finish_reason": chunk.finish_reason if chunk.done else None,
                    }],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"

                if chunk.done:
                    yield "data: [DONE]\n\n"
                    break

            # Log usage after stream completes
            elapsed = int((time.time() - start) * 1000)
            await log_usage(
                _session, _config.supabase_url, _config.supabase_service_key,
                key_info["id"], key_info["user_id"],
                MODEL_NAME,
                f"{API_PREFIX}/chat/completions",
                prompt_tokens, completion_tokens,
                200, elapsed, provider_used,
            )
            return  # Success — don't try next provider

        except Exception as e:
            print(f"Stream provider {provider.name} failed: {e}")
            if _router:
                _router.mark_unhealthy(provider.name)
            continue

    # All providers failed
    error_chunk = {
        "id": f"si-{int(start * 1000)}",
        "object": "chat.completion.chunk",
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {"content": "Error: All providers unavailable. Please try again."},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(error_chunk)}\n\n"
    yield "data: [DONE]\n\n"
