"""
GeniusPro API Gateway
Authenticates API keys against Supabase, proxies to Ollama + Voice,
logs usage for billing.

Runs on port 8000. Nginx routes all traffic here.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from typing import Optional

import aiohttp
from fastapi import FastAPI, Request, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gateway")

# ─── Config ───────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://orajwuisgwffnrbjasaj.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VOICE_SERVER_URL = os.environ.get("VOICE_SERVER_URL", "http://localhost:8001")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="GeniusPro API Gateway", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reusable HTTP session (created at startup)
_http_session: Optional[aiohttp.ClientSession] = None


@app.on_event("startup")
async def startup():
    global _http_session
    _http_session = aiohttp.ClientSession()
    if not SUPABASE_SERVICE_KEY:
        logger.warning("SUPABASE_SERVICE_KEY not set — auth will fail")
    logger.info("GeniusPro API Gateway ready on :8000")


@app.on_event("shutdown")
async def shutdown():
    global _http_session
    if _http_session:
        await _http_session.close()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _supa_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def extract_api_key(request: Request) -> Optional[str]:
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ─── Key Validation (cached per-request) ─────────────────────────────────────

# Simple in-memory cache: hash -> (key_info, timestamp)
_key_cache: dict[str, tuple[dict, float]] = {}
KEY_CACHE_TTL = 60  # seconds


async def validate_key(raw_key: str) -> Optional[dict]:
    key_h = hash_key(raw_key)

    # Check cache
    if key_h in _key_cache:
        info, ts = _key_cache[key_h]
        if time.time() - ts < KEY_CACHE_TTL:
            return info

    url = (
        f"{SUPABASE_URL}/rest/v1/api_keys"
        f"?key_hash=eq.{key_h}&is_active=eq.true"
        f"&select=id,user_id,profile,rate_limit_rpm,rate_limit_tpm"
    )
    async with _http_session.get(url, headers=_supa_headers()) as resp:
        if resp.status != 200:
            return None
        rows = await resp.json()
        if not rows:
            return None

    info = rows[0]
    _key_cache[key_h] = (info, time.time())

    # Update last_used_at (fire-and-forget)
    patch_url = f"{SUPABASE_URL}/rest/v1/api_keys?id=eq.{info['id']}"
    asyncio.create_task(
        _patch(patch_url, {"last_used_at": "now()"})
    )
    return info


async def _patch(url, payload):
    try:
        async with _http_session.patch(url, headers=_supa_headers(), json=payload) as r:
            pass
    except Exception:
        pass


async def log_usage(
    api_key_id: str,
    user_id: str,
    model: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    status_code: int,
    response_time_ms: int,
):
    url = f"{SUPABASE_URL}/rest/v1/usage_logs"
    payload = {
        "api_key_id": api_key_id,
        "user_id": user_id,
        "model": model,
        "endpoint": endpoint,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
    }
    try:
        async with _http_session.post(url, headers=_supa_headers(), json=payload) as r:
            pass
    except Exception:
        traceback.print_exc()


# ─── Auth Dependency ──────────────────────────────────────────────────────────

async def require_auth(request: Request) -> dict:
    raw_key = extract_api_key(request)
    if not raw_key:
        logger.warning("Auth failed: no API key provided from %s", request.client.host if request.client else "?")
        raise HTTPException(
            status_code=401,
            detail="API key required. Use X-API-Key header or Authorization: Bearer.",
        )
    key_info = await validate_key(raw_key)
    if not key_info:
        logger.warning("Auth failed: invalid key from %s", request.client.host if request.client else "?")
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
    logger.info("Auth OK: user=%s key=%s", key_info.get("user_id", "?")[:8], key_info.get("id", "?")[:8])
    _enforce_api_key_profile(key_info, request.url.path)
    return key_info


def _enforce_api_key_profile(key_info: dict, path: str) -> None:
    profile = (key_info.get("profile") or "openai_compat").strip()
    # Legacy support: "universal" used to mean "works everywhere".
    # We no longer issue universal keys; treat them as allowed here but encourage rotation.
    if profile == "universal":
        logger.warning("Legacy API key profile 'universal' used for %s — rotate this key", path)
        return
    if profile == "openai_compat":
        return
    # Coding Superintelligence keys are not valid for the gateway /v1 surface.
    if profile == "coding_superintelligence":
        raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")
    raise HTTPException(status_code=403, detail="API key profile does not allow this endpoint")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "geniuspro-gateway"}


# ─── Models ───────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models(key_info: dict = Depends(require_auth)):
    return {
        "object": "list",
        "data": [
            {
                "id": "geniuspro-coder-v1",
                "object": "model",
                "owned_by": "geniuspro",
            },
            {
                "id": "geniuspro-voice",
                "object": "model",
                "owned_by": "geniuspro",
            },
        ],
    }


# ─── Chat Completions (OpenAI-compatible) ────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, key_info: dict = Depends(require_auth)
):
    start = time.time()
    body = await request.json()

    model = body.get("model", "geniuspro-coder-v1")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    logger.info("Gateway /v1/chat/completions model=%s stream=%s msgs=%d user=%s",
                model, stream, len(messages), key_info.get("user_id", "?")[:8])

    # Build Ollama payload
    ollama_payload = {"model": model, "messages": messages, "stream": stream}

    # Map OpenAI params to Ollama options
    options = {}
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "stop" in body:
        options["stop"] = body["stop"]
    if options:
        ollama_payload["options"] = options

    if stream:
        return StreamingResponse(
            _stream_chat(ollama_payload, key_info, model, start),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Non-streaming
    async with _http_session.post(
        f"{OLLAMA_URL}/api/chat", json=ollama_payload
    ) as resp:
        result = await resp.json()

    content = result.get("message", {}).get("content", "")
    prompt_tokens = result.get("prompt_eval_count", 0) or 0
    completion_tokens = result.get("eval_count", 0) or 0
    elapsed = int((time.time() - start) * 1000)

    logger.info("Gateway complete: model=%s tokens=%d+%d time=%dms", model, prompt_tokens, completion_tokens, elapsed)

    asyncio.create_task(
        log_usage(
            key_info["id"],
            key_info["user_id"],
            model,
            "/v1/chat/completions",
            prompt_tokens,
            completion_tokens,
            200,
            elapsed,
        )
    )

    return {
        "id": f"chatcmpl-gp-{int(start)}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_chat(ollama_payload, key_info, model, start):
    prompt_tokens = 0
    completion_tokens = 0

    async with _http_session.post(
        f"{OLLAMA_URL}/api/chat", json=ollama_payload
    ) as resp:
        async for line in resp.content:
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = data.get("message", {}).get("content", "")
            done = data.get("done", False)

            if done:
                prompt_tokens = data.get("prompt_eval_count", 0) or 0
                completion_tokens = data.get("eval_count", 0) or 0

            chunk = {
                "id": f"chatcmpl-gp-{int(start)}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content} if content else {},
                        "finish_reason": "stop" if done else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if done:
                yield "data: [DONE]\n\n"

    elapsed = int((time.time() - start) * 1000)
    await log_usage(
        key_info["id"],
        key_info["user_id"],
        model,
        "/v1/chat/completions",
        prompt_tokens,
        completion_tokens,
        200,
        elapsed,
    )


# ─── Voice WebSocket ─────────────────────────────────────────────────────────

async def validate_jwt_token(token: str) -> Optional[dict]:
    """Validate Supabase JWT token and return user info."""
    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {token}",
    }
    try:
        async with _http_session.get(url, headers=headers) as resp:
            if resp.status != 200:
                return None
            user_data = await resp.json()
            return {
                "id": f"jwt-{user_data.get('id')}",  # Fake API key ID for logging
                "user_id": user_data.get("id"),
                "rate_limit_rpm": 1000,  # Default limits for JWT users
                "rate_limit_tpm": 100000,
            }
    except Exception:
        return None


@app.websocket("/v1/voice")
async def voice_proxy(ws: WebSocket):
    """
    Auth then bidirectional proxy to voice server.
    Supports:
    - API key: /v1/voice?api_key=sk-gp-...
    - JWT token: /v1/voice?token=eyJ...
    - Auth message: {"type": "auth", "api_key": "..."} or {"type": "auth", "token": "..."}
    """
    await ws.accept()
    logger.info("Voice WS connected from %s", ws.client.host if ws.client else "?")

    # Try query params
    raw_key = ws.query_params.get("api_key")
    token = ws.query_params.get("token")

    if not raw_key and not token:
        # Wait for auth message
        try:
            first_msg = await asyncio.wait_for(ws.receive_text(), timeout=10)
            data = json.loads(first_msg)
            if data.get("type") == "auth":
                raw_key = data.get("api_key")
                token = data.get("token")
        except Exception:
            pass

    # Validate auth
    key_info = None
    if token:
        key_info = await validate_jwt_token(token)
    elif raw_key:
        key_info = await validate_key(raw_key)

    if not key_info:
        logger.warning("Voice WS auth failed — closing")
        await ws.send_text(json.dumps({"type": "error", "message": "Authentication required"}))
        await ws.close(code=1008)
        return

    logger.info("Voice WS authenticated: user=%s", key_info.get("user_id", "?")[:8])
    # Proxy to backend voice server
    voice_ws_url = (
        VOICE_SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")
        + "/v1/voice"
    )
    start = time.time()

    try:
        async with _http_session.ws_connect(voice_ws_url) as backend_ws:

            async def client_to_backend():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if "bytes" in msg and msg["bytes"]:
                            await backend_ws.send_bytes(msg["bytes"])
                        elif "text" in msg and msg["text"]:
                            await backend_ws.send_str(msg["text"])
                except (WebSocketDisconnect, Exception):
                    pass

            async def backend_to_client():
                try:
                    async for msg in backend_ws:
                        if ws.client_state != WebSocketState.CONNECTED:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await ws.send_text(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await ws.send_bytes(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                except Exception:
                    pass

            await asyncio.gather(client_to_backend(), backend_to_client())

    except Exception as exc:
        logger.error("Voice WS proxy error: %s", exc, exc_info=True)
        try:
            await ws.send_text(
                json.dumps({"type": "error", "message": "Voice server unavailable"})
            )
        except Exception:
            pass
    finally:
        elapsed = int((time.time() - start) * 1000)
        logger.info("Voice WS session ended: user=%s duration=%dms", key_info.get("user_id", "?")[:8], elapsed)
        await log_usage(
            key_info["id"],
            key_info["user_id"],
            "geniuspro-voice",
            "/v1/voice",
            0,
            0,
            200,
            elapsed,
        )
