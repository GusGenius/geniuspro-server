"""GeniusPro Superintelligence v1 — Main Application

Routes chat completion requests to the best expert model via Macro-MoE.
"""

import asyncio
import json
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
from superintelligence.config import load_config, SuperintelligenceConfig
from superintelligence.providers.base import ChatMessage, ChatRequest, StreamChunk
from superintelligence.providers.openai_compat import OpenAICompatProvider
from superintelligence.providers.anthropic_provider import AnthropicProvider
from superintelligence.routing.router import SuperintelligenceRouter
from superintelligence.tools.registry import ToolRegistry
from superintelligence.tools.web_search import WebSearchTool
from superintelligence.tools.url_fetch import URLFetchTool
from superintelligence.tools.provider_mapper import ProviderToolMapper
from superintelligence.usage import log_usage

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("superintelligence")

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "geniuspro-superintelligence-v1"
API_PREFIX = "/super-intelligence/v1"
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
    req_id = uuid.uuid4().hex[:8]
    key_info = await _auth(request)
    start = time.time()

    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    client_tools = body.get("tools", [])  # Client-defined tools

    last_msg = messages[-1].get("content", "")[:80] if messages else ""
    logger.info("[%s] ── New request ── user=%s stream=%s msgs=%d last_msg='%s'",
                req_id, key_info.get("user_id", "?")[:8], stream, len(messages), last_msg)

    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Inject default system prompt if client didn't provide one
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    # Merge client tools with built-in tools
    all_tools = []
    if client_tools:
        all_tools.extend(client_tools)
    if _tool_registry:
        all_tools.extend(_tool_registry.get_definitions())

    # Classify the task
    task_type = _router.classify(messages)
    logger.info("[%s] Task classified as: %s", req_id, task_type.value)

    # Get fallback chain (best provider first, then alternatives)
    chain = _router.get_fallback_chain(task_type)
    if not chain:
        logger.error("[%s] No providers available for task_type=%s", req_id, task_type.value)
        raise HTTPException(status_code=503, detail="No providers available")

    chain_names = [p.name for p in chain]
    logger.info("[%s] Fallback chain: %s", req_id, " → ".join(chain_names))

    # Build the normalized request
    chat_request = ChatRequest(
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_tokens"),
        stop=body.get("stop"),
        stream=stream,
        tools=all_tools if all_tools else None,
        tool_choice=body.get("tool_choice", "auto" if all_tools else None),
    )

    if stream:
        return StreamingResponse(
            _stream_with_fallback(chat_request, chain, key_info, task_type, start, client_tools, req_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming with fallback
    return await _complete_with_fallback(
        chat_request, chain, key_info, task_type, start, client_tools, req_id
    )


# ─── Non-streaming with fallback ─────────────────────────────────────────────

async def _complete_with_fallback(
    chat_request: ChatRequest,
    chain: list,
    key_info: dict,
    task_type,
    start: float,
    client_tools: list,
    req_id: str = "",
) -> JSONResponse:
    """Try providers in order until one succeeds. Handles tool execution loop."""
    last_error = ""

    original_tools = list(chat_request.tools) if chat_request.tools else None

    for provider in chain:
        try:
            logger.info("[%s] Trying provider: %s (model=%s)", req_id, provider.name, provider.default_model)
            chat_request.model = provider.default_model
            
            # Convert tools to provider format (use copy so fallback gets original)
            if original_tools:
                chat_request.tools = ProviderToolMapper.to_provider_format(
                    original_tools, provider.name
                )
            else:
                chat_request.tools = None

            # Tool execution loop (max 5 iterations to prevent infinite loops)
            max_iterations = 5
            conversation_messages = list(chat_request.messages)
            final_content = ""
            final_tool_calls = None
            final_finish_reason = "stop"
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for iteration in range(max_iterations):
                logger.debug("[%s] Tool iteration %d/%d", req_id, iteration + 1, max_iterations)
                # Update messages for this iteration
                chat_request.messages = conversation_messages
                
                result = await provider.chat_completion(chat_request)
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                final_content = result.content
                final_tool_calls = result.tool_calls
                final_finish_reason = result.finish_reason

                # If no tool calls, we're done
                if not result.tool_calls:
                    break

                # Separate server-side and client-side tool calls
                server_tool_calls = []
                client_tool_calls = []
                client_tool_names = {tool.get("function", {}).get("name") for tool in client_tools if tool.get("type") == "function"}

                for tc in result.tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    if _tool_registry and _tool_registry.is_server_tool(tool_name):
                        # Check if provider handles it natively
                        if not ProviderToolMapper.is_native_tool(tool_name, provider.name):
                            server_tool_calls.append(tc)
                    elif tool_name in client_tool_names:
                        client_tool_calls.append(tc)

                # Execute server-side tools
                if server_tool_calls:
                    for tc in server_tool_calls:
                        tool_name = tc.get("function", {}).get("name", "")
                        tool_args_str = tc.get("function", {}).get("arguments", "{}")
                        try:
                            tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            tool_result = await _tool_registry.execute(tool_name, tool_args)
                            
                            # Add tool result to conversation
                            conversation_messages.append(ChatMessage(
                                role="assistant",
                                content="",  # Tool calls don't have text content
                            ))
                            conversation_messages.append(ChatMessage(
                                role="tool",
                                content=tool_result,
                            ))
                        except Exception as e:
                            logger.error("[%s] Tool '%s' execution error: %s", req_id, tool_name, e)
                            conversation_messages.append(ChatMessage(
                                role="tool",
                                content=f"Error: {str(e)}",
                            ))

                # If we have client tool calls, break and return them
                if client_tool_calls:
                    logger.info("[%s] Returning %d client tool calls", req_id, len(client_tool_calls))
                    final_tool_calls = client_tool_calls
                    break

            elapsed = int((time.time() - start) * 1000)

            logger.info("[%s] ── Complete ── provider=%s tokens=%d+%d time=%dms",
                        req_id, provider.name, total_prompt_tokens, total_completion_tokens, elapsed)

            # Log usage (fire-and-forget)
            asyncio.create_task(
                log_usage(
                    _session, _config.supabase_url, _config.supabase_service_key,
                    key_info["id"], key_info["user_id"],
                    MODEL_NAME,
                    f"{API_PREFIX}/chat/completions",
                    total_prompt_tokens, total_completion_tokens,
                    200, elapsed, provider.name,
                )
            )

            # Determine finish reason
            finish_reason = final_finish_reason
            if final_tool_calls:
                finish_reason = "tool_calls"

            response_data = {
                "id": f"si-{int(start * 1000)}",
                "object": "chat.completion",
                "created": int(start),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content,
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                },
            }

            # Add tool_calls if present
            if final_tool_calls:
                response_data["choices"][0]["message"]["tool_calls"] = final_tool_calls

            return JSONResponse(response_data)

        except Exception as e:
            last_error = str(e)
            logger.warning("[%s] Provider %s failed: %s", req_id, provider.name, last_error)
            if _router:
                _router.mark_unhealthy(provider.name)
            continue

    logger.error("[%s] All providers failed. Last error: %s", req_id, last_error)
    raise HTTPException(status_code=502, detail=f"All providers failed: {last_error}")


# ─── Streaming with fallback ─────────────────────────────────────────────────

async def _stream_with_fallback(
    chat_request: ChatRequest,
    chain: list,
    key_info: dict,
    task_type,
    start: float,
    client_tools: list,
    req_id: str = "",
):
    """Stream from the first working provider. Handles tool execution loop."""
    prompt_tokens = 0
    completion_tokens = 0
    provider_used = ""
    original_tools = list(chat_request.tools) if chat_request.tools else None

    for provider in chain:
        try:
            logger.info("[%s] Streaming via provider: %s (model=%s)", req_id, provider.name, provider.default_model)
            chat_request.model = provider.default_model
            
            # Convert tools to provider format (use copy so fallback gets original)
            if original_tools:
                chat_request.tools = ProviderToolMapper.to_provider_format(
                    original_tools, provider.name
                )
            else:
                chat_request.tools = None

            # Tool execution loop for streaming (simplified - tools handled on final chunk)
            conversation_messages = list(chat_request.messages)
            accumulated_content = ""
            accumulated_tool_calls = None
            iteration = 0
            max_iterations = 5
            should_continue_loop = False

            while iteration < max_iterations:
                chat_request.messages = conversation_messages
                iteration += 1
                should_continue_loop = False

                async for chunk in provider.stream_chat_completion(chat_request):
                    if chunk.prompt_tokens:
                        prompt_tokens = chunk.prompt_tokens
                    if chunk.completion_tokens:
                        completion_tokens = chunk.completion_tokens

                    # Accumulate content
                    if chunk.content:
                        accumulated_content += chunk.content

                    # Send content chunks
                    if chunk.content:
                        sse_data = {
                            "id": f"si-{int(start * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(start),
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk.content},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(sse_data)}\n\n"

                    # Handle tool calls on final chunk
                    if chunk.done:
                        if chunk.tool_calls_delta:
                            accumulated_tool_calls = chunk.tool_calls_delta

                        # Separate server-side and client-side tool calls
                        if accumulated_tool_calls:
                            server_tool_calls = []
                            client_tool_calls = []
                            client_tool_names = {tool.get("function", {}).get("name") for tool in client_tools if tool.get("type") == "function"}

                            for tc in accumulated_tool_calls:
                                tool_name = tc.get("function", {}).get("name", "")
                                if _tool_registry and _tool_registry.is_server_tool(tool_name):
                                    if not ProviderToolMapper.is_native_tool(tool_name, provider.name):
                                        server_tool_calls.append(tc)
                                elif tool_name in client_tool_names:
                                    client_tool_calls.append(tc)

                            # Execute server-side tools and continue loop
                            if server_tool_calls:
                                for tc in server_tool_calls:
                                    tool_name = tc.get("function", {}).get("name", "")
                                    tool_args_str = tc.get("function", {}).get("arguments", "{}")
                                    logger.info("[%s] Executing server tool: %s", req_id, tool_name)
                                    try:
                                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                                        tool_result = await _tool_registry.execute(tool_name, tool_args)
                                        
                                        conversation_messages.append(ChatMessage(role="assistant", content=accumulated_content))
                                        conversation_messages.append(ChatMessage(role="tool", content=tool_result))
                                        accumulated_content = ""  # Reset for next iteration
                                        accumulated_tool_calls = None
                                        should_continue_loop = True
                                    except Exception as e:
                                        logger.error("[%s] Tool '%s' execution error: %s", req_id, tool_name, e)

                                # Continue to next iteration if we executed server tools
                                if should_continue_loop:
                                    break

                            # If client tool calls, send them and finish
                            if client_tool_calls:
                                sse_data = {
                                    "id": f"si-{int(start * 1000)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(start),
                                    "model": MODEL_NAME,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"tool_calls": client_tool_calls},
                                        "finish_reason": "tool_calls",
                                    }],
                                }
                                yield f"data: {json.dumps(sse_data)}\n\n"
                                yield "data: [DONE]\n\n"
                                
                                elapsed = int((time.time() - start) * 1000)
                                logger.info("[%s] ── Stream complete (tool_calls) ── provider=%s time=%dms",
                                            req_id, provider.name, elapsed)
                                await log_usage(
                                    _session, _config.supabase_url, _config.supabase_service_key,
                                    key_info["id"], key_info["user_id"],
                                    MODEL_NAME,
                                    f"{API_PREFIX}/chat/completions",
                                    prompt_tokens, completion_tokens,
                                    200, elapsed, provider_used,
                                )
                                return

                        # No tool calls, finish normally
                        sse_data = {
                            "id": f"si-{int(start * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(start),
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": chunk.finish_reason or "stop",
                            }],
                        }
                        yield f"data: {json.dumps(sse_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        
                        elapsed = int((time.time() - start) * 1000)
                        logger.info("[%s] ── Stream complete ── provider=%s tokens=%d+%d time=%dms",
                                    req_id, provider.name, prompt_tokens, completion_tokens, elapsed)
                        await log_usage(
                            _session, _config.supabase_url, _config.supabase_service_key,
                            key_info["id"], key_info["user_id"],
                            MODEL_NAME,
                            f"{API_PREFIX}/chat/completions",
                            prompt_tokens, completion_tokens,
                            200, elapsed, provider_used,
                        )
                        return  # Success — don't try next provider

                # If we should continue loop (server tools executed), continue
                if should_continue_loop:
                    continue
                
                # Otherwise, we completed an iteration without tool calls
                break

            # If we exhausted iterations, finish
            if iteration >= max_iterations:
                logger.warning("[%s] Max tool iterations (%d) reached", req_id, max_iterations)
                yield "data: [DONE]\n\n"
                elapsed = int((time.time() - start) * 1000)
                await log_usage(
                    _session, _config.supabase_url, _config.supabase_service_key,
                    key_info["id"], key_info["user_id"],
                    MODEL_NAME,
                    f"{API_PREFIX}/chat/completions",
                    prompt_tokens, completion_tokens,
                    200, elapsed, provider_used,
                )
                return

        except Exception as e:
            logger.warning("[%s] Stream provider %s failed: %s", req_id, provider.name, e)
            if _router:
                _router.mark_unhealthy(provider.name)
            continue

    logger.error("[%s] All stream providers failed", req_id)
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
