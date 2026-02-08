"""
GeniusPro Superintelligence — Completion service

Shared non-streaming + streaming execution with:
- provider fallback
- tool execution loop (server tools + pass-through client tools)
- usage logging

Kept separate from app.py to satisfy the 500-line rule.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
from fastapi import HTTPException

from superintelligence.config import SuperintelligenceConfig
from superintelligence.providers.base import ChatMessage, ChatRequest, StreamChunk
from superintelligence.routing.classifier import TaskType
from superintelligence.routing.router import SuperintelligenceRouter
from superintelligence.tools.provider_mapper import ProviderToolMapper
from superintelligence.tools.registry import ToolRegistry
from superintelligence.usage import log_usage

logger = logging.getLogger("superintelligence.completions")


@dataclass(frozen=True)
class CompletionContext:
    config: SuperintelligenceConfig
    session: aiohttp.ClientSession
    router: SuperintelligenceRouter
    tool_registry: Optional[ToolRegistry]
    model_name: str


def inject_system_message(messages: list[dict], content: str) -> None:
    """
    Insert a system message after the first system message if present,
    otherwise at the beginning.
    """
    if not messages:
        messages.insert(0, {"role": "system", "content": content})
        return

    if messages[0].get("role") == "system":
        messages.insert(1, {"role": "system", "content": content})
    else:
        messages.insert(0, {"role": "system", "content": content})


def build_chat_request(
    *,
    messages: list[dict],
    body: dict,
    tools: Optional[list[dict]],
) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_tokens"),
        stop=body.get("stop"),
        stream=bool(body.get("stream", False)),
        tools=tools if tools else None,
        tool_choice=body.get("tool_choice", "auto" if tools else None),
    )


def merge_tools(
    *,
    client_tools: list[dict],
    tool_registry: Optional[ToolRegistry],
) -> list[dict]:
    all_tools: list[dict] = []
    if client_tools:
        all_tools.extend(client_tools)
    if tool_registry:
        all_tools.extend(tool_registry.get_definitions())
    return all_tools


async def complete_with_fallback(
    *,
    ctx: CompletionContext,
    chat_request: ChatRequest,
    chain: list,
    key_info: dict,
    task_type: TaskType,
    start: float,
    client_tools: list[dict],
    req_id: str,
    endpoint: str,
):
    """Try providers in order until one succeeds. Handles tool execution loop."""
    last_error = ""
    original_tools = list(chat_request.tools) if chat_request.tools else None

    for provider in chain:
        try:
            logger.info("[%s] Trying provider: %s (model=%s)", req_id, provider.name, provider.default_model)
            chat_request.model = provider.default_model

            # Convert tools to provider format (copy so fallback can reuse originals)
            if original_tools:
                chat_request.tools = ProviderToolMapper.to_provider_format(original_tools, provider.name)
            else:
                chat_request.tools = None

            max_iterations = 5
            conversation_messages = list(chat_request.messages)
            final_content = ""
            final_tool_calls = None
            final_finish_reason = "stop"
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for iteration in range(max_iterations):
                logger.debug("[%s] Tool iteration %d/%d", req_id, iteration + 1, max_iterations)
                chat_request.messages = conversation_messages

                result = await provider.chat_completion(chat_request)
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                final_content = result.content
                final_tool_calls = result.tool_calls
                final_finish_reason = result.finish_reason

                if not result.tool_calls:
                    break

                server_tool_calls: list[dict] = []
                client_tool_calls: list[dict] = []
                client_tool_names = {
                    tool.get("function", {}).get("name")
                    for tool in client_tools
                    if tool.get("type") == "function"
                }

                for tc in result.tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    if ctx.tool_registry and ctx.tool_registry.is_server_tool(tool_name):
                        # Skip tools that provider executes natively (eg Anthropic web_search/web_fetch)
                        if not ProviderToolMapper.is_native_tool(tool_name, provider.name):
                            server_tool_calls.append(tc)
                    elif tool_name in client_tool_names:
                        client_tool_calls.append(tc)

                # Execute server-side tools
                if server_tool_calls and ctx.tool_registry:
                    for tc in server_tool_calls:
                        tool_name = tc.get("function", {}).get("name", "")
                        tool_args_str = tc.get("function", {}).get("arguments", "{}")
                        try:
                            tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            tool_result = await ctx.tool_registry.execute(tool_name, tool_args)

                            conversation_messages.append(ChatMessage(role="assistant", content=""))
                            conversation_messages.append(ChatMessage(role="tool", content=tool_result))
                        except Exception as e:
                            logger.error("[%s] Tool '%s' execution error: %s", req_id, tool_name, e)
                            conversation_messages.append(ChatMessage(role="tool", content=f"Error: {str(e)}"))

                # If we have client tool calls, return them
                if client_tool_calls:
                    logger.info("[%s] Returning %d client tool calls", req_id, len(client_tool_calls))
                    final_tool_calls = client_tool_calls
                    break

            elapsed = int((time.time() - start) * 1000)
            logger.info(
                "[%s] ── Complete ── provider=%s tokens=%d+%d time=%dms",
                req_id,
                provider.name,
                total_prompt_tokens,
                total_completion_tokens,
                elapsed,
            )

            asyncio.create_task(
                log_usage(
                    ctx.session,
                    ctx.config.supabase_url,
                    ctx.config.supabase_service_key,
                    key_info["id"],
                    key_info["user_id"],
                    ctx.model_name,
                    endpoint,
                    total_prompt_tokens,
                    total_completion_tokens,
                    200,
                    elapsed,
                    provider.name,
                )
            )

            finish_reason = final_finish_reason
            if final_tool_calls:
                finish_reason = "tool_calls"

            response_data = {
                "id": f"si-{int(start * 1000)}",
                "object": "chat.completion",
                "created": int(start),
                "model": ctx.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final_content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                },
            }

            if final_tool_calls:
                response_data["choices"][0]["message"]["tool_calls"] = final_tool_calls

            return response_data

        except Exception as e:
            last_error = str(e)
            logger.warning("[%s] Provider %s failed: %s", req_id, provider.name, last_error)
            ctx.router.mark_unhealthy(provider.name)
            continue

    logger.error("[%s] All providers failed. Last error: %s", req_id, last_error)
    raise HTTPException(status_code=502, detail=f"All providers failed: {last_error}")


async def stream_with_fallback(
    *,
    ctx: CompletionContext,
    chat_request: ChatRequest,
    chain: list,
    key_info: dict,
    task_type: TaskType,
    start: float,
    client_tools: list[dict],
    req_id: str,
    endpoint: str,
) -> AsyncIterator[str]:
    """Stream from the first working provider. Handles server tools in a loop."""
    original_tools = list(chat_request.tools) if chat_request.tools else None

    for provider in chain:
        try:
            logger.info("[%s] Streaming via provider: %s (model=%s)", req_id, provider.name, provider.default_model)
            provider_used = provider.name
            chat_request.model = provider.default_model

            if original_tools:
                chat_request.tools = ProviderToolMapper.to_provider_format(original_tools, provider.name)
            else:
                chat_request.tools = None

            conversation_messages = list(chat_request.messages)
            accumulated_content = ""
            accumulated_tool_calls = None
            iteration = 0
            max_iterations = 5

            prompt_tokens = 0
            completion_tokens = 0

            while iteration < max_iterations:
                chat_request.messages = conversation_messages
                iteration += 1

                should_continue_loop = False

                async for chunk in provider.stream_chat_completion(chat_request):
                    if chunk.prompt_tokens:
                        prompt_tokens = chunk.prompt_tokens
                    if chunk.completion_tokens:
                        completion_tokens = chunk.completion_tokens

                    if chunk.content:
                        accumulated_content += chunk.content

                    if chunk.content:
                        sse_data = {
                            "id": f"si-{int(start * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(start),
                            "model": ctx.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(sse_data)}\n\n"

                    if chunk.done:
                        if chunk.tool_calls_delta:
                            accumulated_tool_calls = chunk.tool_calls_delta

                        if accumulated_tool_calls:
                            server_tool_calls: list[dict] = []
                            client_tool_calls: list[dict] = []
                            client_tool_names = {
                                tool.get("function", {}).get("name")
                                for tool in client_tools
                                if tool.get("type") == "function"
                            }

                            for tc in accumulated_tool_calls:
                                tool_name = tc.get("function", {}).get("name", "")
                                if ctx.tool_registry and ctx.tool_registry.is_server_tool(tool_name):
                                    if not ProviderToolMapper.is_native_tool(tool_name, provider.name):
                                        server_tool_calls.append(tc)
                                elif tool_name in client_tool_names:
                                    client_tool_calls.append(tc)

                            # Execute server tools and continue loop
                            if server_tool_calls and ctx.tool_registry:
                                for tc in server_tool_calls:
                                    tool_name = tc.get("function", {}).get("name", "")
                                    tool_args_str = tc.get("function", {}).get("arguments", "{}")
                                    logger.info("[%s] Executing server tool: %s", req_id, tool_name)
                                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                                    tool_result = await ctx.tool_registry.execute(tool_name, tool_args)

                                    conversation_messages.append(ChatMessage(role="assistant", content=accumulated_content))
                                    conversation_messages.append(ChatMessage(role="tool", content=tool_result))

                                accumulated_content = ""
                                accumulated_tool_calls = None
                                should_continue_loop = True
                                break

                            # If client tool calls, send them and finish
                            if client_tool_calls:
                                sse_data = {
                                    "id": f"si-{int(start * 1000)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(start),
                                    "model": ctx.model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"tool_calls": client_tool_calls},
                                            "finish_reason": "tool_calls",
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(sse_data)}\n\n"
                                yield "data: [DONE]\n\n"

                                elapsed = int((time.time() - start) * 1000)
                                await log_usage(
                                    ctx.session,
                                    ctx.config.supabase_url,
                                    ctx.config.supabase_service_key,
                                    key_info["id"],
                                    key_info["user_id"],
                                    ctx.model_name,
                                    endpoint,
                                    prompt_tokens,
                                    completion_tokens,
                                    200,
                                    elapsed,
                                    provider_used,
                                )
                                return

                        # No tool calls: finish normally
                        sse_data = {
                            "id": f"si-{int(start * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(start),
                            "model": ctx.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": chunk.finish_reason or "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(sse_data)}\n\n"
                        yield "data: [DONE]\n\n"

                        elapsed = int((time.time() - start) * 1000)
                        await log_usage(
                            ctx.session,
                            ctx.config.supabase_url,
                            ctx.config.supabase_service_key,
                            key_info["id"],
                            key_info["user_id"],
                            ctx.model_name,
                            endpoint,
                            prompt_tokens,
                            completion_tokens,
                            200,
                            elapsed,
                            provider_used,
                        )
                        return

                if should_continue_loop:
                    continue

                break

            logger.warning("[%s] Max tool iterations (%d) reached", req_id, max_iterations)
            yield "data: [DONE]\n\n"
            elapsed = int((time.time() - start) * 1000)
            await log_usage(
                ctx.session,
                ctx.config.supabase_url,
                ctx.config.supabase_service_key,
                key_info["id"],
                key_info["user_id"],
                ctx.model_name,
                endpoint,
                prompt_tokens,
                completion_tokens,
                200,
                elapsed,
                provider_used,
            )
            return

        except Exception as e:
            logger.warning("[%s] Stream provider %s failed: %s", req_id, provider.name, e)
            ctx.router.mark_unhealthy(provider.name)
            continue

    logger.error("[%s] All stream providers failed", req_id)
    error_chunk = {
        "id": f"si-{int(start * 1000)}",
        "object": "chat.completion.chunk",
        "model": ctx.model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Error: All providers unavailable. Please try again."},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(error_chunk)}\n\n"
    yield "data: [DONE]\n\n"

