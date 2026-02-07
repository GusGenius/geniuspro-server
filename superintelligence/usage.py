"""
GeniusPro Superintelligence v1 — Usage Logging

Logs every request to Supabase usage_logs table for billing and analytics.
Tracks provider used internally but never exposes it to the client.
"""

import traceback

import aiohttp


def _supa_headers(service_key: str) -> dict[str, str]:
    """Build Supabase REST headers."""
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


async def log_usage(
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
    api_key_id: str,
    user_id: str,
    model: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    status_code: int,
    response_time_ms: int,
    provider_used: str = "",
) -> None:
    """
    Log a request to Supabase usage_logs.

    The model field always shows 'geniuspro-superintelligence-v1'.
    The provider_used field is internal-only metadata.
    """
    url = f"{supabase_url}/rest/v1/usage_logs"
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
        async with session.post(
            url,
            headers=_supa_headers(supabase_service_key),
            json=payload,
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                print(f"Usage log failed ({resp.status}): {text}")
    except Exception:
        traceback.print_exc()
