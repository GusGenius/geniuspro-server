"""
GeniusPro Superintelligence v1 — Authentication

API key validation against Supabase. Reuses the same pattern as the
existing gateway: SHA-256 hash lookup with in-memory cache.
"""

import hashlib
import time
import traceback
from typing import Optional

import aiohttp
from fastapi import Request, HTTPException


# In-memory cache: hash -> (key_info, timestamp)
_key_cache: dict[str, tuple[dict, float]] = {}
KEY_CACHE_TTL = 60  # seconds


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key for Supabase lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from X-API-Key header or Authorization: Bearer."""
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


async def validate_key(
    raw_key: str,
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
) -> Optional[dict]:
    """Validate an API key against Supabase with caching."""
    key_h = hash_key(raw_key)

    # Check cache
    if key_h in _key_cache:
        info, ts = _key_cache[key_h]
        if time.time() - ts < KEY_CACHE_TTL:
            return info

    headers = {
        "apikey": supabase_service_key,
        "Authorization": f"Bearer {supabase_service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    url = (
        f"{supabase_url}/rest/v1/api_keys"
        f"?key_hash=eq.{key_h}&is_active=eq.true"
        f"&select=id,user_id,rate_limit_rpm,rate_limit_tpm"
    )

    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                return None
            rows = await resp.json()
            if not rows:
                return None
    except Exception:
        traceback.print_exc()
        return None

    info = rows[0]
    _key_cache[key_h] = (info, time.time())
    return info


async def require_auth(
    request: Request,
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
) -> dict:
    """FastAPI dependency: require valid API key or raise 401."""
    raw_key = extract_api_key(request)
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Use X-API-Key header or Authorization: Bearer.",
        )
    key_info = await validate_key(raw_key, session, supabase_url, supabase_service_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
    return key_info
