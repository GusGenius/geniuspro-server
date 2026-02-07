"""
GeniusPro Superintelligence v1 — Authentication

Supports two auth methods:
  1. API key (X-API-Key header or Authorization: Bearer sk-gp-...)
     → SHA-256 hash lookup against Supabase api_keys table
  2. JWT token (Authorization: Bearer eyJ...)
     → Validated against Supabase Auth /auth/v1/user endpoint

API keys are for external/programmatic use.
JWT tokens are for the chat dashboard (logged-in users).
"""

import hashlib
import logging
import time
import traceback
from typing import Optional

import aiohttp
from fastapi import Request, HTTPException

logger = logging.getLogger("superintelligence.auth")


# In-memory cache: hash -> (key_info, timestamp)
_key_cache: dict[str, tuple[dict, float]] = {}
KEY_CACHE_TTL = 60  # seconds

# JWT user cache: token_hash -> (user_info, timestamp)
_jwt_cache: dict[str, tuple[dict, float]] = {}
JWT_CACHE_TTL = 300  # 5 minutes


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key for Supabase lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _is_jwt(token: str) -> bool:
    """Check if a token looks like a JWT (eyJ...) vs an API key (sk-gp-...)."""
    return token.startswith("eyJ")


def extract_bearer(request: Request) -> Optional[str]:
    """Extract token from X-API-Key header or Authorization: Bearer."""
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
            logger.debug("API key cache hit (user=%s)", info.get("user_id", "?")[:8])
            return info

    logger.debug("API key cache miss — querying Supabase")
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
                logger.warning("Supabase API key lookup returned %d", resp.status)
                return None
            rows = await resp.json()
            if not rows:
                logger.info("API key not found or inactive")
                return None
    except Exception:
        logger.error("API key validation error", exc_info=True)
        return None

    info = rows[0]
    _key_cache[key_h] = (info, time.time())
    logger.info("API key validated: user=%s key_id=%s", info.get("user_id", "?")[:8], info.get("id", "?")[:8])
    return info


async def validate_jwt(
    token: str,
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
) -> Optional[dict]:
    """Validate a Supabase JWT token and return user info."""
    token_h = hashlib.sha256(token.encode()).hexdigest()

    # Check cache
    if token_h in _jwt_cache:
        info, ts = _jwt_cache[token_h]
        if time.time() - ts < JWT_CACHE_TTL:
            logger.debug("JWT cache hit (user=%s)", info.get("user_id", "?")[:8])
            return info

    logger.debug("JWT cache miss — validating with Supabase Auth")
    headers = {
        "apikey": supabase_service_key,
        "Authorization": f"Bearer {token}",
    }

    try:
        async with session.get(
            f"{supabase_url}/auth/v1/user", headers=headers
        ) as resp:
            if resp.status != 200:
                logger.warning("JWT validation failed: Supabase returned %d", resp.status)
                return None
            user_data = await resp.json()
            user_id = user_data.get("id")
            if not user_id:
                logger.warning("JWT valid but no user_id in response")
                return None

            info = {
                "id": f"jwt-{user_id}",
                "user_id": user_id,
                "rate_limit_rpm": 120,
                "rate_limit_tpm": 100000,
            }
            _jwt_cache[token_h] = (info, time.time())
            logger.info("JWT validated: user=%s", user_id[:8])
            return info
    except Exception:
        logger.error("JWT validation error", exc_info=True)
        return None


async def require_auth(
    request: Request,
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
) -> dict:
    """Require valid API key or JWT token, or raise 401."""
    token = extract_bearer(request)
    if not token:
        logger.warning("No auth token provided from %s %s",
                        request.method, request.url.path)
        raise HTTPException(
            status_code=401,
            detail="Auth required. Use X-API-Key or Authorization: Bearer.",
        )

    auth_method = "JWT" if _is_jwt(token) else "API key"
    logger.debug("Authenticating via %s for %s %s", auth_method, request.method, request.url.path)

    # JWT tokens (from chat dashboard)
    if _is_jwt(token):
        user_info = await validate_jwt(
            token, session, supabase_url, supabase_service_key
        )
        if not user_info:
            logger.warning("JWT auth failed for %s %s", request.method, request.url.path)
            raise HTTPException(
                status_code=401, detail="Invalid or expired session."
            )
        return user_info

    # API keys (external/programmatic)
    key_info = await validate_key(
        token, session, supabase_url, supabase_service_key
    )
    if not key_info:
        logger.warning("API key auth failed for %s %s", request.method, request.url.path)
        raise HTTPException(
            status_code=401, detail="Invalid or inactive API key."
        )
    return key_info
