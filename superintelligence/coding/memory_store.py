"""
GeniusPro Superintelligence — Coding memory store (Supabase REST)

Stores ONLY:
- plan metadata / summaries
- user-approved snippets

Does NOT store full repos or diffs by default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import aiohttp


@dataclass(frozen=True)
class MemorySnippet:
    user_id: str
    project_slug: str
    content: str
    language: Optional[str] = None
    tags: Optional[list[str]] = None
    workspace_fingerprint: Optional[str] = None

    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def _supa_headers(service_key: str) -> dict[str, str]:
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def save_snippet(
    *,
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
    snippet: MemorySnippet,
) -> dict:
    url = f"{supabase_url}/rest/v1/coding_memory_snippets"
    payload = {
        "user_id": snippet.user_id,
        "project_slug": snippet.project_slug,
        "workspace_fingerprint": snippet.workspace_fingerprint,
        "language": snippet.language,
        "content": snippet.content,
        "content_hash": snippet.content_hash(),
        "tags": snippet.tags or [],
        "source": "user_approved",
    }

    async with session.post(url, headers=_supa_headers(supabase_service_key), data=json.dumps(payload)) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"Supabase save_snippet failed ({resp.status}): {text[:200]}")
        try:
            data = json.loads(text) if text else []
        except json.JSONDecodeError:
            data = []
        return data[0] if isinstance(data, list) and data else {"ok": True}


async def list_snippets(
    *,
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_service_key: str,
    user_id: str,
    project_slug: str,
    limit: int = 20,
) -> list[dict]:
    limit = max(1, min(limit, 100))
    url = (
        f"{supabase_url}/rest/v1/coding_memory_snippets"
        f"?user_id=eq.{user_id}"
        f"&project_slug=eq.{project_slug}"
        f"&select=id,project_slug,language,content,tags,created_at"
        f"&order=created_at.desc"
        f"&limit={limit}"
    )
    async with session.get(url, headers=_supa_headers(supabase_service_key)) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"Supabase list_snippets failed ({resp.status}): {text[:200]}")
        return await resp.json()

