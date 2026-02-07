"""
GeniusPro Superintelligence v1 — Configuration

All provider keys and settings loaded from environment variables.
Never hardcode secrets.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    base_url: str
    api_key: str
    model_id: str
    format: str  # "openai" or "anthropic" or "google"
    enabled: bool = True
    max_tokens_default: int = 4096
    supports_streaming: bool = True


@dataclass
class SuperintelligenceConfig:
    """Top-level configuration for Superintelligence v1."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8100

    # Supabase (auth + logging)
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Routing
    embedding_model: str = "all-MiniLM-L6-v2"
    knn_index_path: str = "routing/knn_index.pkl"

    # Providers
    providers: list[ProviderConfig] = field(default_factory=list)


def load_config() -> SuperintelligenceConfig:
    """Load configuration from environment variables."""

    providers: list[ProviderConfig] = []

    # --- OpenAI ---
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        providers.append(
            ProviderConfig(
                name="openai",
                base_url="https://api.openai.com/v1",
                api_key=openai_key,
                model_id="gpt-4o",  # Default, router overrides per-request
                format="openai",
            )
        )

    # --- Anthropic ---
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        providers.append(
            ProviderConfig(
                name="anthropic",
                base_url="https://api.anthropic.com",
                api_key=anthropic_key,
                model_id="claude-sonnet-4-20250514",
                format="anthropic",
            )
        )

    # --- OpenRouter (Grok, Kimi, etc.) ---
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_key:
        providers.append(
            ProviderConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_id="moonshotai/kimi-k2",
                format="openai",
            )
        )
        # DeepSeek V3.2 via OpenRouter
        providers.append(
            ProviderConfig(
                name="openrouter-deepseek",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_id="deepseek/deepseek-v3.2",
                format="openai",
            )
        )
        # Mistral Large 3 via OpenRouter
        providers.append(
            ProviderConfig(
                name="openrouter-mistral",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_id="mistralai/mistral-large-3-2512",
                format="openai",
            )
        )
        # Gemini 2.5 Pro via OpenRouter
        providers.append(
            ProviderConfig(
                name="openrouter-google",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_id="google/gemini-2.5-pro-preview-06-05",
                format="openai",
            )
        )

    # --- DeepSeek (direct API) ---
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        providers.append(
            ProviderConfig(
                name="deepseek",
                base_url="https://api.deepseek.com/v1",
                api_key=deepseek_key,
                model_id="deepseek-chat",
                format="openai",
            )
        )

    # --- Google Gemini (direct API — reserved for future dedicated provider) ---
    # google_key = os.environ.get("GOOGLE_API_KEY", "")
    # For now, Gemini is routed via OpenRouter (openrouter-google)

    # --- Mistral ---
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if mistral_key:
        providers.append(
            ProviderConfig(
                name="mistral",
                base_url="https://api.mistral.ai/v1",
                api_key=mistral_key,
                model_id="mistral-large-latest",
                format="openai",
            )
        )

    return SuperintelligenceConfig(
        supabase_url=os.environ.get(
            "SUPABASE_URL", "https://orajwuisgwffnrbjasaj.supabase.co"
        ),
        supabase_service_key=os.environ.get("SUPABASE_SERVICE_KEY", ""),
        providers=providers,
    )
