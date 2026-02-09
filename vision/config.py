"""
GeniusPro Vision Service — Configuration

SAM 3 model configuration loaded from environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionConfig:
    """Configuration for Vision Service."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8200

    # Supabase (auth)
    supabase_url: str = ""
    supabase_service_key: str = ""

    # SAM 3 Model
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Provider selection
    provider: str = "local"  # "local" (self-hosted SAM 3) or "roboflow" (Roboflow API)
    
    # Hugging Face (for local SAM 3 checkpoint access)
    hf_token: Optional[str] = None  # Hugging Face token for checkpoint access
    
    # Roboflow (for Roboflow API provider)
    roboflow_api_key: Optional[str] = None
    roboflow_workspace: str = "geniuspro"
    roboflow_workflow_id: str = "sam3"


def load_config() -> VisionConfig:
    """Load configuration from environment variables."""
    
    # Device selection
    device = os.environ.get("SAM3_DEVICE", "cuda")
    if device not in ["cuda", "cpu"]:
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Provider selection
    provider = os.environ.get("SAM3_PROVIDER", "local").lower()
    if provider not in ["local", "roboflow"]:
        provider = "local"
    
    return VisionConfig(
        supabase_url=os.environ.get(
            "SUPABASE_URL", "https://orajwuisgwffnrbjasaj.supabase.co"
        ),
        supabase_service_key=os.environ.get("SUPABASE_SERVICE_KEY", ""),
        device=device,
        provider=provider,
        hf_token=os.environ.get("HUGGING_FACE_TOKEN"),
        roboflow_api_key=os.environ.get("ROBOFLOW_API_KEY"),
        roboflow_workspace=os.environ.get("ROBOFLOW_WORKSPACE", "geniuspro"),
        roboflow_workflow_id=os.environ.get("ROBOFLOW_WORKFLOW_ID", "sam3"),
    )
