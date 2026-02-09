"""GeniusPro Vision Service — SAM 3 Image and Video Segmentation API"""

import logging
import uuid
from pathlib import Path
from typing import Optional, List

import aiohttp
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from vision.config import load_config, VisionConfig
from vision.sam3_predictor import SAM3PredictorManager
from vision.roboflow_provider import RoboflowSAM3Provider

# Import auth from superintelligence (shared pattern)
# Add superintelligence directory to path
import sys
import os
superintelligence_path = os.path.expanduser("~/geniuspro-superintelligence")
if superintelligence_path not in sys.path:
    sys.path.insert(0, superintelligence_path)
from superintelligence.auth import require_auth

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vision")

# ─── Constants ────────────────────────────────────────────────────────────────

API_PREFIX = "/vision/v1"
MODEL_NAME = "GeniusPro-vision-sam3"

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GeniusPro Vision Service",
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

_config: Optional[VisionConfig] = None
_session: Optional[aiohttp.ClientSession] = None
_predictor: Optional[SAM3PredictorManager] = None


@app.on_event("startup")
async def startup() -> None:
    global _config, _session, _predictor
    
    logger.info("Starting GeniusPro Vision Service...")
    _config = load_config()
    _session = aiohttp.ClientSession()
    
    # Initialize predictor based on provider
    if _config.provider == "roboflow":
        if not _config.roboflow_api_key:
            logger.warning("Roboflow provider selected but ROBOFLOW_API_KEY not set")
            _predictor = None
        else:
            try:
                roboflow_client = RoboflowSAM3Provider(
                    api_key=_config.roboflow_api_key,
                    workspace_name=_config.roboflow_workspace,
                    workflow_id=_config.roboflow_workflow_id,
                    session=_session,  # Share session
                )
                _predictor = SAM3PredictorManager(
                    device=_config.device,
                    provider="roboflow",
                    roboflow_client=roboflow_client,
                )
                logger.info("GeniusPro Vision Service ready — Using Roboflow SAM 3 API")
            except Exception as e:
                logger.error("Failed to initialize Roboflow provider", exc_info=True)
                _predictor = None
    else:
        # Local SAM 3 - lazy load on first request
        _predictor = None
        logger.info("GeniusPro Vision Service ready — Local SAM 3 will load on first request")


@app.on_event("shutdown")
async def shutdown() -> None:
    global _session
    if _session:
        await _session.close()
    logger.info("Vision Service shut down.")


# ─── Auth helper ──────────────────────────────────────────────────────────────

async def _auth(request: Request) -> dict:
    """Authenticate the request."""
    if not _config or not _session:
        raise HTTPException(status_code=503, detail="Service not ready")
    return await require_auth(
        request, _session, _config.supabase_url, _config.supabase_service_key
    )


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get(f"{API_PREFIX}/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": MODEL_NAME,
        "model_loaded": _predictor is not None,
        "device": _config.device if _config else "unknown",
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
                "capabilities": ["image_segmentation", "video_segmentation"],
            }
        ],
    }


# ─── Image Segmentation ────────────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/segment-image")
async def segment_image(
    request: Request,
    image: UploadFile = File(...),
    text_prompt: Optional[str] = Form(None),  # Text description (e.g., "a red car")
    input_points: Optional[str] = Form(None),  # JSON string: [[x1, y1], [x2, y2]]
    input_labels: Optional[str] = Form(None),  # JSON string: [1, 0]
    input_box: Optional[str] = Form(None),  # JSON string: [x1, y1, x2, y2]
):
    """
    Segment an image using SAM 3.
    
    Accepts image file and prompts:
    - text_prompt: Text description (e.g., "a red car", "player in white") - SAM 3's main feature
    - input_points: Point coordinates
    - input_box: Bounding box
    """
    req_id = uuid.uuid4().hex[:8]
    await _auth(request)
    await _ensure_predictor()
    
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(image_data)
        
        # Parse prompts
        points = None
        labels = None
        box = None
        
        if input_points:
            import json
            points = json.loads(input_points)
            if input_labels:
                labels = json.loads(input_labels)
        
        if input_box:
            import json
            box = json.loads(input_box)
        
        if not text_prompt and not points and not box:
            raise HTTPException(
                status_code=400,
                detail="At least one of text_prompt, input_points, or input_box is required",
            )
        
        # Segment (handle async for Roboflow)
        if _config.provider == "roboflow":
            import asyncio
            import tempfile
            
            # Save image to temp file for Roboflow
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                pil_image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                result = await _predictor.roboflow_client.segment_image_async(
                    image_path=tmp_path,
                    text_prompt=text_prompt,
                    input_points=points,
                    input_box=box,
                )
            finally:
                Path(tmp_path).unlink()
        else:
            result = _predictor.segment_image(
                image=pil_image,
                text_prompt=text_prompt,
                input_points=points,
                input_labels=labels,
                input_box=box,
            )
        
        logger.info("[%s] Image segmentation completed", req_id)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error("[%s] Image segmentation failed", req_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Video Segmentation ────────────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/segment-video/init")
async def segment_video_init(
    request: Request,
    video: UploadFile = File(...),
):
    """
    Initialize video segmentation state.
    
    Upload a video file to start segmentation.
    """
    req_id = uuid.uuid4().hex[:8]
    await _auth(request)
    await _ensure_predictor()
    
    try:
        # Save uploaded video to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            video_data = await video.read()
            tmp_file.write(video_data)
            tmp_path = tmp_file.name
        
        # Initialize state
        result = _predictor.segment_video_init(tmp_path)
        
        logger.info("[%s] Video segmentation initialized: session_id=%s", req_id, result["session_id"])
        return JSONResponse(result)
        
    except Exception as e:
        logger.error("[%s] Video initialization failed", req_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{API_PREFIX}/segment-video/add-prompt")
async def segment_video_add_prompt(
    request: Request,
    session_id: str = Form(...),
    text_prompt: Optional[str] = Form(None),  # Text description (e.g., "a red car")
    frame_index: int = Form(0),
    input_points: Optional[str] = Form(None),
    input_labels: Optional[str] = Form(None),
    input_box: Optional[str] = Form(None),
):
    """
    Add a prompt to video segmentation session.
    
    Provide session_id from init and prompts (text, points, or box).
    """
    req_id = uuid.uuid4().hex[:8]
    await _auth(request)
    await _ensure_predictor()
    
    try:
        # Parse prompts
        points = None
        labels = None
        box = None
        
        if input_points:
            import json
            points = json.loads(input_points)
            if input_labels:
                labels = json.loads(input_labels)
        
        if input_box:
            import json
            box = json.loads(input_box)
        
        if not text_prompt and not points and not box:
            raise HTTPException(
                status_code=400,
                detail="At least one of text_prompt, input_points, or input_box is required",
            )
        
        # Add prompt
        result = _predictor.segment_video_add_prompt(
            session_id=session_id,
            text_prompt=text_prompt,
            frame_index=frame_index,
            input_points=points,
            input_labels=labels,
            input_box=box,
        )
        
        logger.info("[%s] Prompt added to video session: session_id=%s", req_id, session_id)
        return JSONResponse(result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[%s] Failed to add prompt", req_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{API_PREFIX}/segment-video/propagate")
async def segment_video_propagate(
    request: Request,
    session_id: str = Form(...),
):
    """
    Propagate masks through video frames.
    
    Provide session_id from init. Returns masks for all frames.
    """
    req_id = uuid.uuid4().hex[:8]
    await _auth(request)
    await _ensure_predictor()
    
    try:
        # Propagate
        results = _predictor.segment_video_propagate(session_id=session_id)
        
        logger.info("[%s] Video propagation completed: session_id=%s frames=%d", 
                    req_id, session_id, len(results))
        return JSONResponse({"frames": results})
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("[%s] Video propagation failed", req_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{API_PREFIX}/segment-video/cleanup")
async def segment_video_cleanup(
    request: Request,
    session_id: str = Form(...),
):
    """
    Clean up video segmentation session.
    
    Call this after completing video segmentation to free memory.
    """
    req_id = uuid.uuid4().hex[:8]
    await _auth(request)
    await _ensure_predictor()
    
    try:
        _predictor.cleanup_session(session_id)
        logger.info("[%s] Video session cleaned up: session_id=%s", req_id, session_id)
        return JSONResponse({"ok": True})
        
    except Exception as e:
        logger.error("[%s] Failed to cleanup state", req_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
