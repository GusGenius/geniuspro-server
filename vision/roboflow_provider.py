"""
GeniusPro Vision Service — Roboflow SAM 3 Provider

Uses Roboflow's Inference API for SAM 3 segmentation.
Uses direct HTTP requests (works with Python 3.13+).
"""

import logging
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
import aiohttp

logger = logging.getLogger("vision.roboflow_provider")


class RoboflowSAM3Provider:
    """Roboflow SAM 3 API provider using direct HTTP requests."""
    
    def __init__(
        self,
        api_key: str,
        workspace_name: str = "geniuspro",
        workflow_id: str = "sam3",
        api_url: str = "https://serverless.roboflow.com",
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize Roboflow SAM 3 provider.
        
        Args:
            api_key: Roboflow API key
            workspace_name: Roboflow workspace name
            workflow_id: Workflow ID (default: "sam3")
            api_url: Roboflow API URL
            session: Optional aiohttp session (will create if not provided)
        """
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.api_url = api_url
        self.session = session
        self._own_session = session is None
        
        logger.info("Roboflow SAM 3 provider initialized (workspace=%s, workflow=%s)", 
                    workspace_name, workflow_id)
    
    async def segment_image_async(
        self,
        image_path: str,
        text_prompt: Optional[str] = None,
        input_points: Optional[List[List[float]]] = None,
        input_box: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Segment an image using Roboflow SAM 3 workflow (async).
        
        Args:
            image_path: Path to image file
            text_prompt: Text description (e.g., "a red car")
            input_points: List of [x, y] point coordinates
            input_box: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Dictionary with masks, boxes, and scores
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Prepare request payload
        payload = {
            "image": image_b64,
        }
        
        # Add prompts if provided
        if text_prompt:
            payload["text_prompt"] = text_prompt
        if input_points:
            payload["input_points"] = input_points
        if input_box:
            payload["input_box"] = input_box
        
        # Create session if needed
        if self._own_session and not self.session:
            self.session = aiohttp.ClientSession()
        
        # Call Roboflow API
        # Endpoint format: POST /{workspace_name}/workflows/{workflow_id}
        url = f"{self.api_url}/{self.workspace_name}/workflows/{self.workflow_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Roboflow expects inputs in a nested structure
        request_payload = {
            "api_key": self.api_key,
            "inputs": payload,
            "use_cache": True,
        }
        
        try:
            async with self.session.post(url, json=request_payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Roboflow API error {resp.status}: {error_text}")
                
                result = await resp.json()
                
                # Roboflow returns outputs in a nested structure
                outputs = result.get("outputs", {})
                
                # Extract results from Roboflow response
                # Adjust based on actual Roboflow response structure
                return {
                    "masks": outputs.get("masks", []),
                    "boxes": outputs.get("boxes", []),
                    "scores": outputs.get("scores", []),
                    "provider": "roboflow",
                    "raw_response": result,  # Include full response for debugging
                }
        except Exception as e:
            logger.error("Roboflow API error", exc_info=True)
            raise RuntimeError(f"Roboflow API error: {str(e)}")
    
    def cleanup(self):
        """Clean up resources."""
        if self._own_session and self.session:
            # Note: Don't close here, let app manage session lifecycle
            pass
