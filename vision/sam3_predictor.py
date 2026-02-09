"""
GeniusPro Vision Service — SAM 3 Predictor Wrapper

Wraps SAM 3 for image and video segmentation with text and visual prompts.
Supports both local SAM 3 and Roboflow API providers.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import uuid
import tempfile

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("vision.sam3_predictor")


class SAM3PredictorManager:
    """Manages SAM 3 model instances for image and video prediction."""
    
    def __init__(self, device: str = "cuda", provider: str = "local", roboflow_client: Optional[Any] = None):
        """
        Initialize SAM 3 predictor manager.
        
        Args:
            device: Device to use ("cuda" or "cpu") - only for local provider
            provider: "local" (self-hosted) or "roboflow" (API)
            roboflow_client: Roboflow client instance (if provider="roboflow")
        """
        self.device = device
        self.provider = provider
        self.roboflow_client = roboflow_client
        
        if provider == "roboflow":
            logger.info("Using Roboflow SAM 3 API provider")
            if not roboflow_client:
                raise ValueError("roboflow_client required when provider='roboflow'")
        else:
            logger.info("Loading local SAM 3 model (device: %s)", device)
            self.image_model: Optional[Any] = None
            self.image_processor: Optional[Any] = None
            self.video_predictor: Optional[Any] = None
            self._load_models()
    
    def _load_models(self):
        """Load local SAM 3 models for image and video prediction."""
        if self.provider != "local":
            return
        
        try:
            from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Build SAM 3 image model (automatically downloads from Hugging Face)
            self.image_model = build_sam3_image_model()
            self.image_processor = Sam3Processor(self.image_model)
            logger.info("SAM 3 image model initialized")
            
            # Build SAM 3 video predictor
            self.video_predictor = build_sam3_video_predictor()
            logger.info("SAM 3 video predictor initialized")
            
        except Exception as e:
            logger.error("Failed to load SAM 3 models", exc_info=True)
            raise
    
    def segment_image(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_prompt: Optional[str] = None,
        input_points: Optional[List[List[float]]] = None,
        input_labels: Optional[List[int]] = None,
        input_box: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Segment an image using SAM 3.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            text_prompt: Text description (e.g., "a red car", "player in white")
            input_points: List of [x, y] point coordinates
            input_labels: List of labels (1 for foreground, 0 for background)
            input_box: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Dictionary with masks, boxes, and scores
        """
        # Use Roboflow API if configured
        if self.provider == "roboflow":
            return self._segment_image_roboflow(
                image=image,
                text_prompt=text_prompt,
                input_points=input_points,
                input_box=input_box,
            )
        
        # Local SAM 3
        if self.image_processor is None:
            raise RuntimeError("Image processor not initialized")
        
        # Load image if needed
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Set image in processor
        inference_state = self.image_processor.set_image(image)
        
        # Process prompts
        if text_prompt:
            # Text prompt (SAM 3's main feature)
            output = self.image_processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )
        elif input_box:
            # Box prompt
            box_array = np.array(input_box, dtype=np.float32).reshape(1, 4)
            output = self.image_processor.set_box_prompt(
                state=inference_state,
                box=box_array
            )
        elif input_points:
            # Point prompt
            points = np.array(input_points, dtype=np.float32)
            labels = np.array(input_labels if input_labels else [1] * len(input_points), dtype=np.int32)
            output = self.image_processor.set_point_prompt(
                state=inference_state,
                point_coords=points,
                point_labels=labels
            )
        else:
            raise ValueError("At least one of text_prompt, input_points, or input_box must be provided")
        
        # Extract results
        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])
        
        return {
            "masks": [mask.tolist() if isinstance(mask, np.ndarray) else mask for mask in masks],
            "boxes": [box.tolist() if isinstance(box, np.ndarray) else box for box in boxes],
            "scores": [float(score) if isinstance(score, (np.ndarray, np.generic)) else score for score in scores],
        }
    
    def _segment_image_roboflow(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_prompt: Optional[str] = None,
        input_points: Optional[List[List[float]]] = None,
        input_box: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Segment image using Roboflow API (synchronous wrapper)."""
        import asyncio
        
        # Save image to temp file if needed
        temp_file = None
        try:
            if isinstance(image, (np.ndarray, Image.Image)):
                # Save PIL/numpy image to temp file
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                image.save(temp_file.name)
                image_path = temp_file.name
            else:
                image_path = image
            
            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to handle this differently
                # For now, create a new event loop
                import nest_asyncio
                nest_asyncio.apply()
            
            result = asyncio.run(
                self.roboflow_client.segment_image_async(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    input_points=input_points,
                    input_box=input_box,
                )
            )
            
            return result
        finally:
            # Clean up temp file
            if temp_file:
                try:
                    Path(temp_file.name).unlink()
                except Exception:
                    pass
    
    def segment_video_init(
        self,
        video_path: str,
    ) -> Dict[str, Any]:
        """
        Initialize video segmentation session.
        
        Args:
            video_path: Path to video file (MP4) or directory of JPEG frames
        
        Returns:
            Dictionary with session_id
        """
        if self.video_predictor is None:
            raise RuntimeError("Video predictor not initialized")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Start session
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        
        session_id = response.get("session_id")
        if not session_id:
            raise RuntimeError("Failed to create video session")
        
        # Store session (in production, use a proper state manager)
        if not hasattr(self, "_video_sessions"):
            self._video_sessions = {}
        self._video_sessions[session_id] = {
            "video_path": video_path,
            "predictor": self.video_predictor,
        }
        
        return {
            "session_id": session_id,
            "video_path": video_path,
        }
    
    def segment_video_add_prompt(
        self,
        session_id: str,
        text_prompt: Optional[str] = None,
        frame_index: int = 0,
        input_points: Optional[List[List[float]]] = None,
        input_labels: Optional[List[int]] = None,
        input_box: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Add a prompt to video segmentation session.
        
        Args:
            session_id: Session ID from init
            text_prompt: Text description (e.g., "a red car")
            frame_index: Frame index to add prompt (default: 0)
            input_points: List of [x, y] point coordinates
            input_labels: List of labels (1 for foreground, 0 for background)
            input_box: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Dictionary with outputs (masks, boxes, scores)
        """
        if self.video_predictor is None:
            raise RuntimeError("Video predictor not initialized")
        
        if not hasattr(self, "_video_sessions") or session_id not in self._video_sessions:
            raise ValueError(f"Invalid session_id: {session_id}")
        
        predictor = self._video_sessions[session_id]["predictor"]
        
        # Build request based on prompt type
        if text_prompt:
            request = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text_prompt,
            )
        elif input_box:
            request = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                box=input_box,
            )
        elif input_points:
            request = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                point_coords=input_points,
                point_labels=input_labels if input_labels else [1] * len(input_points),
            )
        else:
            raise ValueError("At least one of text_prompt, input_points, or input_box must be provided")
        
        # Add prompt
        response = predictor.handle_request(request=request)
        outputs = response.get("outputs", {})
        
        return {
            "session_id": session_id,
            "frame_index": frame_index,
            "masks": outputs.get("masks", []),
            "boxes": outputs.get("boxes", []),
            "scores": outputs.get("scores", []),
        }
    
    def segment_video_propagate(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Propagate masks through video frames.
        
        Args:
            session_id: Session ID from init
        
        Returns:
            List of dictionaries with frame_idx, masks, boxes, scores for each frame
        """
        if self.video_predictor is None:
            raise RuntimeError("Video predictor not initialized")
        
        if not hasattr(self, "_video_sessions") or session_id not in self._video_sessions:
            raise ValueError(f"Invalid session_id: {session_id}")
        
        predictor = self._video_sessions[session_id]["predictor"]
        results = []
        
        # Request propagation
        response = predictor.handle_request(
            request=dict(
                type="propagate",
                session_id=session_id,
            )
        )
        
        outputs = response.get("outputs", {})
        frames = outputs.get("frames", [])
        
        for frame_data in frames:
            results.append({
                "frame_index": frame_data.get("frame_index", 0),
                "masks": frame_data.get("masks", []),
                "boxes": frame_data.get("boxes", []),
                "scores": frame_data.get("scores", []),
            })
        
        return results
    
    def cleanup_session(self, session_id: str):
        """Clean up a video session."""
        if hasattr(self, "_video_sessions") and session_id in self._video_sessions:
            del self._video_sessions[session_id]
