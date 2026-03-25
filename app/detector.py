"""
Detection module - Solar panel detection using specialized YOLOv8 segmentation model

Uses a custom-trained YOLOv8 segmentation model for accurate detection of solar panels
in satellite imagery.
"""
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import os


def find_best_model():
    """
    Find the best available model in order of preference:
    1. Latest trained YOLO26 segmentation model (mAP41)
    2. Custom trained model: models/solar_panel_best.pt
    3. Downloaded model: models/best.pt
    4. Pre-trained YOLOv8 (fallback)
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Priority 1: Latest YOLO26 segmentation model (mAP50 = 41%)
    yolo26_model = os.path.join(base_dir, "models", "solar_panel_mAP41_seg.pt")
    if os.path.exists(yolo26_model):
        print(f"🚀 Using YOLO26 segmentation model (mAP50=41%): {yolo26_model}")
        return yolo26_model
    
    # Priority 2: Custom trained model
    custom_model = os.path.join(base_dir, "models", "solar_panel_best.pt")
    if os.path.exists(custom_model):
        print(f"✅ Using custom trained model: {custom_model}")
        return custom_model
    
    # Priority 3: Downloaded finloop model
    downloaded_model = os.path.join(base_dir, "models", "best.pt")
    if os.path.exists(downloaded_model):
        print(f"📦 Using downloaded model: {downloaded_model}")
        return downloaded_model
    
    # Priority 4: Run path from training
    run_model = os.path.join(base_dir, "runs", "segment", "solar_panel_seg", "weights", "best.pt")
    if os.path.exists(run_model):
        print(f"🏋️ Using training run model: {run_model}")
        return run_model
    
    return None


class SolarPanelDetector:
    """
    Detects solar panels in satellite imagery using a specialized YOLOv8 
    segmentation model trained on solar panel satellite imagery.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25
    ):
        """
        Initialize the solar panel detector.
        
        Args:
            model_path: Path to the YOLOv8 segmentation model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path or find_best_model()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 segmentation model."""
        try:
            from ultralytics import YOLO
            
            if self.model_path is None:
                raise FileNotFoundError(
                    "Kein Modell gefunden! Bitte trainiere erst ein Modell:\n"
                    "python train_model.py"
                )
            
            print(f"Loading solar panel model: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Please train a model first: python train_model.py"
                )
            
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully. Task: {self.model.task}")
            
        except ImportError:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect solar panels in an image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            confidence_threshold: Override default confidence threshold
            
        Returns:
            Dictionary containing:
                - detections: List of detection dicts with bbox, confidence, class
                - annotated_image: Image with bounding boxes drawn
                - total_panel_pixels: Estimated total pixels covered by panels
                - num_panels: Number of detected panels
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        conf = confidence_threshold or self.confidence_threshold
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            # Ensure RGB mode (YOLO requires 3 channels)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
        elif isinstance(image, (str, Path)):
            pil_img = Image.open(image)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            image_array = np.array(pil_img)
        else:
            image_array = image
            # If numpy array has wrong number of channels, try to fix
            if len(image_array.shape) == 2:
                # Grayscale to RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[-1] == 4:
                # RGBA to RGB
                image_array = image_array[:, :, :3]
        
        # Get the actual image size and use it for inference
        img_height, img_width = image_array.shape[:2]
        # Use the larger dimension, rounded up to nearest 32 (YOLO requirement)
        img_size = max(img_height, img_width)
        img_size = ((img_size + 31) // 32) * 32  # Round up to multiple of 32
        
        # Run detection with maximum resolution
        results = self.model.predict(
            source=image_array,
            conf=conf,
            imgsz=img_size,  # Use actual image size for maximum detail
            verbose=False
        )
        
        # Process results
        detections = []
        total_pixels = 0
        
        if len(results) > 0:
            result = results[0]
            
            # Process segmentation masks if available (preferred for accuracy)
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                # Get original image dimensions for proper scaling
                orig_h, orig_w = image_array.shape[:2]
                mask_h, mask_w = masks.shape[1:3] if len(masks.shape) == 3 else (orig_h, orig_w)
                
                # Scale factor to convert mask pixels to original image pixels
                scale_x = orig_w / mask_w
                scale_y = orig_h / mask_h
                pixel_scale = scale_x * scale_y
                
                for i, mask in enumerate(masks):
                    # Count pixels in this mask and scale to original resolution
                    mask_pixels_raw = int(np.sum(mask > 0.5))
                    mask_pixels = int(mask_pixels_raw * pixel_scale)
                    total_pixels += mask_pixels
                    
                    # Get corresponding box if available
                    if result.boxes is not None and i < len(result.boxes):
                        box = result.boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "mask_pixels": mask_pixels,
                            "width_pixels": float(x2 - x1),
                            "height_pixels": float(y2 - y1),
                        })
            
            # Fallback to bounding boxes if no masks
            elif result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Calculate bounding box area
                    width = x2 - x1
                    height = y2 - y1
                    area_pixels = width * height
                    total_pixels += area_pixels
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "mask_pixels": int(area_pixels),
                        "width_pixels": float(width),
                        "height_pixels": float(height),
                    })
            
            # Get annotated image with masks shown
            # Use result.plot() with masks=True to ensure masks are drawn
            annotated = result.plot(
                masks=True,      # Draw segmentation masks
                boxes=True,      # Also draw bounding boxes
                labels=True,     # Show labels with confidence
                conf=True,       # Show confidence values
            )
            annotated_image = Image.fromarray(annotated)
            
            # Log mask presence for debugging
            if result.masks is not None:
                print(f"✅ Segmentation masks found: {len(result.masks)} masks")
            else:
                print("⚠️ No segmentation masks - using bounding boxes only")
        else:
            annotated_image = Image.fromarray(image_array) if isinstance(image_array, np.ndarray) else image
        
        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "total_panel_pixels": total_pixels,
            "num_panels": len(detections)
        }
    
    def detect_with_masks(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect solar panels with segmentation masks for more accurate area calculation.
        Note: Requires a segmentation model variant.
        
        For YOLO World v2, we use bounding boxes as approximation.
        For more accurate masks, consider fine-tuning YOLOv8-seg on solar panel data.
        """
        # For now, fall back to bounding box detection
        # Segmentation would require a different model architecture
        return self.detect(image, confidence_threshold)


def detect_solar_panels(
    image: Union[Image.Image, np.ndarray, str, Path],
    confidence_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Convenience function to detect solar panels in an image.
    
    Args:
        image: PIL Image, numpy array, or path to image
        confidence_threshold: Minimum detection confidence
        
    Returns:
        Detection results dictionary
    """
    detector = SolarPanelDetector(confidence_threshold=confidence_threshold)
    return detector.detect(image)


if __name__ == "__main__":
    # Test the detector with a sample image
    print("Solar Panel Detector Test")
    print("=" * 40)
    
    # Create a simple test (will need actual satellite image)
    detector = SolarPanelDetector()
    
    # Test with a placeholder image
    test_image = Image.new("RGB", (640, 640), color=(100, 100, 100))
    results = detector.detect(test_image)
    
    print(f"Detections: {results['num_panels']}")
    print(f"Total panel pixels: {results['total_panel_pixels']}")
    
    for det in results["detections"]:
        print(f"  - {det['class_name']}: {det['confidence']:.2%} confidence")
