import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
from app.models.base_handler import AbstractModelHandler
from typing import Any
from app.core.exceptions import InferenceError, ModelLoadError


class SegmentationModelHandler(AbstractModelHandler):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.input_name = None
        self.output_name = None
        self.input_size = (512, 512) 

    def unload_model(self) -> None:
        """Release ONNX runtime session"""
        if self.model is not None:
            del self.model
            self.model = None
            self.input_name = None
            self.output_name = None
        print(f"Segmentation model unloaded")


    def load_model(self) -> None:
        try:
            self.model = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Automatically determine the names of the inputs/outputs
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            
            print(f"Segmentation model loaded")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {str(e)}")


    def preprocess(self, data: dict) -> Any:
        try:
            if "file_bytes" not in data:
                raise InferenceError("Missing required field 'file_bytes' in input data")
        
            file_bytes = data["file_bytes"]

            if not file_bytes or len(file_bytes) == 0:
                raise InferenceError("Empty image file provided")
        
            # Load image 
            image = Image.open(io.BytesIO(file_bytes))
            
            # Store original size for postprocessing
            original_size = image.size
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to 512x512
            if image.size != self.input_size:
                image = image.resize(self.input_size, Image.Resampling.BILINEAR)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Add channel and batch dimensions: [1, 1, 512, 512]
            img_array = np.expand_dims(img_array, axis=0)  # add channel dimension
            img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

            return {
                "input_tensor": img_array,
                "original_size": original_size
            }
            
        except InferenceError:
            raise  
        except Exception as e:
            raise InferenceError(f"Image preprocessing failed: {str(e)}")


    def predict(self, processed: Any) -> Any:
        """Run segmentation inference via ONNX Runtime"""
        self.ensure_loaded()

        try:
            input_tensor = processed["input_tensor"]
            
            # Run the model
            outputs = self.model.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )
            
            return {
                "segmentation_mask": outputs[0],
                "original_size": processed["original_size"]
            }
            
        except Exception as e:
            print(f"Segmentation prediction error: {e}")
            raise InferenceError(f"Segmentation failed: {str(e)}")


    def postprocess(self, raw_output: Any) -> dict:
        """Convert segmentation mask to usable format"""
        try:
            segmentation_mask = raw_output["segmentation_mask"][0, 0]  # Remove batch and channel dimensions
            original_size = raw_output["original_size"]
            
            # Apply threshold to get binary mask (как в оригинальном датасете: >0.5 -> 1)
            binary_mask = (segmentation_mask > 0.5).astype(np.float32)
            
            # Scale to [0, 255] for image representation
            binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
            
            # Resize mask back to original image size using PIL
            mask_pil = Image.fromarray(binary_mask_uint8)
            if mask_pil.size != original_size:
                mask_pil = mask_pil.resize(original_size, Image.Resampling.BILINEAR)
                binary_mask_uint8 = np.array(mask_pil)
            
            # Convert mask to base64 for JSON serialization
            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format='PNG')
            mask_base64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')
            
            # Calculate some basic statistics
            mask_area = np.sum(binary_mask > 0.5)  # Count pixels > 0.5
            total_pixels = binary_mask.size
            coverage_percentage = (mask_area / total_pixels) * 100
            
            result = {
                "mask_base64": mask_base64,
                "mask_shape": binary_mask_uint8.shape,
                "coverage_percentage": float(coverage_percentage),
                "mask_area": int(mask_area),
                "original_size": original_size
            }
            
            return result
            
        except Exception as e:
            print(f"Segmentation postprocessing error: {e}")
            raise InferenceError(f"Postprocessing failed: {str(e)}")