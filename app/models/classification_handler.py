import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from app.models.base_handler import AbstractModelHandler
from typing import Any
from app.core.exceptions import InferenceError, ModelLoadError


class ClassificationModelHandler(AbstractModelHandler):
    def __init__(self, model_path: str, labels: list[str] | None = None):
        super().__init__(model_path)
        self.labels = ['CAM', 'Control', 'MP265', 'Mecillinam', 'Nalidixate', 'Oblique', 'Rifampicin', 'Vesicle']
        self.input_name = None
        self.output_name = None

    def unload_model(self) -> None:
        """Release ONNX runtime session"""
        if self.model is not None:
            # ONNX Runtime doesn't have explicit close method, 
            # but we can delete the session and let GC handle it
            del self.model
            self.model = None
            self.input_name = None
            self.output_name = None
            print(f"Classification model unloaded")

    def load_model(self) -> None:
        self.model = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Automatically determine the names of the inputs/outputs
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

        print(f"Classification model loaded")


    def preprocess(self, data: dict) -> Any:
        try:
            if "file_bytes" not in data:
                raise InferenceError("Missing required field 'file_bytes' in input data")
        
            file_bytes = data["file_bytes"]

            if not file_bytes or len(file_bytes) == 0:
                raise InferenceError("Empty image file provided")
        
            image = Image.open(io.BytesIO(file_bytes))
            
            # Resize up to 64x64
            if image.size != (64, 64):
                image = image.resize((64, 64))
            
            # Converting to a numpy array
            image_array = np.array(image).astype(np.float32)
            
            # If the image is RGB
            if len(image_array.shape) != 3:
                raise InferenceError(f"Expected 3-dimensional image (H, W, C), got {len(image_array.shape)} dimensions")
            
            if image_array.shape[2] != 3:
                raise InferenceError(f"Expected 3 color channels (RGB), got {image_array.shape[2]} channels")
            
            image_array = np.transpose(image_array, (2, 0, 1))
        
            # Normalization
            image_array = (image_array / 255.0 - 0.5) / 0.5
        
            # Add batch dimension ([1, 3, 64, 64])
            image_array = np.expand_dims(image_array, axis=0)

            if len(image_array.shape) != 4:
                raise InferenceError(f"Expected 4-dimensional tensor (batch, channels, height, width), got {len(image_array.shape)} dimensions")
        
            if image_array.shape != (1, 3, 64, 64):
                raise InferenceError(f"Expected tensor shape (1, 3, 64, 64), got {image_array.shape}")
        
            
            return {"input_tensor": image_array}
            
        except InferenceError:
            raise  
        except Exception as e:
            raise InferenceError(f"Image preprocessing failed: {str(e)}")


    def predict(self, processed: Any) -> Any:
        """Launching inference via ONNX Runtime"""
        self.ensure_loaded()

        try:
            input_tensor = processed["input_tensor"]
            
            # Launching the model
            outputs = self.model.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )
            
            return {"logits": outputs[0]}
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise


    def postprocess(self, raw_output: Any) -> dict:
        """Converting logits into probabilities of 8 classes"""
        try:
            logits = raw_output["logits"][0]
            
            # Softmax for probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Finding the best class
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            result = {
                "predicted_class": self.labels[predicted_class_idx],
                "confidence": float(confidence),
                "all_probabilities": {
                    self.labels[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Postprocessing error: {e}")
            raise