from fastapi import FastAPI
from app.api.routers import predict
from app.core.config import settings
from app.models.registry import registry
from app.models.classification_handler import ClassificationModelHandler
from app.models.segmentation_handler import SegmentationModelHandler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title=settings.APP_NAME)

app.include_router(predict.router)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

def register_models():
    """Registration of models at launch"""

    # Classification model
    classification_path = os.path.join(settings.MODEL_DIR, "classification", "simplest_model/bacterial_cnn.onnx")
    classification_handler = ClassificationModelHandler(
        classification_path)
    registry.register("classification", classification_handler)

    # Segmentation model
    segmentation_path = os.path.join(settings.MODEL_DIR, "segmentation", "simplest_model/segmetation_unet.onnx")
    segmentation_handler = SegmentationModelHandler(segmentation_path)
    registry.register("segmentation", segmentation_handler)

    print("Models registered:", list(registry._handlers.keys())) 

@app.on_event("startup")
async def startup_event():
    register_models()

@app.on_event("shutdown")
async def shutdown_event():
    """Unload all models on shutdown"""
    registry.unload_all()

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    models_status = {}
    for name, handler in registry._handlers.items():
        models_status[name] = {
            "name": name,
            "loaded": handler._loaded,
            "model_path": handler.model_path
        }
    return models_status

@app.post("/unload/{model_name}")
def unload_model(model_name: str):
    """Endpoint to manually unload a specific model"""
    handler = registry._handlers.get(model_name)
    if not handler:
        return {"status": "error", "message": f"Model {model_name} not found"}
    
    if handler._loaded:
        handler.unload()
        return {"status": "success", "message": f"Model {model_name} unloaded"}
    else:
        return {"status": "success", "message": f"Model {model_name} was not loaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
