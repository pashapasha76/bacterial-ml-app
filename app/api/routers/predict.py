from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.registry import registry
import json

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("/{model_name}")
async def predict(
    model_name: str,
    file: UploadFile = File(...),
    payload: str = Form(None)
):
    handler = registry.get(model_name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        file_bytes = await file.read()
        tabular_data = json.loads(payload) if payload else None
        
        input_data = {"file_bytes": file_bytes, "payload": tabular_data}
        
        # model will load here on the first call.
        processed = handler.preprocess(input_data)
        raw_output = handler.predict(processed)
        result = handler.postprocess(raw_output)
        
        return {
            "status": "success",
            "model": model_name,
            "model_loaded": handler._loaded,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")