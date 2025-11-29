from app.models.base_handler import AbstractModelHandler
from typing import Any


class FusionModelHandler(AbstractModelHandler):
    def __init__(self, model_path: str, img_encoder_path: str | None = None, tab_encoder_path: str | None = None):
        super().__init__(model_path)
        self.img_encoder_path = img_encoder_path
        self.tab_encoder_path = tab_encoder_path
        self.image_encoder = None
        self.tab_encoder = None


    def load_model(self) -> None:
        # TODO: load fusion model and optionally encoders
        self.model = "<loaded_fusion_model>"
        self.image_encoder = "<img_encoder>"
        self.tab_encoder = "<tab_encoder>"


    def preprocess(self, data: dict) -> Any:
        # expect {"file_bytes": ..., "payload": {tabular dict}}
        # process image & tabular separately
        return {"img_input": "img_tensor", "tab_input": "tab_tensor"}


    def predict(self, processed: Any) -> Any:
        # TODO: run encoders, fuse embeddings, run fusion model
        return {"raw": "fusion_raw_output"}


    def postprocess(self, raw_output: Any) -> dict:
        # TODO: decode to human-readable
        return {"fusion_result": raw_output}