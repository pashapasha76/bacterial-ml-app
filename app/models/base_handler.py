from abc import ABC, abstractmethod
from typing import Any
from threading import Lock



class AbstractModelHandler(ABC):
    """Base class for all model handlers.

    Subclasses must implement load_model(), preprocess(), predict(), postprocess().
    """


    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._loaded = False
        self._lock = Lock()

    def ensure_loaded(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self.load_model()
                    self._loaded = True

    def unload(self):
        """Unload model from memory"""
        with self._lock:
            if self._loaded:
                self.unload_model()
                self._loaded = False

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model weights / runtime session."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights / runtime session."""
        raise NotImplementedError


    @abstractmethod
    def preprocess(self, data: dict) -> Any:
        """Transform raw input (bytes, json) to model input."""
        raise NotImplementedError


    @abstractmethod
    def predict(self, processed: Any) -> Any:
        """Run inference and return raw output."""
        raise NotImplementedError


    @abstractmethod
    def postprocess(self, raw_output: Any) -> dict:
        """Convert raw model output to serializable dict."""
        raise NotImplementedError