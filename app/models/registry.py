from typing import Dict
from app.models.base_handler import AbstractModelHandler
from typing import Optional
import threading
from contextlib import contextmanager


class ModelRegistry:
    def __init__(self):
        self._handlers: Dict[str, AbstractModelHandler] = {}
        self._current_model: Optional[str] = None
        self._lock = threading.RLock()


    def register(self, name: str, handler: AbstractModelHandler):
        self._handlers[name] = handler


    def get(self, name: str) -> Optional[AbstractModelHandler]:
        # Unload previous model if switching to a different one
        with self._lock:
            if self._current_model and self._current_model != name:
                previous_handler = self._handlers.get(self._current_model)
                if previous_handler and previous_handler._loaded:
                    previous_handler.unload()

            # Set current model
            self._current_model = name

            return self._handlers.get(name)
        
    
    def unload_all(self):
        """Unload all models from memory"""
        for name, handler in self._handlers.items():
            if handler._loaded:
                print(f"Unloading model: {name}")
                handler.unload()
        self._current_model = None


# single global registry instance
registry = ModelRegistry()