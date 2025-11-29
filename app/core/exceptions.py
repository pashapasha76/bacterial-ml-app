class ModelNotFoundError(Exception):
    """Raised when a requested model/handler is not registered."""
    pass


class ModelLoadError(Exception):
    """Raised when a model fails to load."""
    pass


class InferenceError(Exception):
    """Raised on inference failure."""
    pass