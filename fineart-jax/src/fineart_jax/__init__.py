"""FineART-VLA default semantics, data layer, in JAX.

Only the data layer is available at this stage: configuration, NumPy data
processing and tokenization. Importing this package must not touch a GPU.
"""

from .config import DEFAULT_RECIPE, FineARTJaxConfig, default_recipe

__all__ = ["DEFAULT_RECIPE", "FineARTJaxConfig", "default_recipe"]
__version__ = "0.1.0"
