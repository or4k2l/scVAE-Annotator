"""
scVAE-Annotator: Automated cell type annotation for single-cell RNA-seq data.
"""

__version__ = "0.1.0"
__author__ = "scVAE-Annotator Team"

from .annotator import Annotator
from .model import VAEModel

__all__ = ["Annotator", "VAEModel", "__version__"]
