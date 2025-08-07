"""
DISCOVER: A Python Implementation of Sure Independence Screening and Sparsifying Operator
========================================================
DISCOVER package initialiser.

Re-exports the key public classes so users can write

    >>> from discover import SISSOModel, SISSearcher

without digging into sub-modules.

Example – quick start:
    >>> from discover import run_from_config
    >>> run_from_config("config_mohsen.json")
"""
from .models import (
    SISSORegressor,
    SISSOClassifier,
    SISSOLogRegressor,
    SISSOCHClassifier
)
from .utils import print_descriptor_formula

__all__ = [
    'SISSORegressor',
    'SISSOClassifier',
    'SISSOLogRegressor',
    'SISSOCHClassifier',
    'print_descriptor_formula'
]