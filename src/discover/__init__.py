"""
DISCOVER: A Python Implementation of Data-Informed Symbolic Combination of Operators for Variable Equation Regression
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
    DiscoverRegressor,
    DiscoverClassifier,
    DiscoverLogRegressor,
    DiscoverCHClassifier
)
from .utils import print_descriptor_formula

__all__ = [
    'DiscoverRegressor',
    'DiscoverClassifier',
    'DiscoverLogRegressor',
    'DiscoverCHClassifier',
    'print_descriptor_formula'
]