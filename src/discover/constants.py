# -*- coding: utf-8 -*-
"""
This module defines global constants used across the DISCOVER package to ensure
consistency and avoid magic strings.
Global literals and hyper-parameter defaults used throughout DISCOVER.

Keeping them in one place avoids “magic numbers” in the core logic.

Example:
    POOL_SIZE_DEFAULT = 10000     # default number of candidate features
"""

# Task type constants
REGRESSION = 'regression'
MULTITASK = 'multitask'
CH_CLASSIFICATION = 'ch_classification'
CLASSIFICATION_SVM = 'classification_svm'
CLASSIFICATION_LOGREG = 'classification_logreg'
ALL_CLASSIFICATION_TASKS = [CH_CLASSIFICATION, CLASSIFICATION_SVM, 
                            CLASSIFICATION_LOGREG]

# A threshold for warning users about potentially long-running computations.
MAX_COMBINATIONS_WARNING_THRESHOLD = 2_000_000