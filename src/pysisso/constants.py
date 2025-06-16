# -*- coding: utf-8 -*-
"""
This module defines global constants used across the pySISSO package to ensure
consistency and avoid magic strings.
"""

# Task type constants
REGRESSION = 'regression'
MULTITASK = 'multitask'
CH_CLASSIFICATION = 'ch_classification'
CLASSIFICATION_SVM = 'classification_svm'
CLASSIFICATION_LOGREG = 'classification_logreg'
ALL_CLASSIFICATION_TASKS = [CH_CLASSIFICATION, CLASSIFICATION_SVM, CLASSIFICATION_LOGREG]

# A threshold for warning users about potentially long-running computations.
MAX_COMBINATIONS_WARNING_THRESHOLD = 2_000_000