# -*- coding: utf-8 -*-
"""
This module contains the primary user-facing classes of the DISCOVER package.
It defines the `DiscoverBase` class, which orchestrates the entire workflow from
feature generation to model selection, and the Scikit-learn compatible wrapper
classes (`DiscoverRegressor`, `DiscoverClassifier`, etc.).

Lightweight linear-model wrapper for a SISSO descriptor.

Stores:
    • coefficients      (β₁, β₂, …)
    • intercept         (β₀)
    • SymPy expressions (descriptor terms)

Example:
    Expressions :  [A/B]               # 1-D descriptor
    Coeffs      :  [1.23]
    Intercept   :  −0.45
    ⇒ y ≈ −0.45 + 1.23 · (A/B)

"""
import pandas as pd
import numpy as np
import sympy
import warnings
import time
from pathlib import Path

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score, accuracy_score
from scipy.spatial import distance

from .constants import (
    REGRESSION, MULTITASK, CH_CLASSIFICATION, CLASSIFICATION_SVM,
    CLASSIFICATION_LOGREG, ALL_CLASSIFICATION_TASKS
)
from .features import generate_features_iteratively
from .features import CUPY_AVAILABLE, MPS_AVAILABLE
from .scoring import _run_cv, GPUModel

# --- Import all search functions ---
from .search import (
    _find_best_models_brute_force,
    _find_best_models_greedy,
    _find_best_models_sisso_pp,
    _find_best_models_ch_greedy,
    _refine_model_with_nlopt,
    _find_best_models_omp,
    _find_best_models_miqp
)

from .utils import save_results, print_descriptor_formula
from . import utils as plot_utils

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


class _BootstrapOutOfBagSplitter:
    """
    A scikit-learn compatible splitter that generates bootstrap samples.
    """
    def __init__(self, n_splits=30, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        for _ in range(self.n_splits):
            train_idx = self.rng_.choice(indices, size=n_samples, replace=True)
            test_mask = np.ones(n_samples, dtype=bool)
            test_mask[np.unique(train_idx)] = False
            test_idx = indices[test_mask]
            if len(test_idx) == 0: continue
            yield train_idx, test_idx


class DiscoverBase(BaseEstimator):
    """
    Base class for SISSO models, handling configuration, workflow orchestration,
    and results management.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self._parse_config(kwargs)
        self.feature_space_df_ = None
        self.feature_space_sym_map_ = None
        self.sym_clean_to_original_map_ = None
        self.primary_symbols_ = None
        self.models_by_dim_ = {}
        self.cv_results_ = {}
        self.best_descriptor_ = None
        self.best_model_ = None
        self.best_coef_ = None
        self.best_D_ = None
        self.classes_ = None
        self.timing_summary_ = {}
        self.xp_ = np
        self.torch_device_ = None

    def _parse_config(self, config):
        """
        Parses the configuration dictionary and sets class attributes with defaults.
        """
        valid_selection_methods = ['cv', 'bootstrap', 'aic', 'bic']
        
        # Rename legacy keys for backward compatibility
        if 'task_type' in config: config.setdefault('calc_type', config['task_type'])
        if 'max_rung' in config: config.setdefault('depth', config['max_rung'])
        if 'desc_dim' in config: config.setdefault('max_D', config['desc_dim'])
        if 'top_k' in config: config.setdefault('n_sis_select', config['top_k'])
        if 'num_cpu_threads' in config: config.setdefault('n_jobs', config['num_cpu_threads'])
        
        calc_type = config.get('calc_type', REGRESSION)
        sis_select = config.get('n_sis_select')
        self._task_type_raw = calc_type
        self.primary_units = config.get('primary_units', {})
        self.depth = config.get('depth', 2)

        if 'op_rules' in config:
            self.op_rules = config['op_rules']
        elif 'opset' in config:
            warnings.warn("'opset' is deprecated. Use 'op_rules' for advanced control. Converting for this run.")
            self.op_rules = [{"op": op_name} for op_name in config['opset']]
        else:
            default_opset = ['add', 'sub', 'mul', 'div', 'sq', 'cb', 'sqrt', 'cbrt', 'inv', 'abs']
            self.op_rules = [{"op": op_name} for op_name in default_opset]
        
        self.parametric_ops = config.get('parametric_ops', [])
        self.interaction_only = config.get('interaction_only', False)
        self.unary_rungs = config.get('unary_rungs', 1)

        self.min_abs_feat_val = config.get('min_abs_feat_val', 1e-50)
        self.max_abs_feat_val = config.get('max_abs_feat_val', 1e+50)
        
        self.search_strategy = config.get('search_strategy', 'greedy').lower()
        if self.search_strategy not in ['greedy', 'brute_force', 'sisso++', 'omp', 'miqp']:
            raise ValueError(f"search_strategy must be one of 'greedy', 'brute_force', 'sisso++', 'omp', 'miqp'. Got '{self.search_strategy}'")
        
        self.multitask_sis_method = config.get('multitask_sis_method', 'average').lower()
        if self.multitask_sis_method not in ['average', 'cca']:
            raise ValueError("`multitask_sis_method` must be 'average' or 'cca'.")
        self.nlopt_max_iter = config.get('nlopt_max_iter', 100)
        self.beam_width_decay = config.get('beam_width_decay', 1.0)
        self.max_D = config.get('max_D', 3)
        # The 'restarts' with different SIS sizes is removed in favor of the correct iterative methodology.
        self.sis_sizes = config.get('sis_sizes', [sis_select] if sis_select else [100])
        if not isinstance(self.sis_sizes, list): self.sis_sizes = [self.sis_sizes]
        self.n_features_per_sis_iter = self.sis_sizes[0]
        if len(self.sis_sizes) > 1:
            warnings.warn(f"Multiple `sis_sizes` provided. In the iterative SISSO framework, only the first value ({self.n_features_per_sis_iter}) will be used for pruning at each depth.")

        self.max_feat_cross_correlation = config.get('max_feat_cross_correlation', 0.95)
        self.loss = config.get('loss', 'l2')
        self.fix_intercept_ = config.get('fix_intercept', False)
        self.alpha = config.get('alpha', 1e-5)
        self.C_svm = config.get('C_svm', 1.0)
        self.C_logreg = config.get('C_logreg', 1.0)
        self.selection_method = config.get('selection_method', 'cv').lower()
        if self.selection_method not in valid_selection_methods:
            raise ValueError(f"selection_method must be one of {valid_selection_methods}")
        
        self.sis_score_degeneracy_epsilon = config.get('sis_score_degeneracy_epsilon', 1e-4)
            
        self.n_bootstrap = config.get('n_bootstrap', 30)
        self.cv = config.get('cv', 5)
        self.cv_score_tolerance = config.get('cv_score_tolerance', 0.02)
        self.cv_min_improvement = config.get('cv_min_improvement', 0.0)
        self.n_jobs = config.get('n_jobs', -1)
        self.device = config.get('device', 'cpu').lower()
        self.gpu_id = config.get('gpu_id', 0)
        self.dtype = 'float32' if config.get('use_single_precision', False) else config.get('dtype', 'float64')
        self.workdir = config.get('workdir', None)
        self.save_feature_space = config.get('save_feature_space', False)
        self.use_cache = config.get('use_cache', True)
        self.random_state = config.get('random_state', None)
        self.model_params_ = {
            'alpha': self.alpha, 'C_svm': self.C_svm, 'C_logreg': self.C_logreg,
            'loss': self.loss, 'fit_intercept': not self.fix_intercept_
        }

    def get_params(self, deep=True):
        """Gets parameters for this estimator, Scikit-learn compatible."""
        params = {
            'primary_units': self.primary_units, 'depth': self.depth, 'op_rules': self.op_rules,
            'parametric_ops': self.parametric_ops, 'interaction_only': self.interaction_only,
            'unary_rungs': self.unary_rungs,
            'min_abs_feat_val': self.min_abs_feat_val, 'max_abs_feat_val': self.max_abs_feat_val,
            'search_strategy': self.search_strategy, 
            'multitask_sis_method': self.multitask_sis_method,
            'nlopt_max_iter': self.nlopt_max_iter,
            'beam_width_decay': self.beam_width_decay, 'max_D': self.max_D, 'sis_sizes': self.sis_sizes,
            'max_feat_cross_correlation': self.max_feat_cross_correlation, 'loss': self.loss,
            'fix_intercept_': self.fix_intercept_, 'alpha': self.alpha, 'C_svm': self.C_svm,
            'C_logreg': self.C_logreg, 'selection_method': self.selection_method,
            'sis_score_degeneracy_epsilon': self.sis_score_degeneracy_epsilon,
            'n_bootstrap': self.n_bootstrap, 'cv': self.cv, 'cv_score_tolerance': self.cv_score_tolerance,
            'cv_min_improvement': self.cv_min_improvement, 'n_jobs': self.n_jobs, 'device': self.device,
            'gpu_id': self.gpu_id, 'dtype': self.dtype, 'workdir': self.workdir,
            'save_feature_space': self.save_feature_space, 'use_cache': self.use_cache,
            'random_state': self.random_state, '_task_type_raw': self._task_type_raw
        }
        if 'opset' in self.__dict__:
             params['opset'] = self.opset
        return params

    def set_params(self, **params):
        """Sets the parameters of this estimator, Scikit-learn compatible."""
        self._parse_config(params)
        return self
    
    def _setup_task(self, X, y):
        """Prepares the environment, data, and task type for the fit method."""
        if self.device == 'cuda':
            if not CUPY_AVAILABLE: raise ImportError("device='cuda' requires cupy. Please `pip install cupy-cudaXX`.")
            self.xp_ = cp
            if cp: cp.cuda.Device(self.gpu_id).use()
        elif self.device == 'mps':
                    if not MPS_AVAILABLE:
                        # Construct a more helpful error message
                        error_message = (
                            "device='mps' requires a PyTorch installation with MPS support on an Apple Silicon Mac.\n"
                            "It seems PyTorch is not installed or is not a compatible version.\n\n"
                            "To fix this, please install the latest version of PyTorch by running:\n"
                            "    pip install torch torchvision torchaudio\n\n"
                            "Make sure you are running this command in your project's virtual environment."
                        )
                        raise ImportError(error_message)
                    
                    self.xp_, self.torch_device_ = torch, torch.device("mps")
                    if self.dtype == 'float64':
                        warnings.warn("MPS device does not support float64. Forcing dtype to 'float32' for this run.")
                        self.dtype = 'float32'
        else: self.xp_ = np
        if np.dtype(self.dtype).type == np.float32:
            f32_max = np.finfo(np.float32).max
            if self.max_abs_feat_val > f32_max:
                warnings.warn(f"Config 'max_abs_feat_val' ({self.max_abs_feat_val:.2e}) exceeds float32 max ({f32_max:.2e}). "
                                f"Capping it to {f32_max * 0.9:.2e} for this run to avoid overflows.")
                self.max_abs_feat_val = f32_max * 0.9  # Use a safety margin
        self.model_params_['dtype'] = self.dtype
        if self.loss == 'huber' and self.device in ['cuda', 'mps']:
            warnings.warn("HuberRegressor is not GPU-accelerated. Search will run on CPU.")
            self.device = 'cpu'
        if self.parametric_ops and self.device in ['cuda', 'mps']:
             if self.task_type_ != REGRESSION:
                  warnings.warn("Non-linear optimization is currently only supported for regression tasks. Disabling.")
                  self.parametric_ops = []
             else:
                  warnings.warn("Non-linear optimization is CPU-only. This step will run on the CPU.")

        X_df = (pd.DataFrame(X).astype(self.dtype) if not isinstance(X, pd.DataFrame) else X.copy().astype(self.dtype))
        
        task_map = {'regression': REGRESSION, 'multitask': MULTITASK, 'classification_svm': CLASSIFICATION_SVM,
                    'classification_logreg': CLASSIFICATION_LOGREG, 'ch_classification': CH_CLASSIFICATION,
                    'convex_hull': CH_CLASSIFICATION, 'classification': CLASSIFICATION_SVM}
        self.task_type_ = task_map.get(self._task_type_raw.lower(), REGRESSION)

        is_multitask_input = (isinstance(y, pd.DataFrame) and y.shape[1] > 1) or \
                             (isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[1] > 1)

        if self.task_type_ == MULTITASK or (self.task_type_ == REGRESSION and is_multitask_input):
            self.task_type_ = MULTITASK
            y_out = pd.DataFrame(y, index=X_df.index).astype(self.dtype) if not isinstance(y, pd.DataFrame) else y.copy().astype(self.dtype)
        elif self.task_type_ in ALL_CLASSIFICATION_TASKS:
            y_out = pd.Series(y, index=X_df.index, name='target') if not isinstance(y, pd.Series) else y.copy()
            self.classes_ = np.unique(y_out)
            if len(self.classes_) < 2: raise ValueError("Classification requires at least 2 classes.")
        else:
            self.task_type_ = REGRESSION
            y_out = pd.Series(y, index=X_df.index, name='target').astype(self.dtype) if not isinstance(y, pd.Series) else y.copy().astype(self.dtype)
        
        return X_df, y_out

    def _select_best_dimension_cv(self):
        title = "Bootstrap OOB Score Summary" if self.selection_method == 'bootstrap' else "Cross-Validation Score Summary"
        print(f"\n--- {title} (Score to Minimize) ---")
        print(" D | Mean Score | Std Dev"); print("---|------------|----------")
        best_score, abs_best_D = float('inf'), -1
        dims = sorted(self.cv_results_.keys())
        for D in dims:
             mean_s, std_s = self.cv_results_[D]
             print(f"{D:2d} | {mean_s:10.4g} | {std_s:8.4g}")
             if mean_s < best_score - self.cv_min_improvement:
                 best_score, abs_best_D = mean_s, D
        print("---|------------|----------")
        if abs_best_D == -1: print("Model selection failed for all dimensions."); return None

        self.best_D_ = abs_best_D
        score_threshold = best_score * (1.0 + self.cv_score_tolerance) if best_score >=0 else best_score * (1.0 - self.cv_score_tolerance)
        for D in dims:
             if D > abs_best_D: continue
             mean_s, _ = self.cv_results_[D]
             if mean_s <= score_threshold:
                  self.best_D_ = D
                  print(f"Selected Best D = {self.best_D_} (Score: {mean_s:.4g}, within {self.cv_score_tolerance*100:.1f}% of best {best_score:.4g} at D={abs_best_D})")
                  break
        return self.best_D_

    def _select_best_dimension_info_crit(self):
        """Selects the best dimension by finding the minimum AIC or BIC score."""
        title = f"{self.selection_method.upper()} Score Summary"
        print(f"\n--- {title} (Score to Minimize) ---")
        print(" D | Score"); print("---|-------")
        
        best_score, self.best_D_ = float('inf'), -1
        for D, (score, _) in sorted(self.cv_results_.items()):
            print(f"{D:2d} | {score:10.4g}")
            if score < best_score:
                best_score, self.best_D_ = score, D
        print("---|-------")
        
        if self.best_D_ == -1:
            print("Model selection failed for all dimensions.")
            return None
            
        print(f"Selected Best D = {self.best_D_} (Lowest {self.selection_method.upper()} Score: {best_score:.4g})")
        return self.best_D_

    def fit(self, X, y, sample_weight=None):
        t_start_fit = time.time()
        if self.workdir: Path(self.workdir).mkdir(parents=True, exist_ok=True)
        if self.random_state is not None:
            np.random.seed(self.random_state)
            if torch and hasattr(torch, 'manual_seed'): torch.manual_seed(self.random_state)

        X_df, y_s = self._setup_task(X, y)
        
        # `sis_sizes` is removed as it's superseded by the iterative methodology.
        print(f"\n{'='*25}\n--- STARTING ITERATIVE FEATURE GENERATION & SCREENING ---\n{'='*25}")
        t_start_feat_gen = time.time()
        
        (self.feature_space_df_,
         self.feature_space_sym_map_,
         self.sym_clean_to_original_map_) = generate_features_iteratively(
            X=X_df, y=y_s, primary_units=self.primary_units, depth=self.depth,
            n_features_per_sis_iter=self.n_features_per_sis_iter,
            sis_score_degeneracy_epsilon=self.sis_score_degeneracy_epsilon,
            n_jobs=self.n_jobs, use_cache=self.use_cache, workdir=self.workdir,
            op_rules=self.op_rules, parametric_ops=self.parametric_ops,
            min_abs_feat_val=self.min_abs_feat_val,
            max_abs_feat_val=self.max_abs_feat_val,
            interaction_only=self.interaction_only,
            task_type=self.task_type_,
            multitask_sis_method=self.multitask_sis_method,
            unary_rungs=self.unary_rungs,
            xp=self.xp_, dtype=np.dtype(self.dtype), torch_device=self.torch_device_
        )
        self.primary_symbols_ = list(self.sym_clean_to_original_map_.keys())

        self.timing_summary_['Feature Generation'] = time.time() - t_start_feat_gen
        print(f"\nFinal feature space size after iterative screening: {self.feature_space_df_.shape[1]}")
        if self.feature_space_df_.empty:
            raise RuntimeError("Iterative feature generation resulted in an empty space.")
        self.feature_space_df_ = self.feature_space_df_.astype(self.dtype)

        # The rest of the `fit` method now proceeds with the high-quality feature space
        t_start_search = time.time()

        search_args = {"sisso_instance": self, "phi_sis_df": self.feature_space_df_, "y": y_s, "D_max": self.max_D,
                       "task_type": self.task_type_, "max_feat_cross_corr": self.max_feat_cross_correlation,
                       "sample_weight": sample_weight, "device": self.device, "torch_device": self.torch_device_,
                       "X_df": X_df}
        
        # Dispatch logic for search strategies remains the same
        if self.task_type_ == CH_CLASSIFICATION:
            warnings.warn("Task is 'ch_classification'. Overriding search_strategy with specialized 'geometric_greedy' search.")
            search_results = _find_best_models_ch_greedy(**search_args)
            self.timing_summary_['Geometric Greedy Search'] = time.time() - t_start_search
        elif self.search_strategy == 'brute_force':
            search_results = _find_best_models_brute_force(**search_args)
            self.timing_summary_['Brute-Force Search'] = time.time() - t_start_search
        elif self.search_strategy == 'sisso++':
            search_results = _find_best_models_sisso_pp(**search_args)
            self.timing_summary_['SISSO++ Search'] = time.time() - t_start_search
        elif self.search_strategy == 'omp':
            search_results = _find_best_models_omp(**search_args)
            self.timing_summary_['OMP Search'] = time.time() - t_start_search
        elif self.search_strategy == 'miqp':
            search_results = _find_best_models_miqp(**search_args)
            self.timing_summary_['MIQP Search'] = time.time() - t_start_search
        else: # 'greedy' is the default
            search_results = _find_best_models_greedy(**search_args)
            self.timing_summary_['Greedy Search'] = time.time() - t_start_search
        
        if not search_results:
            raise RuntimeError("L0 search found no models from the iteratively generated feature space.")

        self.models_by_dim_ = {}
        if self.parametric_ops and self.task_type_ == REGRESSION:
            t_start_nlopt = time.time()
            for D, model_data in search_results.items():
                if model_data.get('coef') is not None:
                    refined_model = _refine_model_with_nlopt(self, y_s, sample_weight, model_data)
                    self.models_by_dim_[D] = refined_model
                else:
                    self.models_by_dim_[D] = model_data
            self.timing_summary_['NLopt Refinement'] = time.time() - t_start_nlopt
        else:
            self.models_by_dim_ = search_results

        t_start_selection = time.time()
        self.cv_results_ = {}
        n_samples = len(X_df)
        
        # Model selection logic remains the same
        if self.selection_method in ['cv', 'bootstrap']:
            cv_splitter = None
            if self.selection_method == 'bootstrap':
                print(f"\n--- Evaluating Dimensions using {self.n_bootstrap} Bootstrap OOB Samples ---")
                cv_splitter = _BootstrapOutOfBagSplitter(n_splits=self.n_bootstrap, random_state=self.random_state)
            elif self.cv is not None and self.cv > 1:
                if self.cv == -1 or self.cv >= n_samples:
                    print("\n--- Evaluating Dimensions using Leave-One-Out CV (LOOCV) ---")
                    cv_splitter = LeaveOneOut()
                else:
                    print(f"\n--- Evaluating Dimensions using {self.cv}-Fold CV ---")
                    if self.task_type_ in ALL_CLASSIFICATION_TASKS:
                        min_class_count = pd.Series(y_s).value_counts().min()
                        cv_folds = min(self.cv, min_class_count)
                        if cv_folds >= 2:
                            if cv_folds < self.cv: warnings.warn(f"Smallest class has {min_class_count} members. Reducing CV folds to {cv_folds}.")
                            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    else: 
                        cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            
            if cv_splitter and self.models_by_dim_:
                for D, model_data in self.models_by_dim_.items():
                    print(f"  Evaluating D={D}...")
                    if model_data.get('is_parametric'):
                         self.cv_results_[D] = (model_data['score'], 0.0)
                         print(f"    Parametric model: using training score {model_data['score']:.4g} for evaluation.")
                         continue
                    
                    X_d = self.feature_space_df_[model_data['features']]
                    mean_score, std_score = _run_cv(X_d, y_s, cv_splitter, self.task_type_, self.model_params_, sample_weight)
                    if np.isfinite(mean_score): self.cv_results_[D] = (mean_score, std_score)
                    else: print(f"   Evaluation failed for D={D}")
            
            self._select_best_dimension_cv()

        elif self.selection_method in ['aic', 'bic']:
            print(f"\n--- Evaluating Dimensions using {self.selection_method.upper()} ---")
            if self.task_type_ != REGRESSION:
                raise ValueError(f"'{self.selection_method.upper()}' selection is only available for regression tasks.")
            
            for D, model_data in self.models_by_dim_.items():
                rmse = model_data['score']
                if not np.isfinite(rmse): continue
                
                rss = (rmse ** 2) * n_samples
                k = D + (1 if not self.fix_intercept_ else 0)
                
                if rss <= 1e-12:
                    score = -np.inf 
                else:
                    if self.selection_method == 'aic':
                        score = n_samples * np.log(rss / n_samples) + 2 * k
                    else: # bic
                        score = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)

                self.cv_results_[D] = (score, 0.0)
            
            self._select_best_dimension_info_crit()
        
        else:
            print("\nNo validation method specified. Selecting model with lowest training score.")
            if self.models_by_dim_:
                 self.best_D_ = min(self.models_by_dim_, key=lambda D: self.models_by_dim_[D]['score'])
                 print(f"Selected Best D = {self.best_D_}")
            for D, model_data in self.models_by_dim_.items(): self.cv_results_[D] = (model_data['score'], 0.0)
        
        self.timing_summary_['Model Selection'] = time.time() - t_start_selection
            
        if self.best_D_ is None:
             raise RuntimeError("Fit failed: No valid model found after model selection.")
        
        final_model_info = self.models_by_dim_[self.best_D_]
        
        if not final_model_info.get('is_parametric', False) and self.task_type_ == REGRESSION:
            print("\n--- Post-processing Final Model (Computing Diagnostics) ---")
            t_start_post = time.time()
            
            X_df_clean = X_df.copy()
            X_df_clean.columns = list(self.sym_clean_to_original_map_.keys())
            
            X_desc_final = self._generate_descriptor_data(
                X_df_clean,
                final_model_info['sym_features'],
                final_model_info['features'],
                self.dtype
            )
            final_model_info = self._post_process_final_model(final_model_info, X_desc_final, y_s, sample_weight)
            self.models_by_dim_[self.best_D_] = final_model_info
            self.timing_summary_['Post-processing'] = time.time() - t_start_post
        
        self.best_descriptor_ = final_model_info['sym_features']
        self.best_model_ = final_model_info['model']
        self.best_coef_ = final_model_info.get('coef')

        if self.workdir: save_results(self, X_df, y_s, sample_weight)
        
        self.timing_summary_['Total Fit Time'] = time.time() - t_start_fit
        print("\n" + "="*25 + " Timing Summary " + "="*25)
        for stage, duration in self.timing_summary_.items(): print(f"  - {stage:<30}: {duration:.2f} seconds")
        print("="*68 + "\nFit complete.")
        return self

    def _post_process_final_model(self, final_model_info, X_desc, y, sample_weight):
        """
        Calculates VIF, coefficient uncertainties, and performs orthogonalisation
        for the final selected linear model.
        """
        D = len(final_model_info['features'])
        if self.task_type_ != REGRESSION or D < 1:
            return final_model_info
        
        print("  - Calculating coefficient standard errors...")
        try:
            X_mat = X_desc.values
            if not self.fix_intercept_:
                X_mat = np.c_[np.ones(X_mat.shape[0]), X_mat]
            
            y_pred = final_model_info['model'].predict(X_desc)

            n_samples = len(y)
            n_params = X_mat.shape[1]
            if n_samples > n_params:
                residuals = y.values - y_pred
                rss = np.sum(residuals**2)
                mse = rss / (n_samples - n_params)
                
                xtx_inv = np.linalg.inv(X_mat.T @ X_mat)
                coef_cov_matrix = mse * xtx_inv
                coef_stderr = np.sqrt(np.diag(coef_cov_matrix))
                final_model_info['coef_stderr'] = coef_stderr
            else:
                 print("  - Warning: Cannot compute errors, more parameters than samples.")
        except (np.linalg.LinAlgError, ValueError):
            print("  - Warning: Could not compute coefficient errors (singular matrix).")

        if D >= 2:
            print("  - Calculating Variance Inflation Factors (VIF)...")
            vif_data = {}
            for i in range(D):
                target_feat = X_desc.iloc[:, i]
                predictors = X_desc.drop(X_desc.columns[i], axis=1)
                vif_model = LinearRegression(fit_intercept=True).fit(predictors, target_feat)
                r_squared = vif_model.score(predictors, target_feat)
                vif = 1 / (1 - r_squared) if r_squared < 1.0 else float('inf')
                vif_data[final_model_info['features'][i]] = vif
            final_model_info['vif'] = vif_data
        else:
            print("  - Skipping VIF (requires D>=2).")

        if D >= 2:
            print("  - Performing Gram-Schmidt orthogonalisation...")
            try:
                Q, R = np.linalg.qr(X_desc.values)
                Q_df = pd.DataFrame(Q, index=X_desc.index, columns=[f"ortho_{i+1}" for i in range(D)])
                ortho_model = LinearRegression(fit_intercept=not self.fix_intercept_).fit(Q_df, y, sample_weight=sample_weight)
                
                ortho_coefs = np.insert(ortho_model.coef_, 0, ortho_model.intercept_) if not self.fix_intercept_ else ortho_model.coef_
                
                formula_parts = []
                if not self.fix_intercept_:
                    formula_parts.append(f"{ortho_coefs[0]:+.4f}")

                for i in range(D):
                    coef_val = ortho_coefs[i if self.fix_intercept_ else i+1]
                    formula_parts.append(f"{coef_val:+.4f} * (D'_{i+1})")
                
                final_model_info['orthogonal_model_formula'] = f"{' '.join(formula_parts)}"
            except np.linalg.LinAlgError:
                print("  - Warning: Could not perform orthogonalisation.")
        else:
            print("  - Skipping orthogonalisation (requires D>=2).")

        return final_model_info

    def _generate_descriptor_data(self, X_primary, descriptor_sym_expr, descriptor_names, dtype):
        """
        Internal helper to compute descriptor values from symbolic expressions.
        """
        transformed_X = pd.DataFrame(index=X_primary.index, dtype=dtype)
        
        primary_symbols = self.primary_symbols_
        feature_values = [X_primary[col].values for col in X_primary.columns]
        
        for i, expr in enumerate(descriptor_sym_expr):
            try:
               lambda_func = sympy.lambdify(primary_symbols, expr, 'numpy')
               col_name = descriptor_names[i]
               transformed_X[col_name] = lambda_func(*feature_values)
            except Exception as e:
                raise ValueError(f"Error evaluating descriptor {expr}: {e}") from e
        return transformed_X

    def _transform_X(self, X):
        """Transforms a primary feature matrix X into the final descriptor space."""
        if self.best_model_ is None: raise RuntimeError("You must call fit() first.")
        is_parametric = hasattr(self.best_model_, 'is_parametric') and self.best_model_.is_parametric
        if is_parametric:
            raise RuntimeError("_transform_X is not applicable for a parametric model. Use predict() directly.")

        X_df = pd.DataFrame(X).astype(self.dtype) if not isinstance(X, pd.DataFrame) else X.astype(self.dtype)
        
        X_df_clean = X_df.copy()
        X_df_clean.columns = list(self.sym_clean_to_original_map_.keys())

        return self._generate_descriptor_data(
            X_df_clean,
            self.best_descriptor_,
            self.models_by_dim_[self.best_D_]['features'],
            self.dtype
        )

    def predict(self, X):
        if self.best_model_ is None: raise RuntimeError("Call fit() before predict().")

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        is_parametric = hasattr(self.best_model_, 'is_parametric') and self.best_model_.is_parametric
        if is_parametric:
            if len(X_df.columns) != len(self.sym_clean_to_original_map_):
                raise ValueError(f"Prediction input has {len(X_df.columns)} columns, but model was trained on {len(self.sym_clean_to_original_map_)} primary features.")
            
            X_df_clean = X_df.copy()
            X_df_clean.columns = [str(s) for s in self.sym_clean_to_original_map_.keys()]
            return self.best_model_.predict(X_df_clean.astype(self.dtype))

        transformed_X = self._transform_X(X_df)

        if isinstance(self.best_model_, GPUModel):
            y_pred_gpu = self.best_model_.predict(transformed_X.values)
            if self.best_model_.device == 'cuda': return cp.asnumpy(y_pred_gpu)
            elif self.best_model_.device == 'mps': return y_pred_gpu.cpu().numpy()
        elif self.task_type_ == CH_CLASSIFICATION:
            predictions = np.full(len(X), -1, dtype=object)
            centroids = {c: np.mean(hull.points, axis=0) for c, hull in self.best_model_.items()}
            for i, point in enumerate(transformed_X.values):
                 inside_classes = []
                 for c, hull in self.best_model_.items():
                     if np.all(np.add(np.dot(point, hull.equations[:, :-1].T), hull.equations[:, -1]) <= 1e-9): inside_classes.append(c)
                 if len(inside_classes) == 1: predictions[i] = inside_classes[0]
                 elif len(inside_classes) > 1:
                      dists = {c: distance.euclidean(point, centroids[c]) for c in inside_classes}
                      predictions[i] = min(dists, key=dists.get)
            return predictions
        elif self.task_type_ == MULTITASK:
             return pd.DataFrame({task: model.predict(transformed_X) for task, model in self.best_model_.items()})
        else: return self.best_model_.predict(transformed_X)

    def predict_proba(self, X):
         is_parametric = hasattr(self.best_model_, 'is_parametric') and self.best_model_.is_parametric
         if is_parametric:
             raise AttributeError("predict_proba not supported for optimized parametric models.")
         if not hasattr(self.best_model_, 'predict_proba'):
              raise AttributeError(f"Model for task '{self.task_type_}' does not support predict_proba.")
         return self.best_model_.predict_proba(self._transform_X(X))

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if self.task_type_ in [REGRESSION, MULTITASK]:
            return r2_score(y, y_pred, sample_weight=sample_weight)
        else:
            y_true_np = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
            if self.task_type_ == CH_CLASSIFICATION:
                mask = y_pred != -1
                if not np.any(mask): return 0.0
                y_true_filtered, y_pred_filtered = y_true_np[mask], y_pred[mask]
                if y_pred_filtered.dtype != y_true_filtered.dtype: y_pred_filtered = y_pred_filtered.astype(y_true_filtered.dtype)
                sw_filtered = np.asarray(sample_weight)[mask] if sample_weight is not None else None
                return accuracy_score(y_true_filtered, y_pred_filtered, sample_weight=sw_filtered)
            else:
                return accuracy_score(y_true_np, y_pred, sample_weight=sample_weight)

    def best_model_summary(self, target_name='y'):
        """Returns a string containing the symbolic formula of the best model."""
        if self.best_D_ is None or self.best_D_ not in self.models_by_dim_:
            return "No best model found. Please fit the model first."
        
        model_info = self.models_by_dim_[self.best_D_]
        return print_descriptor_formula(
            model_info['sym_features'], model_info.get('coef'), self.task_type_,
            self.fix_intercept_, target_name=target_name,
            model_provider=model_info.get('model'),
            coefficients_stderr=model_info.get('coef_stderr'),
            clean_to_original_map=self.sym_clean_to_original_map_
        )

    def best_model_latex(self, target_name='y'):
        """Returns a LaTeX formatted string of the best model formula."""
        if self.best_D_ is None or self.best_D_ not in self.models_by_dim_:
            return "No best model found. Please fit the model first."
        model_info = self.models_by_dim_[self.best_D_]
        return print_descriptor_formula(
            model_info['sym_features'], model_info.get('coef'), self.task_type_,
            self.fix_intercept_, target_name=target_name, latex_format=True,
            model_provider=model_info.get('model'),
            coefficients_stderr=model_info.get('coef_stderr'),
            clean_to_original_map=self.sym_clean_to_original_map_
        )

    def summary_report(self, X, y, sample_weight):
        """Generates a detailed text summary report of the final results."""
        if self.best_D_ is None: return "Fit was not successful. No summary to report."
        final_model_is_parametric = hasattr(self.best_model_, 'is_parametric') and self.best_model_.is_parametric
        selection_str = self.selection_method.upper()
        if self.selection_method == 'cv':
            selection_str = f"{self.cv}-Fold CV" if self.cv != -1 else "LOOCV"
        elif self.selection_method == 'bootstrap':
            selection_str = f"{self.n_bootstrap} Bootstraps (OOB)"
        
        search_strat_str = self.search_strategy.capitalize()
        if self.task_type_ == CH_CLASSIFICATION:
            search_strat_str = "Geometric Greedy"

        report_lines = [
            "DISCOVER Summary", "=================", f"TASK_TYPE: {self.task_type_}",
            f"MODEL_SELECTION_METHOD: {selection_str}", f"SELECTED_DIMENSION: {self.best_D_}",
            f"SEARCH_STRATEGY: {search_strat_str}",
            f"PARAMETRIC_MODEL_FOUND: {'YES' if final_model_is_parametric else 'NO'}",
            f"FEATURE_DEPTH: {self.depth}", f"N_SAMPLES: {len(X)}",
            f"N_PRIMARY_FEATURES: {len(X.columns)}",
            f"N_TOTAL_FEATURES: {len(self.feature_space_df_.columns) if self.feature_space_df_ is not None else 'N/A'}",
        ]
        
        best_model_data = self.models_by_dim_[self.best_D_]
        best_cv_score, best_cv_std = self.cv_results_.get(self.best_D_, (np.nan, np.nan))
        score_label = plot_utils._get_score_label(self.task_type_, self.loss, self.selection_method).replace('(CV)', f'({selection_str})')
        
        report_lines.append(f"\n******* BEST MODEL (D={self.best_D_}) *******")
        report_lines.append(f"Score {score_label}: {best_cv_score:.6g} +/- {best_cv_std:.6g}")
        report_lines.append(f"R2/Accuracy on full data: {self.score(X, y, sample_weight):.4f}")
        
        formula = self.best_model_summary(target_name="P")
        report_lines.append(formula)
        
        if 'vif' in best_model_data or 'orthogonal_model_formula' in best_model_data:
            report_lines.append("\n--- Model Diagnostics ---")
        if 'vif' in best_model_data:
            report_lines.append("Variance Inflation Factors (VIF):")
            for feat, vif_val in best_model_data['vif'].items():
                report_lines.append(f"  - {feat[:40]:<42}: {vif_val:.4f}")
        
        if 'orthogonal_model_formula' in best_model_data:
            report_lines.append("\nOrthogonalized Model Formula (y = c0 + c1*D'1 + ...):")
            report_lines.append(f"  y = {best_model_data['orthogonal_model_formula']}")

        return "\n".join(report_lines)

    def plot_cv_results(self, **kwargs):
        return plot_utils.plot_cv_results(self.cv_results_, self.best_D_, self.task_type_, self.loss, self.selection_method, **kwargs)
    def plot_parity(self, X, y, **kwargs):
        return plot_utils.plot_parity(self, X, y, **kwargs)
    def plot_classification_2d(self, X, y, **kwargs):
        return plot_utils.plot_classification_2d(self, X, y, **kwargs)
    def plot_classification_3d(self, X, y, **kwargs):
        return plot_utils.plot_classification_3d(self, X, y, **kwargs)
    def plot_descriptor_pairplot(self, X, y, **kwargs):
        return plot_utils.plot_descriptor_pairplot(self, X, y, **kwargs)
    def plot_descriptor_histograms(self, X, **kwargs):
        return plot_utils.plot_descriptor_histograms(self, X, **kwargs)
    
    def plot_feature_importance(self, X, **kwargs):
        """Plots the scaled importance of each descriptor in the final model."""
        return plot_utils.plot_feature_importance(self, X, **kwargs)

    def plot_partial_dependence(self, X, features_to_plot=None, **kwargs):
        """
        Creates a Partial Dependence Plot (PDP) to visualize the model's response.
        """
        return plot_utils.plot_partial_dependence(self, X, features_to_plot, **kwargs)


class DiscoverRegressor(DiscoverBase, RegressorMixin):
    def __init__(self, **kwargs):
        kwargs.setdefault('calc_type', 'regression')
        super().__init__(**kwargs)

class DiscoverClassifier(DiscoverBase, ClassifierMixin):
    def __init__(self, **kwargs):
        kwargs.setdefault('calc_type', 'classification_svm')
        super().__init__(**kwargs)

class DiscoverLogRegressor(DiscoverBase, ClassifierMixin):
    def __init__(self, **kwargs):
        kwargs.setdefault('calc_type', 'classification_logreg')
        super().__init__(**kwargs)

class DiscoverCHClassifier(DiscoverBase, ClassifierMixin):
    def __init__(self, **kwargs):
        kwargs.setdefault('calc_type', 'ch_classification')
        super().__init__(**kwargs)
    
    def predict_proba(self, X):
        raise AttributeError("Convex Hull classifier does not support predict_proba.")