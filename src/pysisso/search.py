# --- START OF FILE search.py ---

# -*- coding: utf-8 -*-
"""
This module implements the different search strategies for finding the best
feature combinations (descriptors). It includes:
- Brute-force search: Exhaustively checks all combinations.
- Greedy search: Iteratively builds up descriptors one feature at a time.
- SISSO++ search: A breadth-first search using efficient QR-based updates.
- A standalone non-linear optimization function for model refinement.
"""
import numpy as np
import sympy
import time
import math
import warnings
from itertools import combinations
from scipy.optimize import minimize

from joblib import Parallel, delayed

from .constants import (
    MULTITASK, ALL_CLASSIFICATION_TASKS, MAX_COMBINATIONS_WARNING_THRESHOLD
)
from .scoring import _score_single_model, run_SIS, GPUModel, _refit_and_score_final_model
from .features import PARAMETRIC_OP_DEFS, CUPY_AVAILABLE, TORCH_AVAILABLE
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error

# Optional GPU imports
try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    # Add a minimal torch definition for type checking if not available
    class torch_dummy:
        @staticmethod
        def from_numpy(x): return x
        @staticmethod
        def linalg_qr(x): return np.linalg.qr(x)
        @staticmethod
        def mean(x, **kwargs): return np.mean(x, **kwargs)
        @staticmethod
        def sum(x, **kwargs): return np.sum(x, **kwargs)
        @staticmethod
        def dot(a, b): return np.dot(a, b)
        @staticmethod
        def isfinite(x): return np.isfinite(x)

    torch = torch_dummy()
except:
    pass # Keep going if torch import fails for other reasons


class ParametricModel:
    """A wrapper for a non-linear model defined by a symbolic expression."""
    def __init__(self, sym_expr, primary_symbols):
        self.sym_expr = sym_expr
        self.primary_symbols = primary_symbols
        self.is_parametric = True
        self._predict_func = sympy.lambdify(primary_symbols, self.sym_expr, 'numpy')

    def predict(self, X):
        """Predicts using the symbolic formula, requires primary features."""
        if not all(str(s) in X.columns for s in self.primary_symbols):
             raise ValueError(f"Prediction input missing primary features. Required: {[str(s) for s in self.primary_symbols]}")
        
        feature_values = [X[str(s)].values for s in self.primary_symbols]
        return self._predict_func(*feature_values)


def _prune_by_correlation(phi_df, threshold):
    """Removes features from a DataFrame that are highly correlated with each other."""
    if threshold >= 1.0:
        return phi_df
    print(f"  Pruning feature space with correlation threshold > {threshold}...")
    corr_matrix = phi_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    pruned_df = phi_df.drop(columns=to_drop)
    print(f"    Removed {len(to_drop)} features due to high cross-correlation. Kept {pruned_df.shape[1]}.")
    return pruned_df


# =============================================================================
# Non-Linear Optimization (NLopt) Refinement
# =============================================================================

def _refine_model_with_nlopt(sisso_instance, y, sample_weight, current_model_data):
    """
    Takes a linear model and attempts to improve it by applying parametric
    operators and performing non-linear optimization.
    """
    print(f"    -> Refining D={len(current_model_data['features'])} with non-linear optimization...")
    
    refined_model = current_model_data.copy()
    initial_linear_coefs = current_model_data['coef'].flatten()
    base_feature_names = current_model_data['features']
    base_sym_features = current_model_data['sym_features']
    base_features_values = sisso_instance.feature_space_df_[base_feature_names].values
    
    fit_intercept = sisso_instance.model_params_.get('fit_intercept', True)
    nlopt_max_iter = getattr(sisso_instance, 'nlopt_max_iter', 100)
    parametric_ops = getattr(sisso_instance, 'parametric_ops', [])

    for i, base_feat_name in enumerate(base_feature_names):
        for op_name in parametric_ops:
            op_def = PARAMETRIC_OP_DEFS.get(op_name)
            if not op_def: continue
            
            def objective_func(params):
                lin_coefs, nonlin_params = params[:len(initial_linear_coefs)], params[len(initial_linear_coefs):]
                y_pred = lin_coefs[0] if fit_intercept else 0
                
                current_desc_vals = base_features_values.copy()
                try:
                   # Apply the parametric function only to the i-th feature
                   current_desc_vals[:, i] = op_def['func'](base_features_values[:, i], nonlin_params)
                   if np.any(~np.isfinite(current_desc_vals[:, i])): return float('inf')
                except (ValueError, OverflowError): return float('inf')
                
                coef_offset = 1 if fit_intercept else 0
                y_pred += np.sum(lin_coefs[coef_offset:] * current_desc_vals, axis=1)
                
                return np.sqrt(mean_squared_error(y.values, y_pred, sample_weight=sample_weight))

            x0 = np.concatenate([initial_linear_coefs, op_def['p_initial']])
            bounds = [(None, None)] * len(initial_linear_coefs) + op_def['p_bounds']
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(objective_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': nlopt_max_iter})

            if result.success and np.isfinite(result.fun) and result.fun < refined_model['score']:
                print(f"      Found better non-linear model! Op: '{op_name}' on Feat: '{base_feat_name}'. Score: {result.fun:.6g}")
                
                refined_model['score'] = result.fun
                refined_model['is_parametric'] = True
                
                opt_lin_coefs = result.x[:len(initial_linear_coefs)]
                opt_nonlin_params = result.x[len(initial_linear_coefs):]
                
                p_symbols = [sympy.Symbol(p_name) for p_name in op_def['p_names']]
                temp_sym_features = base_sym_features.copy()
                temp_sym_features[i] = op_def['sym_func'](base_sym_features[i], p_symbols)
                
                full_model_expr = sympy.Number(opt_lin_coefs[0]) if fit_intercept else 0
                coef_offset = 1 if fit_intercept else 0
                for j, feat_sym in enumerate(temp_sym_features):
                    full_model_expr += sympy.Number(opt_lin_coefs[j + coef_offset]) * feat_sym
                
                final_expr = full_model_expr.subs(zip(p_symbols, opt_nonlin_params))
                
                # The symbolic features now represent the full model expression
                refined_model['sym_features'] = [final_expr]
                refined_model['model'] = ParametricModel(final_expr, sisso_instance.primary_symbols_)
                refined_model['coef'] = None # Coefficients are baked into the expression
    
    return refined_model


# =============================================================================
# Search Strategy Implementations
# =============================================================================

def _score_ch_combo(new_feature, base_features, phi_df, y, task_type, model_params, sample_weight):
    """Helper function to score a feature combination for the CH geometric search."""
    combo_features = base_features + [new_feature]
    X_combo_df = phi_df[combo_features]
    # For CH search, we don't use GPU scoring, so device args are fixed to CPU
    score, model_data = _score_single_model(X_combo_df, y, task_type, model_params, sample_weight, 'cpu', None)
    if model_data:
        return score, new_feature, model_data
    return float('inf'), new_feature, None


def _find_best_models_ch_greedy(sisso_instance, phi_sis_df, y, D_max, task_type,
                                max_feat_cross_corr, sample_weight, **kwargs):
    """
    Finds models for Convex Hull classification using a specialized greedy search
    based on the geometric overlap score, not a statistical proxy.
    """
    print("\n" + "="*20 + " Starting Geometric Greedy Search (Convex Hull) " + "="*20)

    models_by_dim = {}
    selected_features = []
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    candidate_features = list(phi_pruned.columns)
    model_params = sisso_instance.model_params_
    
    for D_iter in range(1, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- Geometric Greedy Search: Dimension {D_iter} ---")
        
        remaining_features = [f for f in candidate_features if f not in selected_features]
        if not remaining_features:
            print("  No more features to select. Stopping."); break
        
        print(f"  Evaluating {len(remaining_features)} candidate features to add to the descriptor...")
        
        tasks = (delayed(_score_ch_combo)(
            new_feat, selected_features, phi_pruned, y, task_type, model_params, sample_weight
        ) for new_feat in remaining_features)
        
        results = Parallel(n_jobs=sisso_instance.n_jobs, prefer="threads")(tasks)
        
        valid_results = [res for res in results if np.isfinite(res[0])]
        if not valid_results:
            print(f"  No valid models could be formed for D={D_iter}. Stopping."); break
            
        best_score, best_new_feature, best_model_data = min(valid_results, key=lambda x: x[0])
        
        # Check if the new model is an improvement (only for D>1)
        if D_iter > 1:
            prev_best_score = models_by_dim[D_iter-1]['score']
            if best_score >= prev_best_score:
                print(f"  No improvement found for D={D_iter}. Best D={D_iter-1} score was {prev_best_score:.4g}, current is {best_score:.4g}. Stopping.")
                break

        selected_features.append(best_new_feature)
        sisso_instance.timing_summary_[f'Geometric Search D={D_iter}'] = time.time() - t_start_d
        print(f"    Best feature for D={D_iter}: '{best_new_feature}'")
        print(f"    Best model score for D={D_iter} (overlap): {best_score:.6g}")

        sym_features = [sisso_instance.feature_space_sym_map_.get(f, sympy.sympify(f)) for f in selected_features]
        models_by_dim[D_iter] = {
            'features': selected_features.copy(),
            'score': best_score,
            'model': best_model_data.get('model'),
            'coef': None, # CH models don't have coefficients in the traditional sense
            'sym_features': sym_features,
            'is_parametric': False
        }

    return models_by_dim


def _score_brute_force_combo(combo, phi_df, y, task_type, model_params, sample_weight, device, torch_device):
    """Helper function for parallelized brute-force scoring."""
    X_combo_df = phi_df[list(combo)]
    score, model_data = _score_single_model(X_combo_df, y, task_type, model_params, sample_weight, device, torch_device)
    if model_data:
        return score, combo, model_data.get('coef'), model_data.get('model')
    return float('inf'), combo, None, None


def _find_best_models_brute_force(sisso_instance, phi_sis_df, y, D_max, task_type,
                                   max_feat_cross_corr, sample_weight, device, torch_device):
    """
    Finds the best model for each dimension by testing all combinations.
    """
    print("\n" + "="*20 + " Starting Brute-Force Search " + "="*20)
    print(f"INFO: Brute-force search will operate on the provided {phi_sis_df.shape[1]} features.")

    models_by_dim = {}
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    feature_candidates = list(phi_pruned.columns)
    n_features = len(feature_candidates)
    
    model_params = sisso_instance.model_params_
    timing_summary = sisso_instance.timing_summary_

    for D_iter in range(1, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- Brute-Force: Searching for Dimension {D_iter} ---")
        if D_iter > n_features:
            print(f"  Dimension {D_iter} is larger than the number of available features ({n_features}). Stopping.")
            break
        try:
            n_combos = math.comb(n_features, D_iter)
        except ValueError:
            print(f"  Cannot compute combinations for D={D_iter} from {n_features} features. Stopping."); break
        
        print(f"  Evaluating {n_combos:,} combinations of {D_iter} features.")
        if n_combos > MAX_COMBINATIONS_WARNING_THRESHOLD:
            warnings.warn(f"Number of combinations ({n_combos:,}) is very large. This may take a long time.")
        if n_combos == 0: continue
            
        combos = combinations(feature_candidates, D_iter)
        tasks = (delayed(_score_brute_force_combo)(
                    combo, phi_pruned, y, task_type, model_params, sample_weight, device, torch_device
                 ) for combo in combos)
        results = Parallel(n_jobs=sisso_instance.n_jobs, prefer="threads")(tasks)
        
        valid_results = [res for res in results if np.isfinite(res[0])]
        if not valid_results:
            print(f"  No valid models found for D={D_iter}. Stopping."); break

        best_score, best_combo, best_coef, best_model = min(valid_results, key=lambda x: x[0])
        timing_summary[f'Brute-Force Search D={D_iter}'] = time.time() - t_start_d
        print(f"    Best model for D={D_iter} found with score: {best_score:.6g}")
        
        sym_features = [sisso_instance.feature_space_sym_map_.get(f, sympy.sympify(f)) for f in best_combo]
        models_by_dim[D_iter] = {
            'features': list(best_combo), 'score': best_score, 'model': best_model,
            'coef': best_coef, 'sym_features': sym_features, 'is_parametric': False
        }
    return models_by_dim


def _find_best_models_greedy(sisso_instance, phi_sis_df, y, X_df, D_max, task_type,
                             max_feat_cross_corr, sample_weight, device, torch_device):
    """
    Finds models using a greedy, iterative approach (Sparsifying Operator).
    """
    print("\n" + "="*20 + " Starting Greedy (Sparsifying Operator) Search " + "="*20)
    print(f"INFO: Greedy search will operate on the provided {phi_sis_df.shape[1]} features.")
    
    models_by_dim = {}
    selected_features = []
    current_target = y.copy()
    
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    
    model_params = sisso_instance.model_params_
    timing_summary = sisso_instance.timing_summary_

    for D_iter in range(1, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- Greedy Search: Searching for Dimension {D_iter} ---")
        features_to_screen_df = phi_pruned.drop(columns=selected_features, errors='ignore')
        if features_to_screen_df.empty:
            print("  No more features to select. Stopping."); break

        sorted_candidates = run_SIS(features_to_screen_df, current_target, task_type, xp=sisso_instance.xp_)
        if not sorted_candidates:
            print(f"  SIS found no new correlated features. Stopping at D={D_iter-1}."); break

        best_new_feature = sorted_candidates[0]
        selected_features.append(best_new_feature)
        print(f"  Selected feature for D={D_iter}: '{best_new_feature}'")

        X_current_dim_df = sisso_instance.feature_space_df_[selected_features]
        score, model_data = _score_single_model(X_current_dim_df, y, task_type, model_params, sample_weight, device, torch_device)
        timing_summary[f'Greedy Search D={D_iter} (Linear)'] = time.time() - t_start_d

        if model_data is None:
            print(f"  Could not fit a valid linear model for D={D_iter}. Stopping.")
            selected_features.pop(); break

        current_sym_features = [sisso_instance.feature_space_sym_map_.get(f, sympy.sympify(f)) for f in selected_features]
        models_by_dim[D_iter] = {
            'features': selected_features.copy(), 'score': score, 'model': model_data.get('model'),
            'coef': model_data.get('coef'), 'sym_features': current_sym_features, 'is_parametric': False
        }
        print(f"    Linear model score for D={D_iter} (min): {score:.6g}")
        
        model_object = models_by_dim[D_iter]['model']
        if isinstance(model_object, GPUModel):
            y_pred_gpu = model_object.predict(X_current_dim_df.values)
            y_pred = cp.asnumpy(y_pred_gpu) if CUPY_AVAILABLE and device == 'cuda' else y_pred_gpu.cpu().numpy()
        else:
             y_pred = model_object.predict(X_current_dim_df)
        
        y_numeric = y.values.flatten()
        current_target = y_numeric - y_pred

    return models_by_dim


def _find_best_models_sisso_pp(sisso_instance, phi_sis_df, y, D_max, task_type,
                               max_feat_cross_corr, sample_weight, device, torch_device):
    """
    Finds models using the SISSO++ breadth-first, QR-accelerated search.
    This version is fully GPU-accelerated.
    """
    print("\n" + "="*20 + " Starting SISSO++ (Breadth-First QR) Search " + "="*20)
    
    # --- 0. Set up GPU/CPU backend ---
    xp = np
    X_data = phi_sis_df.values
    dtype = np.dtype(sisso_instance.dtype).type
    
    if device == 'cuda' and CUPY_AVAILABLE:
        print("  Using CUDA backend for SISSO++ search.")
        xp = cp
        X_data = xp.asarray(X_data, dtype=dtype)
    elif device == 'mps' and TORCH_AVAILABLE:
        print("  Using MPS backend for SISSO++ search.")
        xp = torch
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
        X_data = torch.from_numpy(X_data).to(torch_device, dtype=torch_dtype)
    else:
        print("  Using CPU (NumPy) backend for SISSO++ search.")

    # --- 1. Prepare data and OLS proxy target ---
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    # The pruning is done on CPU, now we get the indices and use them with the GPU array
    pruned_indices = [phi_sis_df.columns.get_loc(col) for col in phi_pruned.columns]
    feature_names = list(phi_pruned.columns)
    
    X = X_data[:, pruned_indices]
    n_samples, n_features = X.shape

    if task_type in ALL_CLASSIFICATION_TASKS:
        warnings.warn("INFO: For 'sisso++' classification, using OLS on a one-hot encoded target as a proxy for search.")
        lb = LabelBinarizer()
        y_proxy_np = lb.fit_transform(y)
        if y_proxy_np.ndim == 1: y_proxy_np = y_proxy_np.reshape(-1, 1)
    elif task_type == MULTITASK:
        print("INFO: For 'sisso++' multitask, using sum of OLS-RSS across all targets as a proxy for search.")
        y_proxy_np = y.values
    else: # REGRESSION
        y_proxy_np = y.values.reshape(-1, 1)

    # Move proxy target to the same device as X
    y_proxy = xp.asarray(y_proxy_np, dtype=dtype) if xp != torch else torch.from_numpy(y_proxy_np).to(torch_device, dtype=X.dtype)

    fit_intercept = sisso_instance.model_params_.get('fit_intercept', True)
    if fit_intercept:
        y_mean = xp.mean(y_proxy, axis=0)
        y_c = y_proxy - y_mean
        X_mean = xp.mean(X, axis=0)
        X_c = X - X_mean
    else:
        y_c, X_c = y_proxy, X

    # --- 2. Initialize search variables ---
    models_by_dim = {}
    timing_summary = sisso_instance.timing_summary_
    beam_width = n_features
    beam_width_decay = getattr(sisso_instance, 'beam_width_decay', 1.0)
    print(f"INFO: Initial beam width: {beam_width}, Decay factor: {beam_width_decay}")

    # --- 3. Dimension 1 search ---
    t_start_d1 = time.time()
    print("\n--- SISSO++ Search: Dimension 1 ---")
    d1_results = []
    initial_rss_vec = xp.sum(y_c**2, axis=0)
    
    for i in range(n_features):
        x_i = X_c[:, i]
        norm_sq_xi = xp.dot(x_i, x_i)
        if norm_sq_xi < 1e-18: continue
        
        rss_vec = initial_rss_vec - (y_c.T @ x_i)**2 / norm_sq_xi
        total_rss = float(xp.sum(rss_vec))
        proxy_score = np.sqrt(total_rss / (n_samples * y_c.shape[1]))
        
        if np.isfinite(proxy_score):
            d1_results.append({'proxy_score': proxy_score, 'rss': total_rss, 'indices': [i]})

    if not d1_results:
        print("  No valid 1D models found. Stopping."); return {}

    d1_results.sort(key=lambda item: item['proxy_score'])
    best_models_at_level = d1_results[:beam_width]

    # Refit the best 1D model with the actual scoring function
    best_d1_info = best_models_at_level[0]
    final_score, final_model, final_coef = _refit_and_score_final_model(
        best_d1_info['indices'], phi_pruned.values, y, task_type, sisso_instance.model_params_,
        sample_weight, device, torch_device, feature_names
    )
    
    if final_model is not None:
        sym_features = [sisso_instance.feature_space_sym_map_[feature_names[i]] for i in best_d1_info['indices']]
        models_by_dim[1] = {'features': [feature_names[i] for i in best_d1_info['indices']], 'score': final_score,
                            'model': final_model, 'coef': final_coef, 'sym_features': sym_features, 'is_parametric': False}
        print(f"    Best model for D=1 found with score: {final_score:.6g}")
    timing_summary['SISSO++ Search D=1'] = time.time() - t_start_d1
    
    upper_bound_rss = best_d1_info['rss']

    # --- 4. Search for higher dimensions (D > 1) ---
    for D_iter in range(2, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- SISSO++ Search: Dimension {D_iter} ---")
        next_level_candidates = []
        
        for model_info in best_models_at_level:
            if model_info['rss'] > upper_bound_rss:
                continue

            prev_indices = model_info['indices']
            # QR decomposition on the GPU
            if 'q' not in model_info:
                if xp == torch:
                    model_info['q'], _ = torch.linalg.qr(X_c[:, prev_indices])
                else: # CuPy/NumPy
                    model_info['q'], _ = xp.linalg.qr(X_c[:, prev_indices])
            
            prev_q = model_info['q']
            residual_after_proj = y_c - prev_q @ (prev_q.T @ y_c)
            rss_after_proj_vec = xp.sum(residual_after_proj**2, axis=0)
            
            start_idx = prev_indices[-1] + 1
            for new_feat_idx in range(start_idx, n_features):
                new_col = X_c[:, new_feat_idx]
                w = new_col - prev_q @ (prev_q.T @ new_col)
                norm_sq_w = xp.dot(w, w)
                
                if norm_sq_w < 1e-18: continue

                new_rss_vec = rss_after_proj_vec - (residual_after_proj.T @ w)**2 / norm_sq_w
                new_total_rss = float(xp.sum(new_rss_vec))
                new_proxy_score = np.sqrt(new_total_rss / (n_samples * y_c.shape[1]))
                
                new_indices = prev_indices + [new_feat_idx]
                next_level_candidates.append({'proxy_score': new_proxy_score, 'rss': new_total_rss, 'indices': new_indices})

        if not next_level_candidates:
            print("  No new valid models found for this dimension. Stopping search."); break
            
        next_level_candidates.sort(key=lambda item: item['proxy_score'])
        
        upper_bound_rss = min(upper_bound_rss, next_level_candidates[0]['rss'])
        
        if beam_width_decay < 1.0:
            beam_width = max(D_iter + 1, int(beam_width * beam_width_decay))
        
        best_models_at_level = next_level_candidates[:beam_width]

        best_d_iter_info = best_models_at_level[0]
        final_score, final_model, final_coef = _refit_and_score_final_model(
            best_d_iter_info['indices'], phi_pruned.values, y, task_type, sisso_instance.model_params_,
            sample_weight, device, torch_device, feature_names
        )
        
        if final_model is not None:
            sym_features = [sisso_instance.feature_space_sym_map_[feature_names[i]] for i in best_d_iter_info['indices']]
            models_by_dim[D_iter] = {'features': [feature_names[i] for i in best_d_iter_info['indices']], 'score': final_score,
                                    'model': final_model, 'coef': final_coef, 'sym_features': sym_features, 'is_parametric': False}
            print(f"    Best model for D={D_iter} found with score: {final_score:.6g} (Beam size for next level: {len(best_models_at_level)})")
        timing_summary[f'SISSO++ Search D={D_iter}'] = time.time() - t_start_d

    return models_by_dim