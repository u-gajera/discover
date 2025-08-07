
"""
This module implements the different search strategies for finding the best
feature combinations (descriptors). It includes:
- Brute-force search: Exhaustively checks all combinations.
- Greedy search: Iteratively builds up descriptors one feature at a time.
- SISSO++ search: A breadth-first search using efficient QR-based updates.
- OMP search: A robust greedy search using orthogonal matching pursuit.
- MIQP search: An exact L0 solver using mixed-integer quadratic programming.
- A standalone non-linear optimization function for model refinement.

Core SIS/SISSO search engine.

Workflow per dimension D:
    1. **SIS** – keep the top-K single features most correlated with the target.
       Example: from 100 000 raw candidates keep K = 300.
    2. **SO**  – solve a sparse linear regression on those K features to pick
       the best D-term combination.

Iterates D = 1…max_D, saving JSON summaries after each pass.
"""
import numpy as np
import pandas as pd
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
from sklearn.linear_model import LinearRegression

# --- Import Gurobi for MIQP ---
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

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


def _format_feature_str(sisso_instance, feature_name):
    """Helper to get a human-readable formula string for a feature name."""
    sym_expr = sisso_instance.feature_space_sym_map_.get(feature_name)
    if sym_expr and hasattr(sisso_instance, 'sym_clean_to_original_map_') and sisso_instance.sym_clean_to_original_map_:
        # Substitute the clean symbols (f0, f1) with original names (E_tet, etc.)
        subbed_expr = sym_expr.subs(sisso_instance.sym_clean_to_original_map_)
        return str(subbed_expr)
    return feature_name


class ParametricModel:
    """A wrapper for a non-linear model defined by a symbolic expression."""
    def __init__(self, sym_expr, primary_symbols):
        self.sym_expr = sym_expr
        # Store the STRING version of the symbols for lookup.
        self.primary_symbols_str = [str(s) for s in primary_symbols]
        self.is_parametric = True
        # Create the callable function using the string names, which matches DataFrame columns.
        self._predict_func = sympy.lambdify(self.primary_symbols_str, self.sym_expr, 'numpy')

    def predict(self, X):
        """Predicts using the symbolic formula, requires primary features."""
        # This check now correctly compares strings to strings.
        if not all(s in X.columns for s in self.primary_symbols_str):
             raise ValueError(f"Prediction input missing primary features. "
                              f"Required: {self.primary_symbols_str}. "
                              f"Provided: {list(X.columns)}")
        
        # Look up values in the DataFrame using the string names.
        feature_values = [X[s].values for s in self.primary_symbols_str]
        return self._predict_func(*feature_values)


def _prune_by_correlation(phi_df, threshold):
    """Removes features from a DataFrame that are highly correlated with each other, keeping one from each cluster."""
    if threshold >= 1.0:
        return phi_df
    print(f"  Pruning feature space with correlation threshold > {threshold}...")
    corr_matrix = phi_df.corr().abs()
    
    # Greedy keep-one clustering logic
    keep = []
    seen = set()
    for col in phi_df.columns:
        if col in seen:
            continue
        keep.append(col)
        seen.add(col)
        # Find other columns highly correlated with the current one and mark them to be skipped
        correlated_cols = corr_matrix.index[corr_matrix[col] > threshold].tolist()
        seen.update(correlated_cols)
    
    n_dropped = phi_df.shape[1] - len(keep)
    if n_dropped > 0:
        print(f"    Removed {n_dropped} features due to high cross-correlation. Kept {len(keep)}.")
    
    return phi_df[keep]

# Non-Linear Optimization (NLopt) Refinement

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

# Search Strategy Implementations

def _find_best_models_omp(sisso_instance, phi_sis_df, y, D_max, task_type,
                           max_feat_cross_corr, sample_weight, device, torch_device, **kwargs):
    """
    Finds models using Orthogonal Matching Pursuit (OMP), a robust greedy method.
    Note: OMP is only defined for regression tasks.
    """
    print("\n" + "="*20 + " Starting OMP (Orthogonal Matching Pursuit) Search " + "="*20)
    if task_type not in ['regression', 'multitask']:
        raise ValueError("OMP search is currently implemented only for regression and multitask regression.")

    models_by_dim = {}
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    candidate_features_df = phi_pruned.copy()
    
    selected_features = []
    residual = y.copy()
    
    model_params = sisso_instance.model_params_
    timing_summary = sisso_instance.timing_summary_

    for D_iter in range(1, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- OMP Search: Dimension {D_iter} ---")
        
        if candidate_features_df.empty:
            print("  No more features to select. Stopping."); break

        # Find the feature most correlated with the current residual
        correlations = candidate_features_df.corrwith(residual).abs()
        best_new_feature = correlations.idxmax()
        
        selected_features.append(best_new_feature)
        candidate_features_df.drop(columns=[best_new_feature], inplace=True)
        formula_str = _format_feature_str(sisso_instance, best_new_feature)
        print(f"  Selected feature for D={D_iter}: '{formula_str}'")

        # Solve the least squares problem on the current set of selected features
        X_current_dim_df = phi_pruned[selected_features]
        score, model_data = _score_single_model(X_current_dim_df, y, task_type, model_params, sample_weight, device, torch_device)
        timing_summary[f'OMP Search D={D_iter}'] = time.time() - t_start_d

        if model_data is None:
            print(f"  Could not fit a valid model for D={D_iter}. Stopping.")
            selected_features.pop(); break
            
        print(f"    Model score for D={D_iter} (min): {score:.6g}")

        # Store the results for this dimension
        current_sym_features = [sisso_instance.feature_space_sym_map_.get(f, sympy.sympify(f)) for f in selected_features]
        models_by_dim[D_iter] = {
            'features': selected_features.copy(), 'score': score, 'model': model_data.get('model'),
            'coef': model_data.get('coef'), 'sym_features': current_sym_features, 'is_parametric': False
        }

        # Update the residual
        model_object = model_data['model']
        if isinstance(model_object, GPUModel):
            y_pred_gpu = model_object.predict(X_current_dim_df.values)
            y_pred = cp.asnumpy(y_pred_gpu) if CUPY_AVAILABLE and device == 'cuda' else y_pred_gpu.cpu().numpy()
        else:
            y_pred = model_object.predict(X_current_dim_df)
        
        residual = y - y_pred

    return models_by_dim

def _find_best_models_miqp(sisso_instance, phi_sis_df, y, D_max, task_type,
                           max_feat_cross_corr, sample_weight, device, torch_device, **kwargs):
    """
    Finds the provably optimal L0-norm model using MIQP. Requires Gurobi.
    Note: MIQP is only defined for regression tasks with L2 loss.
    """
    print("\n" + "="*20 + " Starting MIQP (Exact L0) Search " + "="*20)
    if not GUROBI_AVAILABLE:
        raise ImportError("MIQP search requires the 'gurobipy' package. Please install it and acquire a license.")
    if task_type != 'regression' or sisso_instance.loss != 'l2':
        raise ValueError("MIQP search is currently implemented only for regression with L2 loss.")

    models_by_dim = {}
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    X_pruned = phi_pruned.values
    y_np = y.values
    
    n_samples, n_features_pruned = X_pruned.shape
    feature_names = list(phi_pruned.columns)
    model_params = sisso_instance.model_params_
    
    fit_intercept = model_params.get('fit_intercept', True)
    X = X_pruned
    if fit_intercept:
        X = np.c_[np.ones(n_samples), X_pruned]

    # Estimate a reasonable "Big M" value for the MIQP constraints
    try:
        ridge_model = LinearRegression(fit_intercept=False).fit(X, y_np)
        M = 10 * np.max(np.abs(ridge_model.coef_))
        if M < 1.0 or not np.isfinite(M): M = 10.0
    except Exception:
        M = 10.0 # Fallback M
    print(f"  Using Big-M value: {M:.2f}")

    # Precompute Q = X.T @ X and c = X.T @ y for the quadratic objective
    Q = X.T @ X
    c_lin = X.T @ y_np

    for D_iter in range(1, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- MIQP Search: Dimension {D_iter} ---")
        
        try:
            m = gp.Model("miqp_sisso")
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', 300)

            beta = m.addMVar(shape=X.shape[1], lb=-GRB.INFINITY, name="beta")
            z = m.addMVar(shape=X.shape[1], vtype=GRB.BINARY, name="z")
            
            m.setObjective(beta @ Q @ beta - 2 * c_lin @ beta, GRB.MINIMIZE)
            
            if fit_intercept:
                m.addConstr(z[0] == 1, "force_intercept")
                m.addConstr(z.sum() == D_iter + 1, "cardinality")
            else:
                m.addConstr(z.sum() == D_iter, "cardinality")

            m.addConstr(beta <= M * z, "bigM_pos")
            m.addConstr(beta >= -M * z, "bigM_neg")

            m.optimize()
            
            if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                print(f"  MIQP solver failed for D={D_iter} (Status: {m.Status}). Stopping.")
                break
            if m.SolCount == 0:
                print(f"  MIQP solver found no feasible solution for D={D_iter}. Stopping.")
                break
            
            beta_sol = beta.X
            selected_indices = np.where(np.abs(beta_sol) > 1e-6)[0]
            
            final_feature_names = [feature_names[i - 1] for i in selected_indices if i > 0] if fit_intercept else [feature_names[i] for i in selected_indices]
            
            if not final_feature_names: continue

            X_combo_df = phi_pruned[final_feature_names]
            score, model_data = _score_single_model(X_combo_df, y, task_type, model_params, sample_weight, device, torch_device)
            sisso_instance.timing_summary_[f'MIQP Search D={D_iter}'] = time.time() - t_start_d

            if model_data is None: continue

            formulas = [_format_feature_str(sisso_instance, f) for f in final_feature_names]
            print(f"    Optimal model for D={len(final_feature_names)} found with score: {score:.6g}")
            print(f"      Features: {', '.join(formulas)}")
            
            sym_features = [sisso_instance.feature_space_sym_map_.get(f, sympy.sympify(f)) for f in final_feature_names]
            models_by_dim[len(final_feature_names)] = {
                'features': final_feature_names, 'score': score, 'model': model_data.get('model'),
                'coef': model_data.get('coef'), 'sym_features': sym_features, 'is_parametric': False
            }

        except gp.GurobiError as e:
            print(f"  Gurobi error on D={D_iter}: {e}. Stopping MIQP search."); break
        except Exception as e:
            print(f"  An unexpected error occurred during MIQP search for D={D_iter}: {e}"); break
            
    return models_by_dim


def _score_ch_combo(new_feature, base_features, phi_df, y, task_type, model_params, sample_weight):
    """Helper function to score a feature combination for the CH geometric search."""
    combo_features = base_features + [new_feature]
    X_combo_df = phi_df[combo_features]
    score, model_data = _score_single_model(X_combo_df, y, task_type, model_params, sample_weight, 'cpu', None)
    if model_data:
        return score, new_feature, model_data
    return float('inf'), new_feature, None


def _find_best_models_ch_greedy(sisso_instance, phi_sis_df, y, D_max, task_type,
                                max_feat_cross_corr, sample_weight, **kwargs):
    """
    Finds models for Convex Hull classification using a specialized greedy search.
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
        
        tasks = (delayed(_score_ch_combo)(new_feat, selected_features, phi_pruned, y, task_type, model_params, sample_weight) for new_feat in remaining_features)
        results = Parallel(n_jobs=sisso_instance.n_jobs, prefer="threads")(tasks)
        
        valid_results = [res for res in results if np.isfinite(res[0])]
        if not valid_results:
            print(f"  No valid models could be formed for D={D_iter}. Stopping."); break
            
        best_score, best_new_feature, best_model_data = min(valid_results, key=lambda x: x[0])
        
        if D_iter > 1 and best_score >= models_by_dim[D_iter-1]['score']:
            print(f"  No improvement found for D={D_iter}. Best D={D_iter-1} score was {models_by_dim[D_iter-1]['score']:.4g}, current is {best_score:.4g}. Stopping.")
            break

        selected_features.append(best_new_feature)
        sisso_instance.timing_summary_[f'Geometric Search D={D_iter}'] = time.time() - t_start_d
        formula_str = _format_feature_str(sisso_instance, best_new_feature)
        print(f"    Best feature for D={D_iter}: '{formula_str}'")
        print(f"    Best model score for D={D_iter} (overlap): {best_score:.6g}")

        sym_features = [sisso_instance.feature_space_sym_map_.get(f, sympy.sympify(f)) for f in selected_features]
        models_by_dim[D_iter] = {
            'features': selected_features.copy(), 'score': best_score,
            'model': best_model_data.get('model'), 'coef': None,
            'sym_features': sym_features, 'is_parametric': False
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
                                   max_feat_cross_corr, sample_weight, device, torch_device, **kwargs):
    """
    Finds the best model for each dimension by testing all combinations.
    """
    print("\n" + "="*20 + " Starting Brute-Force Search " + "="*20)
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
            print(f"  Dimension {D_iter} is larger than the number of available features ({n_features}). Stopping."); break
        try:
            n_combos = math.comb(n_features, D_iter)
        except ValueError:
            print(f"  Cannot compute combinations for D={D_iter} from {n_features} features. Stopping."); break
        
        print(f"  Evaluating {n_combos:,} combinations of {D_iter} features.")
        if n_combos > MAX_COMBINATIONS_WARNING_THRESHOLD:
            warnings.warn(f"Number of combinations ({n_combos:,}) is very large. This may take a long time.")
        if n_combos == 0: continue
            
        combos = combinations(feature_candidates, D_iter)
        tasks = (delayed(_score_brute_force_combo)(combo, phi_pruned, y, task_type, model_params, sample_weight, device, torch_device) for combo in combos)
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
                             max_feat_cross_corr, sample_weight, device, torch_device, **kwargs):
    """
    Finds models using a greedy, iterative approach (Sparsifying Operator).
    """
    print("\n" + "="*20 + " Starting Greedy (Sparsifying Operator) Search " + "="*20)
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

        sorted_candidates_series = run_SIS(features_to_screen_df, current_target, task_type, xp=sisso_instance.xp_)
        if sorted_candidates_series.empty:
            print(f"  SIS found no new correlated features. Stopping at D={D_iter-1}."); break

        print(f"  SIS screening on residual complete. Top candidates:")
        top_features_data = []
        for i in range(min(5, len(sorted_candidates_series))):
            feat_name = sorted_candidates_series.index[i]
            sis_score = sorted_candidates_series.iloc[i]
            formula = _format_feature_str(sisso_instance, feat_name)
            
            rmse_on_residual = np.nan
            try:
                X_feat = features_to_screen_df[[feat_name]].values
                model = LinearRegression().fit(X_feat, current_target)
                y_pred_residual = model.predict(X_feat)
                rmse_on_residual = np.sqrt(mean_squared_error(current_target, y_pred_residual))
            except Exception:
                pass

            top_features_data.append((i + 1, sis_score, rmse_on_residual, formula))
        
        if top_features_data:
            max_formula_len = max(len(f) for _, _, _, f in top_features_data) if top_features_data else 0
            # Header
            rank_w, sis_w, rmse_w = 5, 11, 15
            header = f"    {'Rank':<{rank_w}} | {'SIS Score':<{sis_w}} | {'RMSE (on res.)':<{rmse_w}} | Formula"
            separator = f"    {'-'*rank_w}-+-{'-'*sis_w}-+-{'-'*rmse_w}-+-{'-'*max(7, max_formula_len)}"
            print(header)
            print(separator)
            # Rows
            for rank, sis_score, rmse, formula in top_features_data:
                print(f"    {rank:<{rank_w}} | {sis_score:<{sis_w}.4f} | {rmse:<{rmse_w}.4f} | {formula}")

        best_new_feature = sorted_candidates_series.index[0]
        selected_features.append(best_new_feature)
        formula_str = _format_feature_str(sisso_instance, best_new_feature)
        print(f"  Selected feature for D={D_iter}: '{formula_str}'")

        X_current_dim_df = phi_pruned[selected_features]
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
        current_target = pd.Series(y_numeric - y_pred, index=y.index)
    return models_by_dim


def _find_best_models_sisso_pp(sisso_instance, phi_sis_df, y, D_max, task_type,
                               max_feat_cross_corr, sample_weight, device, 
                               torch_device, **kwargs):
    """
    Finds models using the SISSO++ breadth-first, QR-accelerated search.
    """
    print("\n" + "="*20 + " Starting SISSO++ (Breadth-First QR) Search " + "="*20)
    xp = np
    dtype = np.dtype(sisso_instance.dtype).type
    
    # Prune first, then move data to the selected device
    phi_pruned = _prune_by_correlation(phi_sis_df, max_feat_cross_corr)
    feature_names = list(phi_pruned.columns)
    X_data_cpu = phi_pruned.values.astype(dtype)

    if device == 'cuda' and CUPY_AVAILABLE:
        print("  Using CUDA backend for SISSO++ search."); xp = cp
        X_data = xp.asarray(X_data_cpu)
    elif device == 'mps' and TORCH_AVAILABLE:
        print("  Using MPS backend for SISSO++ search."); xp = torch
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
        X_data = torch.from_numpy(X_data_cpu).to(torch_device, dtype=torch_dtype)
    else:
        print("  Using CPU (NumPy) backend for SISSO++ search.")
        X_data = X_data_cpu

    n_samples, n_features = X_data.shape

    if task_type in ALL_CLASSIFICATION_TASKS:
        warnings.warn("INFO: For 'sisso++' classification, using OLS on a one-hot encoded target as a proxy for search.")
        lb = LabelBinarizer()
        y_proxy_np = lb.fit_transform(y)
        if y_proxy_np.ndim == 1: y_proxy_np = y_proxy_np.reshape(-1, 1)
    elif task_type == MULTITASK:
        print("INFO: For 'sisso++' multitask, using sum of OLS-RSS across all targets as a proxy for search.")
        y_proxy_np = y.values
    else:
        y_proxy_np = y.values.reshape(-1, 1)

    y_proxy_np = y_proxy_np.astype(dtype)
    y_proxy = xp.asarray(y_proxy_np) if xp != torch else torch.from_numpy(y_proxy_np).to(torch_device, dtype=X_data.dtype)

    fit_intercept = sisso_instance.model_params_.get('fit_intercept', True)
    if fit_intercept:
        y_mean, X_mean = xp.mean(y_proxy, axis=0), xp.mean(X_data, axis=0)
        y_c, X_c = y_proxy - y_mean, X_data - X_mean
    else:
        y_c, X_c = y_proxy, X_data

    models_by_dim, timing_summary = {}, sisso_instance.timing_summary_
    beam_width = n_features
    beam_width_decay = getattr(sisso_instance, 'beam_width_decay', 1.0)

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

    best_d1_info = best_models_at_level[0]
    final_score, final_model, final_coef = _refit_and_score_final_model(
        best_d1_info['indices'], X_data_cpu, y, task_type, sisso_instance.model_params_,
        sample_weight, device, torch_device, feature_names)
    
    if final_model is not None:
        sym_features = [sisso_instance.feature_space_sym_map_[feature_names[i]] for i in best_d1_info['indices']]
        models_by_dim[1] = {'features': [feature_names[i] for i in best_d1_info['indices']], 'score': final_score,
                            'model': final_model, 'coef': final_coef, 'sym_features': sym_features, 'is_parametric': False}
        print(f"    Best model for D=1 found with score: {final_score:.6g}")
    timing_summary['SISSO++ Search D=1'] = time.time() - t_start_d1
    upper_bound_rss = best_d1_info['rss']

    for D_iter in range(2, D_max + 1):
        t_start_d = time.time()
        print(f"\n--- SISSO++ Search: Dimension {D_iter} ---")
        next_level_candidates = []
        
        for model_info in best_models_at_level:
            if model_info['rss'] > upper_bound_rss: continue
            prev_indices = model_info['indices']
            if 'q' not in model_info:
                model_info['q'], _ = xp.linalg.qr(X_c[:, prev_indices]) if xp != torch else torch.linalg.qr(X_c[:, prev_indices])
            
            prev_q = model_info['q']
            residual_after_proj = y_c - prev_q @ (prev_q.T @ y_c)
            rss_after_proj_vec = xp.sum(residual_after_proj**2, axis=0)
            
            for new_feat_idx in range(prev_indices[-1] + 1, n_features):
                w = X_c[:, new_feat_idx] - prev_q @ (prev_q.T @ X_c[:, new_feat_idx])
                norm_sq_w = xp.dot(w, w)
                if norm_sq_w < 1e-18: continue

                new_rss_vec = rss_after_proj_vec - (residual_after_proj.T @ w)**2 / norm_sq_w
                new_total_rss = float(xp.sum(new_rss_vec))
                new_proxy_score = np.sqrt(new_total_rss / (n_samples * y_c.shape[1]))
                
                next_level_candidates.append({'proxy_score': new_proxy_score, 'rss': new_total_rss, 'indices': prev_indices + [new_feat_idx]})

        if not next_level_candidates:
            print("  No new valid models found for this dimension. Stopping search."); break
            
        next_level_candidates.sort(key=lambda item: item['proxy_score'])
        upper_bound_rss = min(upper_bound_rss, next_level_candidates[0]['rss'])
        
        if beam_width_decay < 1.0:
            beam_width = max(D_iter + 1, int(beam_width * beam_width_decay))
        best_models_at_level = next_level_candidates[:beam_width]

        best_d_iter_info = best_models_at_level[0]
        final_score, final_model, final_coef = _refit_and_score_final_model(
            best_d_iter_info['indices'], X_data_cpu, y, task_type, sisso_instance.model_params_,
            sample_weight, device, torch_device, feature_names)
        
        if final_model is not None:
            sym_features = [sisso_instance.feature_space_sym_map_[feature_names[i]] for i in best_d_iter_info['indices']]
            models_by_dim[D_iter] = {'features': [feature_names[i] for i in best_d_iter_info['indices']], 'score': final_score,
                                    'model': final_model, 'coef': final_coef, 'sym_features': sym_features, 'is_parametric': False}
            print(f"    Best model for D={D_iter} found with score: {final_score:.6g} (Beam size for next level: {len(best_models_at_level)})")
        timing_summary[f'SISSO++ Search D={D_iter}'] = time.time() - t_start_d

    return models_by_dim