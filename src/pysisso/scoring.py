"""
This module contains all functions related to model evaluation, screening, and
cross-validation. It includes:
- Sure Independence Screening (SIS) with CPU and GPU implementations.
- Scoring functions for various task types (regression, classification, etc.).
- GPU-accelerated kernels for Ridge regression.
- The main cross-validation loop.

Objective functions and ranking helpers.

Provides R², RMSE, MAE, plus tiny wrappers that decide which models are
“plottable” or “best” at each dimension.

Example:
    r2, rmse = score_model(y_true, y_pred)
"""
import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import Ridge, LogisticRegression, HuberRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.feature_selection import f_classif
from scipy.spatial import ConvexHull, qhull
from sklearn.cross_decomposition import CCA

from .constants import (
    REGRESSION, MULTITASK, CH_CLASSIFICATION, CLASSIFICATION_SVM,
    CLASSIFICATION_LOGREG, ALL_CLASSIFICATION_TASKS
)
from .features import CUPY_AVAILABLE, TORCH_AVAILABLE
try:
    import cupy as cp
except ImportError:
    class cp_dummy:
        ndarray = type(None)
        def asnumpy(self, x): return np.asarray(x)
    cp = cp_dummy()

try:
    import torch
except ImportError:
    class torch_dummy:
        Tensor = type(None)
    torch = torch_dummy()

#  Screening and Scoring

def run_SIS(phi, y, task_type, xp=np, multitask_sis_method='average', phi_tensor=None, phi_names=None):
    """
    Performs Sure Independence Screening (SIS) to rank features.
    For multi-task problems, can use simple 'average' correlation or 'cca'.
    For GPU paths, it expects a pre-loaded `phi_tensor` and a list of `phi_names`.
    For CPU paths, it expects a pandas `phi` DataFrame.

    as values, sorted in descending order of importance.
    """
    phi_df = phi
    is_gpu_run = task_type == REGRESSION and xp != np and (xp == cp or xp == torch) and phi_tensor is not None

    if is_gpu_run:
        print(f"Running SIS to screen {phi_tensor.shape[1]} features on GPU...")
        try:
            phi_gpu = phi_tensor

            y_np = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

            if xp == torch:
                # For PyTorch, get the dtype from the tensor and convert numpy array directly
                y_gpu = torch.from_numpy(y_np).to(device=phi_gpu.device, dtype=phi_gpu.dtype)
            else: # cupy
                # For CuPy, we can use the .name attribute as before
                y_gpu = cp.asarray(y_np, dtype=phi_gpu.dtype.name)

            if y_gpu.ndim > 1: y_gpu = y_gpu.squeeze()

            n_samples = phi_gpu.shape[0]
            phi_c = phi_gpu - xp.mean(phi_gpu, axis=0)
            y_c = y_gpu - xp.mean(y_gpu)

            std_args = {'unbiased': True} if xp == torch else {'ddof': 1}
            phi_std = xp.std(phi_c, dim=0 if xp == torch else 0, **std_args)
            y_std = xp.std(y_c, **std_args)

            cov_vec = (phi_c.T @ y_c) / (n_samples - 1)
            correlations_gpu = xp.zeros_like(phi_std)
            valid_std_mask = (phi_std > 1e-9) & (y_std > 1e-9)

            if xp.any(valid_std_mask):
                 correlations_gpu[valid_std_mask] = cov_vec[valid_std_mask] / (phi_std[valid_std_mask] * y_std)

            if CUPY_AVAILABLE and xp == cp:
                correlations_cpu = cp.asnumpy(xp.abs(correlations_gpu))
            elif TORCH_AVAILABLE and xp == torch:
                correlations_cpu = xp.abs(correlations_gpu).cpu().numpy()
            else:
                correlations_cpu = np.abs(correlations_gpu)

            correlations = pd.Series(correlations_cpu, index=phi_names)
            if not correlations.empty:
                print(f"  GPU SIS screening complete. Top feature: '{correlations.idxmax()}'")
            else:
                print("  GPU SIS screening complete. No correlated features found.")
            return correlations.dropna().sort_values(ascending=False)
        except Exception as e:
            warnings.warn(f"GPU-based SIS failed: {e}. Falling back to CPU-based SIS.")
            cpu_values = phi_tensor.cpu().numpy() if TORCH_AVAILABLE and isinstance(phi_tensor, torch.Tensor) else cp.asnumpy(phi_tensor)
            phi_df = pd.DataFrame(cpu_values, columns=phi_names, index=y.index)

    # CPU path (also serves as the fallback path for a failed GPU run)
    if phi_df is None:
        warnings.warn("SIS called in CPU mode but `phi` DataFrame is None. Returning empty Series.")
        return pd.Series([], dtype=float)

    print(f"Running SIS to screen {phi_df.shape[1]} features on CPU...")
    if phi_df.empty:
        return pd.Series([], dtype=float)

    if task_type == REGRESSION:
        y_s = pd.Series(y, index=phi_df.index)
        correlations = phi_df.corrwith(y_s).abs().dropna().sort_values(ascending=False)
    elif task_type in ALL_CLASSIFICATION_TASKS:
         try:
            non_constant_cols = phi_df.columns[phi_df.var() > 1e-9]
            if len(non_constant_cols) == 0: return pd.Series([], dtype=float)
            phi_subset = phi_df[non_constant_cols]
            f_values, _ = f_classif(phi_subset, y)
            correlations = pd.Series(f_values, index=phi_subset.columns).fillna(0).sort_values(ascending=False)
         except ValueError as e:
             warnings.warn(f"SIS f_classif failed: {e}. Returning empty list."); return pd.Series([], dtype=float)

    elif task_type == MULTITASK:
        y_df = y if isinstance(y, pd.DataFrame) else pd.DataFrame(y, index=phi_df.index)

        if multitask_sis_method == 'cca':
            print("  Using Canonical Correlation Analysis (CCA) for multi-task screening.")
            cca = CCA(n_components=1)
            cca_corrs = {}
            for feature_name in phi_df.columns:
                try:
                    X_feat = phi_df[[feature_name]].values
                    X_c, y_c = cca.fit_transform(X_feat, y_df.values)
                    corr = np.corrcoef(X_c.T, y_c.T)[0, 1]
                    cca_corrs[feature_name] = np.abs(corr)
                except np.linalg.LinAlgError:
                    cca_corrs[feature_name] = 0.0
            correlations = pd.Series(cca_corrs).fillna(0).sort_values(ascending=False)
        else: # Default 'average' method
            print("  Using average correlation for multi-task screening.")
            avg_correlations = phi_df.apply(lambda feature: y_df.corrwith(feature).abs().mean())
            correlations = avg_correlations.dropna().sort_values(ascending=False)
    else:
        raise ValueError(f"task_type '{task_type}' not recognized.")

    if not correlations.empty:
        print(f"  CPU SIS screening complete. Top feature: '{correlations.index[0]}'")

    return correlations


def regression_score(X, y, model_params, sample_weight=None):
     """Scores a regression model using Ridge or Huber loss on the CPU."""
     try:
        loss = model_params.get('loss', 'l2')
        fit_intercept = model_params.get('fit_intercept', True)
        if loss == 'l2':
            model = Ridge(alpha=model_params.get('alpha', 1e-6), fit_intercept=fit_intercept)
        elif loss == 'huber':
            model = HuberRegressor(alpha=model_params.get('alpha', 1e-6), epsilon=1.35, fit_intercept=fit_intercept)
        else: raise ValueError(f"Unknown loss function: {loss}")

        model.fit(X, y, sample_weight=sample_weight)
        y_pred = model.predict(X)
        score = np.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        if fit_intercept:
            coef = np.hstack([[model.intercept_], model.coef_]).reshape(1, -1)
        else:
            coef = model.coef_.reshape(1, -1)
        return score, {'model': model, 'coef': coef}
     except (np.linalg.LinAlgError, ValueError): return float('inf'), None

class GPUModel:
    """A minimal model-like object to store coefficients from GPU calculations."""
    def __init__(self, c, i, dev, xp_mod, fit_intercept=True):
        self.coef_ = c
        self.intercept_ = i
        self.device = dev
        self.xp = xp_mod
        self.fit_intercept_ = fit_intercept

    def predict(self, X_new):
        if self.xp == torch and not isinstance(X_new, torch.Tensor):
            dtype = self.coef_.dtype
            X_new = torch.tensor(X_new, dtype=dtype, device=self.coef_.device)
        elif self.xp == cp and not isinstance(X_new, cp.ndarray):
            dtype = self.coef_.dtype
            X_new = cp.asarray(X_new, dtype=dtype)

        pred = X_new @ self.coef_
        if self.fit_intercept_:
            pred += self.intercept_
        return pred

def _cuda_ridge_score(X_gpu, y_gpu, alpha, fit_intercept, sample_weight_gpu=None):
    """Performs Ridge regression directly on a CUDA GPU using CuPy."""
    try:
        y_orig = y_gpu.copy()
        X_orig = X_gpu.copy()
        if sample_weight_gpu is not None:
            sw_sqrt = cp.sqrt(sample_weight_gpu)
            X_gpu = X_gpu * sw_sqrt[:, None]
            y_gpu = y_gpu * sw_sqrt

        if fit_intercept:
            X_mean, y_mean = X_gpu.mean(axis=0), y_gpu.mean()
            X_c, y_c = X_gpu - X_mean, y_gpu - y_mean
            A = X_c.T @ X_c
            b = X_c.T @ y_c
        else:
            A = X_gpu.T @ X_gpu
            b = X_gpu.T @ y_gpu

        A[cp.diag_indices_from(A)] += alpha
        coef = cp.linalg.solve(A, b)

        if fit_intercept:
            intercept = y_mean - X_mean @ coef
        else:
            intercept = 0

        model_obj = GPUModel(coef, intercept, 'cuda', cp, fit_intercept)
        y_pred_unweighted = model_obj.predict(X_orig)
        score = cp.sqrt(cp.mean((y_orig - y_pred_unweighted)**2))

        if fit_intercept:
            full_coef = cp.hstack([[intercept], coef]).reshape(1, -1)
        else:
            full_coef = coef.reshape(1, -1)

        return float(score), {'model': model_obj, 'coef': cp.asnumpy(full_coef)}
    except cp.linalg.LinAlgError: return float('inf'), None

def _mps_ridge_score(X_mps, y_mps, alpha, fit_intercept, sample_weight_mps=None):
    """Performs Ridge regression directly on an Apple Silicon GPU using PyTorch/MPS."""
    try:
        y_orig = y_mps.clone()
        X_orig = X_mps.clone()
        if sample_weight_mps is not None:
            sw_sqrt = torch.sqrt(sample_weight_mps)
            X_mps = X_mps * sw_sqrt.unsqueeze(1)
            y_mps = y_mps * sw_sqrt

        if fit_intercept:
            X_mean, y_mean = X_mps.mean(dim=0), y_mps.mean()
            X_c, y_c = X_mps - X_mean, y_mps - y_mean
            A = X_c.T @ X_c
            b = X_c.T @ y_c
        else:
            A = X_mps.T @ X_mps
            b = X_mps.T @ y_mps

        A.diagonal().add_(alpha)
        coef = torch.linalg.solve(A, b)

        if fit_intercept:
            intercept = y_mean - (X_mean @ coef)
        else:
            intercept = torch.tensor(0.0, device=X_mps.device, dtype=X_mps.dtype)

        model_obj = GPUModel(coef, intercept, 'mps', torch, fit_intercept)
        y_pred_unweighted = model_obj.predict(X_orig)
        score = torch.sqrt(torch.mean((y_orig - y_pred_unweighted)**2))

        if fit_intercept:
            full_coef = torch.cat([intercept.unsqueeze(0), coef]).reshape(1, -1)
        else:
            full_coef = coef.reshape(1, -1)

        return score.item(), {'model': model_obj, 'coef': full_coef.cpu().numpy()}
    except torch.linalg.LinAlgError: return float('inf'), None


def multitask_score(X, Y_df, model_params, sample_weight=None):
    """Scores a multi-target regression model by averaging RSS over all tasks."""
    total_rss, coefs_list, intercepts_list, models = 0, [], [], {}
    n_tasks = len(Y_df.columns)
    if n_tasks == 0: return float('inf'), None
    fit_intercept = model_params.get('fit_intercept', True)

    for task_col in Y_df.columns:
        y_task = Y_df[task_col]
        sw_task = sample_weight[task_col] if sample_weight is not None and isinstance(sample_weight, pd.DataFrame) else sample_weight
        try:
            model = Ridge(alpha=model_params.get('alpha', 1e-6), fit_intercept=fit_intercept).fit(X, y_task, sample_weight=sw_task)
            y_pred = model.predict(X)
            rss = np.sum(((y_task - y_pred)**2) * (sw_task if sw_task is not None else 1))
            total_rss += rss / (np.sum(sw_task) if sw_task is not None else len(y_task))
            coefs_list.append(model.coef_)
            intercepts_list.append(model.intercept_)
            models[task_col] = model
        except np.linalg.LinAlgError: return float('inf'), None

    if fit_intercept:
        coef = np.hstack([np.array(intercepts_list).reshape(-1, 1), np.vstack(coefs_list)])
    else:
        coef = np.vstack(coefs_list)
    return total_rss / n_tasks , {'model': models, 'coef': coef}

def ch_overlap_score(X, y, model_params, sample_weight=None, n_mc_points=1500):
    """Scores a classification by the geometric overlap of class convex hulls."""
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_np = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
    classes = np.unique(y_np)
    if len(classes) < 2: return float('inf'), None
    min_points_for_hull = X_np.shape[1] + 1
    hulls, points_dict = {}, {}
    try:
        for c in classes:
            points = X_np[y_np == c]
            points_dict[c] = points
            if len(points) < min_points_for_hull or np.all(np.ptp(points, axis=0) < 1e-9): return float('inf'), None
            try: hulls[c] = ConvexHull(points)
            except qhull.QhullError: return float('inf'), None
    except (ValueError): return float('inf'), None

    min_bounds, max_bounds = X_np.min(axis=0), X_np.max(axis=0)
    padding = (max_bounds - min_bounds) * 0.05
    mc_points = np.random.uniform(min_bounds-padding, max_bounds+padding, size=(n_mc_points, X_np.shape[1]))
    in_hull_counts = np.zeros(n_mc_points, dtype=int)
    total_points_in_any_hull = 0
    for hull in hulls.values():
        try:
           is_inside = np.all(np.add(np.dot(mc_points, hull.equations[:, :-1].T), hull.equations[:, -1]) <= 1e-9, axis=1)
           in_hull_counts += is_inside
           total_points_in_any_hull += np.sum(is_inside)
        except ValueError: return float('inf'), None

    points_in_overlap = np.sum(in_hull_counts > 1)
    overlap_fraction = points_in_overlap / total_points_in_any_hull if total_points_in_any_hull > 0 else 1.0

    for c in classes:
         if len(points_dict[c]) < 2 * min_points_for_hull: overlap_fraction += 0.1

    return overlap_fraction, {'model': hulls, 'coef': None}

def classification_score(X, y, task_type, model_params, sample_weight=None):
      """Scores a classification model (SVM or Logistic Regression) using log-loss."""
      try:
          fit_intercept = model_params.get('fit_intercept', True)
          if task_type == CLASSIFICATION_SVM:
               model = SVC(C=model_params.get('C_svm', 1.0), probability=True, gamma='scale', kernel='rbf')
          elif task_type == CLASSIFICATION_LOGREG:
               model = LogisticRegression(C=model_params.get('C_logreg', 1.0), solver='liblinear', multi_class='ovr', fit_intercept=fit_intercept)
          else: return float('inf'), None
          if len(np.unique(y)) < 2: return float('inf'), None

          model.fit(X, y, sample_weight=sample_weight)
          proba = model.predict_proba(X)
          score = log_loss(y, proba, sample_weight=sample_weight, labels=model.classes_)
          return score, {'model': model, 'coef': None}
      except (ValueError, np.linalg.LinAlgError): return float('inf'), None

SCORE_FUNCTIONS = {
    REGRESSION: regression_score,
    MULTITASK: multitask_score,
    CH_CLASSIFICATION: ch_overlap_score,
    CLASSIFICATION_SVM: lambda X, y, p, sw: classification_score(X, y, CLASSIFICATION_SVM, p, sw),
    CLASSIFICATION_LOGREG: lambda X, y, p, sw: classification_score(X, y, CLASSIFICATION_LOGREG, p, sw),
}


#  Main L0 Search and Cross-Validation


def _score_single_model(X_combo_df, y, task_type, model_params, sample_weight, device, torch_device):
    """
    Scores a single combination of features, dispatching to the correct
    CPU or GPU implementation based on the task and device settings.
    """
    X_combo_values = X_combo_df.values
    is_reg_task = task_type == REGRESSION and model_params.get('loss') != 'huber'
    fit_intercept = model_params.get('fit_intercept', True)
    alpha = model_params.get('alpha', 1e-5)
    dtype = np.dtype(model_params.get('dtype', 'float64')).type

    if device == 'cuda' and is_reg_task and CUPY_AVAILABLE:
        X_gpu = cp.asarray(X_combo_values, dtype=dtype)
        y_gpu = cp.asarray(y, dtype=dtype)
        sw_gpu = cp.asarray(sample_weight, dtype=dtype) if sample_weight is not None else None
        score, model_data = _cuda_ridge_score(X_gpu, y_gpu, alpha, fit_intercept, sw_gpu)
        if cp: cp.get_default_memory_pool().free_all_blocks()
    elif device == 'mps' and is_reg_task and TORCH_AVAILABLE:
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
        X_mps = torch.from_numpy(X_combo_values).to(torch_device, dtype=torch_dtype)
        y_mps_np = y.to_numpy() if isinstance(y, pd.Series) else y
        y_mps = torch.from_numpy(y_mps_np).to(torch_device, dtype=torch_dtype)
        sw_mps = torch.from_numpy(sample_weight).to(torch_device, dtype=torch_dtype) if sample_weight is not None else None
        score, model_data = _mps_ridge_score(X_mps, y_mps, alpha, fit_intercept, sw_mps)
    else:
        score_func = SCORE_FUNCTIONS[task_type]
        score, model_data = score_func(X_combo_df, y, model_params, sample_weight)

    if model_data is not None and np.isfinite(score):
        return score, model_data
    else:
        return float('inf'), None

def _run_cv(X_features, y, cv_splitter, task_type, model_params, sample_weight):
    """Runs a full cross-validation loop for a given set of features."""
    scores = []
    score_func = SCORE_FUNCTIONS[task_type]
    sw_series = pd.Series(sample_weight, index=y.index) if sample_weight is not None and not isinstance(sample_weight, pd.DataFrame) else sample_weight

    for train_idx, val_idx in cv_splitter.split(X_features, y):
         X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
         y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
         y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

         if sw_series is not None:
             sw_train = sw_series.iloc[train_idx].values if isinstance(sw_series, pd.Series) else sw_series.iloc[train_idx]
             sw_val = sw_series.iloc[val_idx].values if isinstance(sw_series, pd.Series) else sw_series.iloc[val_idx]
         else:
             sw_train, sw_val = None, None

         if task_type in ALL_CLASSIFICATION_TASKS:
              if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                  scores.append(float('inf')); continue
              if task_type == CH_CLASSIFICATION and len(y_train) <= X_train.shape[1]:
                  scores.append(float('inf')); continue
         try:
            _, model_data = score_func(X_train, y_train, model_params, sw_train)
            if model_data is None:
                scores.append(float('inf')); continue

            val_score = float('inf')
            if task_type == REGRESSION:
                 y_pred = model_data['model'].predict(X_val)
                 val_score = np.sqrt(mean_squared_error(y_val, y_pred, sample_weight=sw_val))
            elif task_type == MULTITASK:
                  total_rss_val = 0
                  for task_col, model in model_data['model'].items():
                        y_pred = model.predict(X_val)
                        sw_val_task = sw_val[task_col] if sw_val is not None and isinstance(sw_val, pd.DataFrame) else sw_val
                        rss = np.sum(((y_val[task_col] - y_pred)**2) * (sw_val_task if sw_val_task is not None else 1))
                        total_rss_val += rss / (np.sum(sw_val_task) if sw_val_task is not None else len(y_val))
                  val_score = total_rss_val / len(model_data['model'])
            elif task_type == CH_CLASSIFICATION:
                 val_score, _ = ch_overlap_score(X_val, y_val, model_params, sw_val)
            elif task_type in [CLASSIFICATION_SVM, CLASSIFICATION_LOGREG]:
                 try:
                    proba_val = model_data['model'].predict_proba(X_val)
                    val_score = log_loss(y_val, proba_val, labels=model_data['model'].classes_, sample_weight=sw_val)
                 except ValueError: val_score = float('inf')
            scores.append(val_score)
         except Exception: scores.append(float('inf'))

    valid_scores = [s for s in scores if np.isfinite(s)]
    return (np.mean(valid_scores), np.std(valid_scores)) if valid_scores else (float('inf'), float('inf'))

def _refit_and_score_final_model(indices, X_data, y_true, task_type, model_params, sample_weight, device, torch_device, feature_names):
    """
    Helper function for SISSO++ to refit the best feature combination with the
    actual model and scoring function for the task.
    """
    combo_names = [feature_names[i] for i in indices]
    X_combo_df = pd.DataFrame(X_data[:, indices], columns=combo_names)

    score, model_data = _score_single_model(
        X_combo_df, y_true, task_type, model_params, sample_weight, device, torch_device
    )

    if model_data:
        return score, model_data.get('model'), model_data.get('coef')
    return float('inf'), None, None