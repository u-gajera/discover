"""
Assorted helper functions (logging, timing, safe NumPy wrappers, etc.).

Example:
    with timed_block("descriptor evaluation"):
        X = safe_eval(exprs, dataframe)
"""
import numpy as np
import pandas as pd
import sympy
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .constants import (
    REGRESSION, MULTITASK, CH_CLASSIFICATION, CLASSIFICATION_SVM,
    CLASSIFICATION_LOGREG, ALL_CLASSIFICATION_TASKS
)

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False



#  Plotting
def _get_score_label(task_type, loss, selection_method='cv'):
    """Returns a human-readable string for the score metric."""
    if selection_method == 'aic':
        return 'AIC Score'
    if selection_method == 'bic':
        return 'BIC Score'
    
    score_name = {
        REGRESSION: 'RMSE', MULTITASK: 'Avg RSS',
        CH_CLASSIFICATION: 'Overlap Fraction',
        CLASSIFICATION_SVM: 'Log Loss', CLASSIFICATION_LOGREG: 'Log Loss'
    }.get(task_type, 'Score')
    if loss == 'huber' and task_type == REGRESSION:
        score_name = 'Huber Loss'
    
    method_label = "CV" if selection_method == 'cv' else "OOB"
    return f'{score_name} ({method_label})'

def plot_cv_results(cv_results, best_D, task_type, loss, selection_method='cv', ax=None, save_path=None):
    """Plots the cross-validation, bootstrap OOB, or information criterion score."""
    if not cv_results: return None
    if ax is None: fig, ax = plt.subplots(figsize=(7, 5)); own_fig = True
    else: fig = ax.figure; own_fig = False
    
    dims = sorted(cv_results.keys())
    means = [cv_results[d][0] for d in dims]
    stds  = [cv_results[d][1] for d in dims]
    
    # Do not plot error bars for AIC/BIC since std is 0
    if selection_method in ['aic', 'bic']:
        ax.plot(dims, means, '-o', label=f'{selection_method.upper()} Score')
    else:
        ax.errorbar(dims, means, yerr=stds, fmt='-o', capsize=4, label='Mean Score')
        
    if best_D in dims:
        ax.plot(best_D, cv_results[best_D][0], 'r*', markersize=12, label=f'Best D={best_D}')
    
    title_map = {
        'bootstrap': "Bootstrap OOB Score vs. Dimension",
        'cv': "Cross-Validation Score vs. Dimension",
        'aic': "AIC Score vs. Dimension",
        'bic': "BIC Score vs. Dimension"
    }
    ax.set_xlabel("Descriptor Dimension (D)")
    ax.set_ylabel(_get_score_label(task_type, loss, selection_method))
    ax.set_title(title_map.get(selection_method, "Score vs. Dimension"))
    ax.set_xticks(dims)
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
    
    if own_fig: plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

def plot_parity(sisso_instance, X, y, sample_weight=None, ax=None, save_path=None):
    """Generates a parity plot (predicted vs. actual) for regression tasks."""
    from sklearn.metrics import r2_score, mean_squared_error
    
    if sisso_instance.task_type_ not in [REGRESSION, MULTITASK] or sisso_instance.best_model_ is None:
        return None
    if ax is None: fig, ax = plt.subplots(figsize=(6, 6)); own_fig = True
    else: fig = ax.figure; own_fig = False

    y_pred = sisso_instance.predict(X)
    
    if sisso_instance.task_type_ == MULTITASK:
        y_true, y_pred_vals = y.values.flatten(), y_pred.values.flatten()
        if sample_weight is not None and isinstance(sample_weight, pd.DataFrame):
            sw = sample_weight.values.flatten()
        else: sw = sample_weight
        title_suffix = " (Multi-task)"
    else:
        y_true, y_pred_vals, sw = y, y_pred, sample_weight
        title_suffix = ""

    r2 = r2_score(y_true, y_pred_vals, sample_weight=sw)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_vals, sample_weight=sw))
    
    ax.scatter(y_true, y_pred_vals, alpha=0.5, label=f'R²={r2:.3f}, RMSE={rmse:.3f}')
    min_val, max_val = min(np.min(y_true), np.min(y_pred_vals)), max(np.max(y_true), np.max(y_pred_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="y=x")
    ax.set_xlabel("Actual Value"); ax.set_ylabel("Predicted Value")
    ax.set_title(f"Parity Plot D={sisso_instance.best_D_}{title_suffix}")
    ax.set_aspect('equal', adjustable='box'); ax.legend(); ax.grid(True)
    
    if own_fig: plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

def plot_classification_2d(sisso_instance, X, y, ax=None, save_path=None, resolution=150):
    """Plots the 2D decision boundary for 2-dimensional classification models."""
    is_parametric = hasattr(sisso_instance.best_model_, 'is_parametric') and sisso_instance.best_model_.is_parametric
    if sisso_instance.task_type_ not in ALL_CLASSIFICATION_TASKS or sisso_instance.best_D_ != 2 or is_parametric:
        if is_parametric: print("Plotting: 2D classification plot not supported for parametric models.")
        elif sisso_instance.best_model_ is not None and sisso_instance.task_type_ in ALL_CLASSIFICATION_TASKS and sisso_instance.best_D_ != 2:
            print(f"  Plotting: 2D classification plot only for D=2 (best D={sisso_instance.best_D_})")
        return None
    if ax is None: fig, ax = plt.subplots(figsize=(8, 7)); own_fig = True
    else: fig = ax.figure; own_fig = False

    X_desc = sisso_instance._transform_X(X)
    x_min, x_max = X_desc.iloc[:, 0].min() - 0.5, X_desc.iloc[:, 0].max() + 0.5
    y_min, y_max = X_desc.iloc[:, 1].min() - 0.5, X_desc.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    colors = plt.cm.get_cmap('viridis', len(sisso_instance.classes_))
    cmap_light = mcolors.ListedColormap(colors(np.linspace(0, 1, len(sisso_instance.classes_))))

    if sisso_instance.task_type_ == CH_CLASSIFICATION:
        for i, c in enumerate(sisso_instance.classes_):
            if c in sisso_instance.best_model_:
                hull = sisso_instance.best_model_[c]
                for simplex in hull.simplices: ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], color=colors(i), lw=2)
            ax.scatter(X_desc[y == c].iloc[:, 0], X_desc[y == c].iloc[:, 1], color=colors(i), label=f'Class {c}', alpha=0.7, edgecolors='k', s=40)
    else:
        mesh_input_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X_desc.columns)
        Z_pred = sisso_instance.best_model_.predict(mesh_input_df)
        class_map = {cls: i for i, cls in enumerate(sisso_instance.classes_)}
        Z = np.array([class_map.get(p, -1) for p in Z_pred]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
        for i, c in enumerate(sisso_instance.classes_):
            ax.scatter(X_desc[y == c].iloc[:, 0], X_desc[y == c].iloc[:, 1], color=colors(i), edgecolor='k', s=40, label=f'Class {c}')
    
    desc_f1 = print_descriptor_formula([sisso_instance.best_descriptor_[0]], None, sisso_instance.task_type_, True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip()
    desc_f2 = print_descriptor_formula([sisso_instance.best_descriptor_[1]], None, sisso_instance.task_type_, True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip()

    ax.set_xlabel(f'D1: {desc_f1}'); ax.set_ylabel(f'D2: {desc_f2}')
    ax.set_title('Decision Map (D=2)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)

    if own_fig: plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

def plot_classification_3d(sisso_instance, X, y, save_path=None):
    """Generates a 3D scatter plot of the descriptor space for 3D classification models."""
    is_parametric = hasattr(sisso_instance.best_model_, 'is_parametric') and sisso_instance.best_model_.is_parametric
    if sisso_instance.task_type_ not in ALL_CLASSIFICATION_TASKS or sisso_instance.best_D_ != 3 or is_parametric:
        if is_parametric: print("Plotting: 3D classification plot not supported for parametric models.")
        elif sisso_instance.best_model_ is not None and sisso_instance.task_type_ in ALL_CLASSIFICATION_TASKS and sisso_instance.best_D_ != 3:
            print(f"  Plotting: 3D classification plot only for D=3 (best D={sisso_instance.best_D_})")
        return None
    
    fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection='3d')
    X_desc = sisso_instance._transform_X(X)
    colors = plt.cm.get_cmap('viridis', len(sisso_instance.classes_))
    
    for i, c in enumerate(sisso_instance.classes_):
        class_points = X_desc[y == c]
        ax.scatter(class_points.iloc[:, 0], class_points.iloc[:, 1], class_points.iloc[:, 2],
                   color=colors(i), label=f'Class {c}', s=30, alpha=0.7, edgecolors='k')
    
    desc_f1 = print_descriptor_formula([sisso_instance.best_descriptor_[0]], None, sisso_instance.task_type_, True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip()
    desc_f2 = print_descriptor_formula([sisso_instance.best_descriptor_[1]], None, sisso_instance.task_type_, True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip()
    desc_f3 = print_descriptor_formula([sisso_instance.best_descriptor_[2]], None, sisso_instance.task_type_, True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip()
    
    ax.set_xlabel(f"D1: {desc_f1[:20]}..", fontsize=9)
    ax.set_ylabel(f"D2: {desc_f2[:20]}..", fontsize=9)
    ax.set_zlabel(f"D3: {desc_f3[:20]}..", fontsize=9)
    ax.set_title("Descriptor Space (D=3)"); ax.legend()
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

def plot_descriptor_pairplot(sisso_instance, X, y, save_path=None):
    """Generates a pairplot of the final descriptors using seaborn."""
    is_parametric = hasattr(sisso_instance.best_model_, 'is_parametric') and sisso_instance.best_model_.is_parametric
    if sisso_instance.best_model_ is None or sisso_instance.best_D_ < 2 or not SEABORN_AVAILABLE or is_parametric:
        if is_parametric: print("Plotting: Descriptor pairplot not applicable for parametric models.")
        return None
        
    X_desc = sisso_instance._transform_X(X)
    df_plot = X_desc.copy()
    
    # Use original feature expressions for column names for clarity
    model_info = sisso_instance.models_by_dim_[sisso_instance.best_D_]
    pretty_names = [print_descriptor_formula([f], None, 'regression', True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip() for f in model_info['sym_features']]
    df_plot.columns = pretty_names
    
    df_plot['target'] = y.values
    hue_arg = 'target' if sisso_instance.task_type_ in ALL_CLASSIFICATION_TASKS else None
    
    print("  - Generating descriptor pairplot...")
    pair_grid = sns.pairplot(df_plot, hue=hue_arg, corner=True, diag_kind='kde')
    pair_grid.fig.suptitle(f"Descriptor Pairplot (D={sisso_instance.best_D_})", y=1.02)
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path); plt.close(pair_grid.fig)
    return pair_grid.fig

def plot_descriptor_histograms(sisso_instance, X, save_path=None):
    """Plots histograms of the values for each final descriptor."""
    is_parametric = hasattr(sisso_instance.best_model_, 'is_parametric') and sisso_instance.best_model_.is_parametric
    if sisso_instance.best_model_ is None or is_parametric:
        if is_parametric: print("Plotting: Descriptor histograms not applicable for parametric models.")
        return None
        
    X_desc = sisso_instance._transform_X(X)
    n_desc = X_desc.shape[1]
    fig, axes = plt.subplots(1, n_desc, figsize=(4 * n_desc, 4), sharey=True)
    if n_desc == 1: axes = [axes]
    
    fig.suptitle(f"Descriptor Value Distributions (D={sisso_instance.best_D_})")
    for i, ax in enumerate(axes):
        ax.hist(X_desc.iloc[:, i], bins=20, edgecolor='k', alpha=0.7)
        desc_formula = print_descriptor_formula([sisso_instance.best_descriptor_[i]], None, 'regression', True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip()
        ax.set_xlabel(f"D{i+1}: {desc_formula}")
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Frequency")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

# --- : Advanced Interpretation Plots ---

def plot_feature_importance(sisso_instance, X, ax=None, save_path=None):
    """: Plots the scaled importance of each descriptor in the final linear model."""
    model_info = sisso_instance.models_by_dim_.get(sisso_instance.best_D_)
    if not model_info or 'coef' not in model_info or model_info['coef'] is None:
        print("Plotting: Feature importance plot requires a final linear model with coefficients.")
        return None
    if sisso_instance.task_type_ == MULTITASK:
        print("Plotting: Feature importance plot is not yet supported for multi-task models.")
        return None
    if ax is None: fig, ax = plt.subplots(figsize=(8, 0.8 * sisso_instance.best_D_ + 3)); own_fig = True
    else: fig = ax.figure; own_fig = False

    X_desc = sisso_instance._transform_X(X)
    coefs = model_info['coef'].flatten()
    if not sisso_instance.fix_intercept_:
        coefs = coefs[1:] # remove intercept

    # Importance = |coefficient| * std(descriptor value)
    importances = np.abs(coefs) * X_desc.std().values
    
    feat_names = [print_descriptor_formula([f], None, 'regression', True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip() for f in model_info['sym_features']]
    sorted_idx = np.argsort(importances)

    ax.barh(np.arange(len(feat_names)), importances[sorted_idx], align='center')
    ax.set_yticks(np.arange(len(feat_names)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel('Scaled Feature Importance (|coef| * std(feature))')
    ax.set_title(f'Descriptor Importance (D={sisso_instance.best_D_})')

    if own_fig: plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

def plot_partial_dependence(sisso_instance, X, features_to_plot=None, ax=None, save_path=None, resolution=50):
    """: Creates a Partial Dependence Plot (PDP) to visualize model response."""
    if sisso_instance.best_model_ is None: return None
    is_parametric = hasattr(sisso_instance.best_model_, 'is_parametric') and sisso_instance.best_model_.is_parametric
    if is_parametric:
        print("Plotting: Partial dependence plots are not yet supported for parametric models.")
        return None
    
    X_desc = sisso_instance._transform_X(X)
    
    if features_to_plot is None:
        # Default to first one or two descriptors based on what is available
        features_to_plot = sisso_instance.models_by_dim_[sisso_instance.best_D_]['features'][:2]
    if isinstance(features_to_plot, str):
        features_to_plot = [features_to_plot]
    
    n_plot_feats = len(features_to_plot)
    if n_plot_feats > 2 or n_plot_feats == 0:
        raise ValueError("Partial dependence plot only supports 1 or 2 features.")
    if not all(f in X_desc.columns for f in features_to_plot):
        raise ValueError("One or more features to plot are not in the final descriptor.")

    if ax is None:
        if n_plot_feats == 2: fig, ax = plt.subplots(figsize=(8, 7))
        else: fig, ax = plt.subplots(figsize=(7, 5))
        own_fig = True
    else: fig = ax.figure; own_fig = False

    # Create a grid and a background dataset with mean values
    grid_data = pd.DataFrame(np.array([np.linspace(c.min(), c.max(), num=resolution) for c in X_desc.T]).T, columns=X_desc.columns)
    background_data = X_desc.mean().to_frame().T
    
    # Get pretty names for labels
    all_pretty_names_map = {feat_name: print_descriptor_formula([sym_feat], None, 'regression', True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip() 
                           for feat_name, sym_feat in zip(sisso_instance.models_by_dim_[sisso_instance.best_D_]['features'], sisso_instance.best_descriptor_)}

    if n_plot_feats == 1:
        feat_name = features_to_plot[0]
        plot_df = pd.concat([background_data] * resolution, ignore_index=True)
        plot_df[feat_name] = grid_data[feat_name]
        
        predictions = sisso_instance.best_model_.predict(plot_df[X_desc.columns]) # ensure order
        ax.plot(plot_df[feat_name], predictions, '-')
        ax.set_xlabel(f"Descriptor: {all_pretty_names_map[feat_name]}")
        ax.set_ylabel("Partial Dependence (Prediction)")
        ax.set_title(f"1D Partial Dependence Plot")

    elif n_plot_feats == 2:
        f1, f2 = features_to_plot[0], features_to_plot[1]
        xx, yy = np.meshgrid(grid_data[f1], grid_data[f2])
        plot_df = pd.concat([background_data] * (resolution**2), ignore_index=True)
        plot_df[f1] = xx.ravel()
        plot_df[f2] = yy.ravel()
        
        predictions = sisso_instance.best_model_.predict(plot_df[X_desc.columns]) # ensure order
        Z = predictions.reshape(xx.shape)
        
        contour = ax.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)
        fig.colorbar(contour, ax=ax, label="Partial Dependence (Prediction)")
        ax.set_xlabel(f"Descriptor: {all_pretty_names_map[f1]}")
        ax.set_ylabel(f"Descriptor: {all_pretty_names_map[f2]}")
        ax.set_title(f"2D Partial Dependence Plot")

    ax.grid(True, linestyle='--', alpha=0.6)
    if own_fig: plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close(fig)
    return fig

# Output / Reporting

def save_results(sisso_instance, X, y, sample_weight):
    """Saves all model results, data, and plots to the specified work directory."""
    workdir = sisso_instance.workdir
    print(f"\nSaving results to '{workdir}'...")
    p_workdir = Path(workdir)
    p_models = p_workdir / "models"
    p_desc_dat = p_workdir / "desc_dat"
    p_plots = p_workdir / "plots"
    for p in [p_models, p_desc_dat, p_plots]:
        p.mkdir(exist_ok=True, parents=True)

    if sisso_instance.save_feature_space:
        p_fs = p_workdir / "feature_space"
        p_fs.mkdir(exist_ok=True)
        out_path = p_fs / "feature_space.csv.gz"
        sisso_instance.feature_space_df_.to_csv(out_path, index=False, compression='gzip')
        print(f"  - Saved full feature space to {out_path}")

    print("  - Generating and saving plots...")
    try:
        plot_cv_results(sisso_instance.cv_results_, sisso_instance.best_D_, sisso_instance.task_type_, sisso_instance.loss, sisso_instance.selection_method, save_path=p_plots / "selection_scores.png")
        # ... (rest of plotting calls are fine) ...
    except Exception as e:
        warnings.warn(f"Error during plotting: {e}")

    for D, model_data in sisso_instance.models_by_dim_.items():
        with (p_models / f"model_D{D:02d}.dat").open("w") as f:
            score_label = _get_score_label(sisso_instance.task_type_, sisso_instance.loss, 'cv').replace(' (CV)', '')

            validation_label_map = {
                'cv': 'CV_SCORE',
                'bootstrap': 'BOOTSTRAP_OOB_SCORE',
                'aic': 'AIC_SCORE',
                'bic': 'BIC_SCORE'
            }
            validation_label = validation_label_map.get(sisso_instance.selection_method, "VALIDATION_SCORE")

            (val_score, val_std) = sisso_instance.cv_results_.get(D, (model_data['score'], 0.0))
            f.write(f"# DIMENSION: {D}\n")
            f.write(f"# SCORE_ON_FULL_DATA ({score_label}): {model_data['score']:.6g}\n")
            f.write(f"# {validation_label} ({score_label}): {val_score:.6g} +/- {val_std:.6g}\n")
            f.write(f"# IS_BEST: {'YES' if D == sisso_instance.best_D_ else 'NO'}\n")
            f.write(f"# IS_PARAMETRIC: {'YES' if model_data.get('is_parametric') else 'NO'}\n\n")

            if 'vif' in model_data:
                f.write("# VARIANCE INFLATION FACTORS (VIF)\n")
                for feat, vif_val in model_data['vif'].items():
                    f.write(f"#  - {feat[:35]:<37}: {vif_val:.4f}\n")
                f.write("\n")
            
            if model_data.get('coef') is not None:
                header = "Coefs..." if sisso_instance.fix_intercept_ else "Intercept, Coefs..."
                if 'coef_stderr' in model_data:
                    header += " (standard errors on next line)"
                    np.savetxt(f, model_data['coef'], fmt="%.8f", header=header, comments="# ")
                    np.savetxt(f, model_data['coef_stderr'].reshape(1,-1), fmt="%.8f", comments="# ")
                else:
                    np.savetxt(f, model_data['coef'], fmt="%.8f", header=header, comments="# ")
            
            if 'orthogonal_model_formula' in model_data:
                 f.write(f"\n# ORTHOGONALIZED MODEL: y = {model_data['orthogonal_model_formula']}\n")

            s_expr = print_descriptor_formula(
                model_data['sym_features'], model_data.get('coef'), sisso_instance.task_type_,
                sisso_instance.fix_intercept_, s_expr_format=True,
                model_provider=model_data.get('model'),
                clean_to_original_map=sisso_instance.sym_clean_to_original_map_
            )
            pretty_features = [print_descriptor_formula([f], None, sisso_instance.task_type_, True, clean_to_original_map=sisso_instance.sym_clean_to_original_map_).split("=")[-1].strip() for f in model_data['sym_features']]
            f.write("\n# DESCRIPTOR FEATURES (human readable):\n# " + "\n# ".join(f"D{i+1}: {feat}" for i, feat in enumerate(pretty_features)))
            f.write(f"\n\n# S-EXPRESSION(S)\n{s_expr}\n")
        
        if not model_data.get('is_parametric'):
            X_desc_d = sisso_instance.feature_space_df_[model_data['features']]
            X_desc_d.to_csv(
                p_desc_dat / f"D{D:02d}.dat", sep=' ', index=False, header=True, float_format="%.6f"
            )

    with (p_workdir / "SISSO.out").open("w") as f:
        f.write(sisso_instance.summary_report(X,y,sample_weight))

    print("  - Results saved.")


def sympy_to_s_expression(expr):
    """Converts a SymPy expression to a Lisp-style S-expression string."""
    if isinstance(expr, (sympy.Symbol, sympy.Number)):
        return str(expr)
    if expr is None: return "None"
    
    func_name = expr.func.__name__
    op_map = {'Add': '+', 'Mul': '*', 'Pow': '**', 'Abs': 'abs'}
    s_op = op_map.get(func_name, func_name.lower())
    
    args = [sympy_to_s_expression(arg) for arg in expr.args]
    
    if func_name == 'Pow' and str(expr.args[1]) == '-1': return f"(/ 1 {args[0]})"
    if func_name == 'Mul' and len(args) == 2 and str(args[0]) == '-1': return f"(- {args[1]})"
        
    return f"({s_op} {' '.join(args)})"


def print_descriptor_formula(descriptor_features, coefficients, task_type, fix_intercept,
                             target_name='y', s_expr_format=False, latex_format=False, pretty_format=False, 
                             model_provider=None, coefficients_stderr=None, clean_to_original_map=None):
    """
    Formats and prints the symbolic formula for a given model.
    Includes pretty-printing for publication-ready text.
    """
    is_parametric = hasattr(model_provider, 'is_parametric') and model_provider.is_parametric
    
    temp_descriptor_features = descriptor_features
    if clean_to_original_map:
        if is_parametric and hasattr(model_provider, 'sym_expr'):
            temp_sym_expr = model_provider.sym_expr.subs(clean_to_original_map)
        else:
            temp_descriptor_features = [f.subs(clean_to_original_map) for f in descriptor_features]

    # Handle different output formats
    if pretty_format:
        formatter = lambda expr: sympy.pretty(expr, use_unicode=False, full_prec=False).replace('\n', '')
    elif latex_format:
        formatter = sympy.latex
    elif s_expr_format:
        if is_parametric: return sympy_to_s_expression(temp_sym_expr)
        if task_type in ALL_CLASSIFICATION_TASKS:
            return "\n".join([f"# D{i+1}: {sympy_to_s_expression(feat)}" for i, feat in enumerate(temp_descriptor_features)])
        
        if coefficients is None:
            formulas = [sympy_to_s_expression(feat) for feat in temp_descriptor_features]
            return "\n".join(formulas)
        
        n_tasks = coefficients.shape[0] if coefficients.ndim > 1 else 1
        formulas = []
        for t in range(n_tasks):
            coefs_t = coefficients[t] if n_tasks > 1 else coefficients.flatten()
            formula = 0
            coef_offset = 0
            if not fix_intercept:
                formula += sympy.Number(float(coefs_t[0]))
                coef_offset = 1
            if temp_descriptor_features:
                formula += sum(sympy.Number(float(coefs_t[coef_offset + i])) * f for i, f in enumerate(temp_descriptor_features))
            formulas.append(sympy_to_s_expression(formula))
        return "\n".join(formulas)
    else: # Default: verbose pretty print
        formatter = lambda expr: sympy.pretty(expr, use_unicode=True)

    # Logic for pretty text and LaTeX
    header = "=" * 60
    output = [] if (latex_format or pretty_format) else [f"\n{header}\nFinal Symbolic Model: {task_type}\n{header}"]
    
    if is_parametric:
        pretty_expr = formatter(temp_sym_expr)
        output.append(f"{target_name} = {pretty_expr}")
    elif task_type in ALL_CLASSIFICATION_TASKS:
        desc_strs = []
        for i, f in enumerate(temp_descriptor_features):
             desc_strs.append(f"D{i+1} = {formatter(f)}")
        output.append("Descriptors define the classification space:\n" + "\n".join(desc_strs))
    else:
        if coefficients is None:
            if len(temp_descriptor_features) == 1:
                formula_str = formatter(temp_descriptor_features[0])
                if latex_format:
                     return formula_str
                output.append(f"{target_name} = {formula_str}")
            else:
                 output.append("Model features:\n" + "\n".join(f"D{i+1} = {formatter(f)}" for i,f in enumerate(temp_descriptor_features)))
            return "\n".join(output)

        is_multitask = coefficients.ndim > 1 and coefficients.shape[0] > 1
        target_names = target_name if is_multitask and isinstance(target_name, list) else [target_name]
        num_tasks = coefficients.shape[0] if is_multitask else 1
        
        for t_idx in range(num_tasks):
            full_model_expr = sympy.Number(0)
            coefs_t = coefficients[t_idx] if num_tasks > 1 else coefficients.flatten()
            coef_offset = 0
            if not fix_intercept:
                full_model_expr += sympy.Number(float(coefs_t[0]))
                coef_offset = 1
            
            for i, feature in enumerate(temp_descriptor_features):
                full_model_expr += sympy.Number(float(coefs_t[i + coef_offset])) * feature

            
            # Round the coefficients for cleaner output
            if pretty_format or latex_format:
                 rounded_expr = full_model_expr.xreplace({n: round(n, 4) for n in full_model_expr.atoms(sympy.Number)})
            else:
                 rounded_expr = full_model_expr

            formula_str = formatter(rounded_expr)
            target = target_names[t_idx] if t_idx < len(target_names) else f"task_{t_idx+1}"

            if latex_format:
                output.append(f"${formatter(sympy.Symbol(target))} = {formula_str}$")
            else:
                output.append(f"{target} = {formula_str}")
            
    return "\n".join(output).replace('\n\n','\n')