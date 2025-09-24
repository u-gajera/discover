"""
Plot best model for each descriptor dimension:
    python plot_results.py --config config_manuel.json

Plot top N SIS candidates (univariate regressions):
    python plot_results.py --config config_manuel.json --mode sis --top 3

Plot the DISCOVER model for a specific dimension:
    python plot_results.py --config config_manuel.json --mode discover --dimension 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open('r') as f:
        return json.load(f)

def _read_config_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open('r') as f:
        lines = [line for line in f if not line.strip().startswith('//')]
        return json.loads("".join(lines))

def _load_data(data_path: Path, property_key: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Unsupported data extension – use CSV or Excel.")
    if property_key not in df.columns:
        raise KeyError(f"Target column '{property_key}' not found in {data_path}.")
    y = df[property_key].astype(float)
    Xdf = df.drop(columns=[property_key])
    return Xdf, y

def _sym_from_srepr_str(srepr_str: str) -> sp.Expr:
    return sp.sympify(srepr_str)


def _substitute_clean_symbols(expr: sp.Expr, clean_to_orig: Dict[str, str]) -> sp.Expr:
    repl = {}
    for s in expr.free_symbols:
        name = s.name
        if name in clean_to_orig:
            repl[s] = sp.Symbol(clean_to_orig[name])
    if repl:
        expr = expr.xreplace(repl)
    return expr


def _evaluate_descriptor(exprs: Sequence[sp.Expr], Xdf: pd.DataFrame) -> np.ndarray:

    syms = sorted({s for e in exprs for s in e.free_symbols}, key=lambda s: s.name)
    missing = [s.name for s in syms if s.name not in Xdf.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    func = sp.lambdify([sp.Symbol(s.name) for s in syms], 
                       exprs, modules={'real': np, 'Abs': np.abs, 'sqrt': np.sqrt})
    vals = func(*[Xdf[s.name].values for s in syms])
    arr = np.vstack(vals) if len(exprs) > 1 else np.array(vals)
    if arr.ndim == 2 and arr.shape[0] == len(exprs):
        arr = arr.T
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.asarray(arr, dtype=float)

def _fmt_num(x: float, digits: int = 4) -> str:
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def _wrap_text(s: str, width: int = 60) -> str:
    if len(s) <= width:
        return s
    out = []
    line = []
    count = 0
    for tok in s.split(' '):
        if count + len(tok) + 1 > width and line:
            out.append(' '.join(line))
            line = [tok]
            count = len(tok) + 1
        else:
            line.append(tok)
            count += len(tok) + 1
    if line:
        out.append(' '.join(line))
    return '\n'.join(out)


def _linear_formula_string(exprs: Sequence[sp.Expr], coefs: Sequence[float], 
                           intercept: float, y_name: str = 'y') -> str:
    """Return a human-readable linear formula string.

    Example: ``y = 0.12*(A/B) - 3.4*C + 5.6``
    ``exprs`` are SymPy expressions *already substituted* with original feature names.
    ``coefs`` array aligned with exprs.
    """
    parts: List[str] = []
    for c, e in zip(coefs, exprs):
        c_str = _fmt_num(c)
        e_str = str(e)
        if isinstance(e, sp.Symbol):
            term = f"{c_str}*{e_str}"
        else:
            term = f"{c_str}*({e_str})"
        parts.append(term)
    rhs = ' + '.join(parts)
    if intercept != 0:
        sign = '+' if intercept >= 0 else '-'  # intercept sign
        intercept_abs = _fmt_num(abs(intercept))
        rhs = f"{rhs} {sign} {intercept_abs}"
    formula = f"{y_name} = {rhs}" if rhs else f"{y_name} = {_fmt_num(intercept)}"
    return _wrap_text(formula)


def _univariate_formula_string(c: float, intercept: float, feat_name: str, 
                               y_name: str = 'y') -> str:
    term = f"{_fmt_num(c)}*{feat_name}"
    if intercept != 0:
        sign = '+' if intercept >= 0 else '-'
        term = f"{term} {sign} {_fmt_num(abs(intercept))}"
    return _wrap_text(f"{y_name} = {term}")

def _parse_coef_vector(coefs: Sequence[float], D: int, 
                       fix_intercept: bool) -> Tuple[float, np.ndarray]:
    """Return (intercept, coef_vector) from stored coefficients.

    Convention used in DISCOVER summary: when ``fix_intercept`` is *False*, the
    *first* element in the coefficient array is the intercept. Otherwise there is
    no intercept and the array length should equal D.
    """
    coefs = np.asarray(coefs, dtype=float)
    if not fix_intercept and coefs.size == D + 1:
        return float(coefs[0]), coefs[1:]
    return 0.0, coefs[:D]


def _predict_linear(descriptor: np.ndarray, intercept: float, 
                    coefs: np.ndarray) -> np.ndarray:
    return intercept + np.dot(descriptor, coefs)

def _parity_plot(y_true: pd.Series, y_pred: np.ndarray, title: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    else:
        fig = ax.figure
    ax.scatter(y_true, y_pred, s=36, alpha=0.8, edgecolor='k', linewidth=0.3)
    lims = [np.nanmin([y_true.min(), y_pred.min()]), 
            np.nanmax([y_true.max(), y_pred.max()])]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_title(f"{title}\nR²={r2:.3f}, RMSE={rmse:.3g}")
    return fig, ax

def _mode_best_allD(workdir: Path, Xdf: pd.DataFrame, y: pd.Series, 
                    show: bool, write_individual: bool, y_name: str = 'y'):
    summary_path = workdir / 'final_models_summary.json'
    symmap_path = workdir / 'symbol_map.json'
    models_summary = _read_json(summary_path)
    clean_to_orig = _read_json(symmap_path) if symmap_path.exists() else {}

    plottable = [m for m in models_summary if m.get('is_plottable')]
    if not plottable:
        raise RuntimeError("No plottable models found in final_models_summary.json")

    n = len(plottable)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    for ax, m in zip(axes.flat, plottable):
        D = int(m['Dimension'])
        fix_intercept = bool(m.get('fix_intercept', False))
        coef_list = m.get('coefficients') or []
        exprs = []
        for srepr_str in m.get('sym_features', []):
            e = _sym_from_srepr_str(srepr_str)
            e = _substitute_clean_symbols(e, clean_to_orig)
            exprs.append(e)
        if not exprs:
            ax.set_visible(False)
            continue
        desc = _evaluate_descriptor(exprs, Xdf)
        intercept, coefs = _parse_coef_vector(coef_list, D=len(exprs), 
                                              fix_intercept=fix_intercept)
        y_pred = _predict_linear(desc, intercept, coefs)
        formula_title = _linear_formula_string(exprs, coefs, intercept, 
                                               y_name=y_name)
        _parity_plot(y, y_pred, formula_title, ax=ax)
        if write_individual:
            out_path = workdir / f"parity_best_D{D}.png"
            fig_i, ax_i = _parity_plot(y, y_pred, formula_title)
            fig_i.tight_layout()
            fig_i.savefig(out_path, dpi=300)
            plt.close(fig_i)

    for j in range(len(plottable), len(axes.flat)):
        axes.flat[j].set_visible(False)

    fig.tight_layout()
    out_all = workdir / 'parity_best_allD.png'
    fig.savefig(out_all, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {out_all}")


def _mode_sis_topN(workdir: Path, Xdf: pd.DataFrame, 
                   y: pd.Series, top: int, show: bool, 
                   write_individual: bool, y_name: str = 'y'):
    sis_path = workdir / 'top_sis_candidates.csv'
    if not sis_path.is_file():
        raise FileNotFoundError(f"SIS candidates file not found: {sis_path}")

    sis_df = pd.read_csv(sis_path)
    feature_col = None
    for cand in ['Feature', 'feature', 'symbol', 'name', 'descriptor']:
        if cand in sis_df.columns:
            feature_col = cand
            break
    if feature_col is None:
        feature_col = sis_df.columns[0]

    top_df = sis_df.head(top)

    n = len(top_df)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    for ax, (_, row) in zip(axes.flat, top_df.iterrows()):
        feat_name = row[feature_col]
        if feat_name not in Xdf.columns:
            symmap_path = workdir / 'symbol_map.json'
            if symmap_path.is_file():
                clean_to_orig = _read_json(symmap_path)
                # invert mapping
                orig = clean_to_orig.get(feat_name)
                if orig and orig in Xdf.columns:
                    feat_name = orig
        if feat_name not in Xdf.columns:
            ax.set_visible(False)
            continue
        x = Xdf[feat_name].values.reshape(-1,1)
        lr = LinearRegression(fit_intercept=True)
        lr.fit(x, y.values)
        y_pred = lr.predict(x)
        formula_title = _univariate_formula_string(lr.coef_[0], 
                                                   lr.intercept_, 
                                                   feat_name, 
                                                   y_name=y_name)
        _parity_plot(y, y_pred, formula_title, ax=ax)
        if write_individual:
            out_path = workdir / f"parity_sis_{feat_name}.png"
            fig_i, ax_i = _parity_plot(y, y_pred, formula_title)
            fig_i.tight_layout()
            fig_i.savefig(out_path, dpi=300)
            plt.close(fig_i)

    for j in range(len(top_df), len(axes.flat)):
        axes.flat[j].set_visible(False)

    fig.tight_layout()
    out_all = workdir / f'parity_sis_top{top}.png'
    fig.savefig(out_all, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {out_all}")


def _mode_discover_D(workdir: Path, Xdf: pd.DataFrame, y: pd.Series, 
                  D_req: int, show: bool, write_individual: bool, y_name: str = 'y'):
    summary_path = workdir / 'final_models_summary.json'
    symmap_path = workdir / 'symbol_map.json'
    models_summary = _read_json(summary_path)
    clean_to_orig = _read_json(symmap_path) if symmap_path.exists() else {}

    match = None
    for m in models_summary:
        if int(m['Dimension']) == int(D_req):
            match = m
            break
    if match is None:
        raise ValueError(f"Requested dimension D={D_req} not found in final_models_summary.json")
    if not match.get('is_plottable'):
        raise RuntimeError(f"Model at D={D_req} is not plottable (parametric or missing coefficients).")

    exprs = []
    for srepr_str in match.get('sym_features', []):
        e = _sym_from_srepr_str(srepr_str)
        e = _substitute_clean_symbols(e, clean_to_orig)
        exprs.append(e)
    desc = _evaluate_descriptor(exprs, Xdf)
    intercept, coefs = _parse_coef_vector(match.get('coefficients') or [], 
                                    D=len(exprs), 
                                    fix_intercept=match.get('fix_intercept', False))
    y_pred = _predict_linear(desc, intercept, coefs)
    formula_title = _linear_formula_string(exprs, coefs, intercept, y_name=y_name)
    fig, ax = _parity_plot(y, y_pred, formula_title)
    fig.tight_layout()
    out_path = workdir / f'parity_discover_D{D_req}.png'
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {out_path}")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Plot results from a DISCOVER run using a config file.")
    
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the JSON configuration file used in the DISCOVER run.")
    
    parser.add_argument('--mode', choices=['best','sis','discover'], default='best', 
                        help="Plotting mode. 'best': all dimensions, 'sis': top candidates, 'discover': one dimension. Default: 'best'.")
    parser.add_argument('--top', type=int, default=3, 
                        help="Top N SIS candidates to plot (for --mode sis). Default: 3.")
    parser.add_argument('--dimension','-D', type=int, default=None, 
                        help="Descriptor dimension to plot (for --mode discover).")
    parser.add_argument('-y','--property-key', default=None, 
                        help="Name of target column in data file. Overrides value from config file.")
    parser.add_argument('--show', action='store_true', 
                        help="Display figures interactively in addition to saving.")
    parser.add_argument('--no-individual', action='store_true', 
                        help="Skip writing individual PNGs (only combined figure).")

    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = _read_config_json(config_path)

    workdir_str = config.get('workdir')
    if not workdir_str:
        parser.error("The 'workdir' key is missing from the config file.")
    
    data_file_str = config.get('data_file')
    if not data_file_str:
        parser.error("The 'data_file' key is missing from the config file.")

    resolved_property_key = args.property_key or config.get('property_key')
    if not resolved_property_key:
        parser.error("The 'property_key' must be specified either in the config file or with the --property-key argument.")

    workdir = Path(workdir_str).expanduser().resolve()
    data_path = Path(data_file_str).expanduser().resolve()

    if not workdir.is_dir():
        raise NotADirectoryError(f"Workdir specified in config not found: {workdir}")

    Xdf, y = _load_data(data_path, resolved_property_key)

    write_individual = not args.no_individual

    if args.mode == 'best':
        _mode_best_allD(workdir, Xdf, y, show=args.show, write_individual=write_individual, y_name=resolved_property_key)
    elif args.mode == 'sis':
        _mode_sis_topN(workdir, Xdf, y, top=args.top, show=args.show, write_individual=write_individual, y_name=resolved_property_key)
    elif args.mode == 'discover':
        if args.dimension is None:
            parser.error("--mode discover requires --dimension")
        _mode_discover_D(workdir, Xdf, y, D_req=args.dimension, show=args.show, write_individual=write_individual, y_name=resolved_property_key)
    else:
        raise RuntimeError(f"Unhandled mode: {args.mode}")


if __name__ == '__main__':  
    main()