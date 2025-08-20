"""
Feature-space construction utilities.

Takes the primary features listed in your CSV and, using a grammar of basic
operators (+, −, ×, ÷, log, √, etc.), generates the huge pool of candidate
descriptors examined by SISSO.

Example:
    Primary features  :  A (lattice constant), B (bulk modulus)
    Generated pool    :  A+B, A−B, A/B, log(B), √A, (A/B)², …

These symbolic expressions are stored as SymPy trees so they can be manipulated
and evaluated efficiently.
"""

import pandas as pd
import numpy as np
import sympy
from pathlib import Path
import warnings

from joblib import Memory
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from .constants import ALL_CLASSIFICATION_TASKS, REGRESSION


# Optional CuPy and Torch acceleration -----------------------------------------------------------
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    class cp_dummy:
        ndarray = type(None)
        def asnumpy(self, x): return np.asarray(x)
    cp = cp_dummy()

try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    class torch:
        Tensor = type(None)
        @staticmethod
        def device(x): return None
        @staticmethod
        def from_numpy(x): return x
        @staticmethod
        def cos(arr): return np.cos(arr)

# Pint -----------------------------------------------------------------
try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

# ------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------

def _safe_min_max(arr, xp=np):
    if xp == torch:
        amax = float(arr.max().item())
        amin = float(arr.min().item())
    else:
        amax = float(xp.max(arr))
        amin = float(xp.min(arr))
    return amin, amax

def _calculate_complexity(expr):
    """Calculate the complexity of a SymPy expression by counting operations."""
    return 1 + sympy.count_ops(expr)


#  Operator Definitions
PARAMETRIC_OP_DEFS = {
    'exp-': {
        'func': lambda f, p, xp: xp.exp(-p[0] * f),
        'sym_func': lambda s, p_sym: sympy.exp(-p_sym[0] * s),
        'p_names': ['p'],
        'p_initial': [1.0],
        'p_bounds': [(1e-4, 1e4)],
        'unit_check': lambda u: u.dimensionless,
    },
    'pow': {
        'func': lambda f, p, xp: xp.power(f, p[0]),
        'sym_func': lambda s, p_sym: s**p_sym[0],
        'p_names': ['p'],
        'p_initial': [1.0],
        'p_bounds': [(-3.0, 3.0)],
        'domain_check': lambda f_uf: f_uf.is_positive,
        'unit_check': None,
    },
}
CUSTOM_UNARY_OP_DEFS = {
    'log1p': {
        'func': np.log1p,
        'sym_func': lambda s: sympy.log(s + 1),
        'unit_check': lambda u: u.dimensionless,
        'domain_check': lambda f: np.all(f.values > -1),
        'name_func': lambda n: f"log1p({n})",
    },
    'gaussian': {
        'func': lambda x: np.exp(-x**2),
        'sym_func': lambda s: sympy.exp(-s**2),
        'unit_check': lambda u: u.dimensionless,
        'domain_check': None,
        'name_func': lambda n: f"gaussian({n})",
    }
}
CUSTOM_BINARY_OP_DEFS = {
    'harmonic_mean': {
        'func': lambda f1, f2: 2 * f1 * f2 / (f1 + f2),
        'sym_func': lambda s1, s2: 2 * s1 * s2 / (s1 + s2),
        'unit_op': lambda u1, u2: u1,
        'unit_check': lambda u1, u2: u1.dimensionality == u2.dimensionality,
        'domain_check': lambda f1, f2: not np.any(np.abs(f1.values + f2.values) < 1e-9),
        'op_name': "<H>"
    },
    'abs_diff': {
        'func': lambda f1, f2: abs(f1 - f2),
        'sym_func': lambda s1, s2: sympy.Abs(s1 - s2),
        'unit_op': lambda u1, u2: u1,
        'unit_check': lambda u1, u2: u1.dimensionality == u2.dimensionality,
        'domain_check': None,
        'op_name': "|-|"
    }
}
#  Feature Generation (with Unit-Awareness and Operator Control)
class UFeature:
    def __init__(self, name, values, unit, ureg, sym_expr, 
                 base_features, xp=np, dtype=np.float64, torch_device=None):
        self.name = name
        self.xp = xp
        self.torch_device = torch_device
        self.ureg = ureg
        self.sym_expr = sym_expr
        self.base_features = base_features

        vals = values.magnitude if hasattr(values, 'magnitude') else values

        if xp == torch:
            if isinstance(vals, torch.Tensor):
                # If it's already a tensor, just use it. No need to copy.
                self.values = vals
            else:
                # Otherwise, create it from numpy/list.
                torch_dtype = {np.float32: torch.float32,
                               np.float64: torch.float64}.get(np.dtype(dtype).type, torch.float32)
                self.values = torch.tensor(vals, dtype=torch_dtype, device=self.torch_device)
        else:
            self.values = xp.asarray(vals, dtype=dtype)

        self.unit = ureg(unit) if isinstance(unit, str) and ureg else unit
        self.is_positive = self.xp.all(self.values > 1e-9)
        self.has_zero    = self.xp.any(self.xp.abs(self.values) < 1e-9)
        self.min_val = self.xp.min(self.values)
        self.max_val = self.xp.max(self.values)

    def _apply_binary_op(self, other, op_func, sym_op_func, op_name, unit_op=None, 
                         unit_check=None, domain_check=None):
         if self.ureg and unit_check and not unit_check(self.unit, 
                                                        other.unit): return None
         if domain_check and not domain_check(self, other): return None

         if not self.ureg:
              try:
                  new_values = op_func(self.values, other.values)
                  new_sym_expr = sym_op_func(self.sym_expr, other.sym_expr)
                  new_name = f"({self.name}{op_name}{other.name})"
                  new_base_features = self.base_features.union(other.base_features)
                  return UFeature(new_name, new_values, None, self.ureg, 
                                  new_sym_expr, new_base_features, self.xp, 
                                  dtype=self.values.dtype, 
                                  torch_device=self.torch_device)
              except: return None
         try:
            new_unit = unit_op(self.unit, other.unit) if unit_op else op_func(self.unit, 
                                                                            other.unit)
            new_values = op_func(self.values, other.values)
            new_sym_expr = sym_op_func(self.sym_expr, other.sym_expr)
            new_name = f"({self.name}{op_name}{other.name})"
            new_base_features = self.base_features.union(other.base_features)
            return UFeature(new_name, new_values, new_unit, self.ureg, new_sym_expr, 
                            new_base_features, self.xp, dtype=self.values.dtype, 
                            torch_device=self.torch_device)
         except (pint.DimensionalityError, ValueError, TypeError, 
                            OverflowError): return None

    def __add__(self, other): return self._apply_binary_op(other, 
                                        lambda a, b: a + b, lambda a, b: a + b, '+')
    def __sub__(self, other): return self._apply_binary_op(other, 
                                        lambda a, b: a - b, lambda a, b: a - b, '-')
    def __mul__(self, other): return self._apply_binary_op(other, 
                                        lambda a, b: a * b, lambda a, b: a * b, '*')
    def __truediv__(self, other):
        if other.has_zero: return None
        return self._apply_binary_op(other, lambda a, b: a / b, lambda a, b: a / b, '/')

    def _apply_unary_op(self, val_op, unit_op, sym_op, name_func, 
                        unit_check=None, domain_check=None):
        if self.xp == torch: finfo = torch.finfo(self.values.dtype)
        else: finfo = self.xp.finfo(self.values.dtype)
        if self.max_val > finfo.max / 2 or self.min_val < finfo.min / 2:
            if 'exp' in name_func(''): return None

        if self.ureg and unit_check and not unit_check(self.unit): return None
        if domain_check and not domain_check(self): return None
        try:
            new_values = val_op(self.values)
            if not self.xp.all(self.xp.isfinite(new_values)): return None
            new_unit = unit_op(self.unit) if self.ureg else None
            new_sym_expr = sym_op(self.sym_expr)
            new_name = name_func(self.name)
            new_base_features = self.base_features
            return UFeature(new_name, new_values, new_unit, self.ureg, new_sym_expr, 
                            new_base_features, self.xp, dtype=self.values.dtype, 
                            torch_device=self.torch_device)
        except (ValueError, TypeError, pint.DimensionalityError, 
                            OverflowError): return None

    def power(self, p):
        if p != int(p) and not self.is_positive : return None
        if self.has_zero and p < 0: return None
        return self._apply_unary_op(lambda a: a**p, lambda u: u**p, 
                                    lambda s: s**p, lambda n: f"({n})**{p}")

    def sqrt(self): return self._apply_unary_op(lambda a: self.xp.sqrt(a), 
                            lambda u: u**0.5, sympy.sqrt, lambda n: f"sqrt({n})", 
                            domain_check=lambda s: s.is_positive or s.has_zero)
    def cbrt(self): return self._apply_unary_op(lambda a: a.cbrt() if self.xp==torch else self.xp.cbrt(a), 
                            lambda u: u**(1/3.), sympy.cbrt, lambda n: f"cbrt({n})")
    def abs(self):  return self._apply_unary_op(self.xp.abs, lambda u: u, sympy.Abs, 
                                                lambda n: f"Abs({n})")
    def inv(self):  return self._apply_unary_op(lambda a: 1.0/a, lambda u: 1.0/u, 
                            lambda s: 1/s, lambda n: f"1/({n})", 
                            domain_check=lambda s: not s.has_zero)
    def sign(self): return self._apply_unary_op(self.xp.sign, 
                            lambda u: u, sympy.sign, lambda n: f"sign({n})", 
                            unit_check=lambda u: u.dimensionless)
    def log(self): return self._apply_unary_op(self.xp.log, lambda u: u, sympy.log, 
                            lambda n: f"log({n})", unit_check=lambda u: u.dimensionless, 
                            domain_check=lambda s: s.is_positive)
    def exp(self): return self._apply_unary_op(self.xp.exp, lambda u: u, sympy.exp, 
                            lambda n: f"exp({n})", 
                            unit_check=lambda u: u.dimensionless)
    def exp_neg(self): return self._apply_unary_op(lambda a: self.xp.exp(-a), 
                            lambda u: u, lambda s: sympy.exp(-s), 
                            lambda n: f"exp(-{n})", 
                            unit_check=lambda u: u.dimensionless)
    def sin(self): return self._apply_unary_op(self.xp.sin, lambda u: u, 
                            sympy.sin, lambda n: f"sin({n})", 
                            unit_check=lambda u: u.dimensionless)
    def cos(self): return self._apply_unary_op(self.xp.cos, lambda u: u, 
                            sympy.cos, lambda n: f"cos({n})", 
                            unit_check=lambda u: u.dimensionless)

    @staticmethod
    def get_all_unary_op_names():
       return [
           'sqrt', 'cbrt', 'abs', 'log', 'exp', 'exp-', 'sin', 'cos',
           'inv', 'sign', 'sq', 'cb', 'pow6', 'pow-2', 'pow-3',
           'pow0.5', 'pow-0.5'
       ]

    def get_unary_op_func(self, op_name):
        """Returns the function for a given unary operator name."""
        op_map = {
           'sqrt': self.sqrt, 'cbrt': self.cbrt, 'abs': self.abs, 'log': self.log,
           'exp': self.exp, 'exp-': self.exp_neg, 'sin': self.sin, 'cos': self.cos,
           'inv': self.inv, 'sign': self.sign,
           'sq': lambda: self.power(2), 'cb': lambda: self.power(3), 
           'pow6': lambda: self.power(6),
           'pow-2': lambda: self.power(-2), 'pow-3': lambda: self.power(-3),
           'pow0.5': self.sqrt, 'pow-0.5': lambda: self.power(-0.5)
        }
        return op_map.get(op_name)

    @staticmethod
    def get_all_binary_op_names():
         return ['add', 'sub', 'mul', 'div']

    @staticmethod
    def get_binary_op_method_name(op_name):
        return {'add': '__add__', 'sub': '__sub__', 'mul': '__mul__', 
                'div': '__truediv__'}.get(op_name)


def apply_op(f, op_func, min_val, max_val):
    """Helper to apply a unary operator and check for validity."""
    if f.values.dtype in [np.float32, (torch.float32 if TORCH_AVAILABLE else None)]:
        epsilon = 1e-7
    else:
        epsilon = 1e-9
    try:
       new_feat = op_func()
       if new_feat and new_feat.xp.all(new_feat.xp.isfinite(new_feat.values)):
           abs_vals = new_feat.xp.abs(new_feat.values)
           if new_feat.xp.any(abs_vals < min_val) or new_feat.xp.any(abs_vals > max_val):
               return None
           if new_feat.xp.all(abs_vals < epsilon): return None
           amin, amax = _safe_min_max(new_feat.values, xp=new_feat.xp)
           if amax - amin < epsilon: return None
           return new_feat.sym_expr, new_feat
    except Exception: pass
    return None


def apply_binary_op(f1, f2, op_name, min_val, max_val):
    """Helper to apply a binary operator and check for validity."""
    epsilon = 1e-7 if f1.xp == torch and f1.values.dtype == torch.float32 else 1e-9
    try:
        op_method_name = UFeature.get_binary_op_method_name(op_name)
        if not op_method_name: return None

        if op_method_name in ['__sub__', '__truediv__'] and f1.sym_expr == f2.sym_expr:
            return None

        new_feat = getattr(f1, op_method_name)(f2)

        if new_feat and new_feat.xp.all(new_feat.xp.isfinite(new_feat.values)):
            abs_vals = new_feat.xp.abs(new_feat.values)
            if new_feat.xp.any(abs_vals < min_val) or new_feat.xp.any(abs_vals > max_val):
                return None
            if new_feat.xp.all(abs_vals < epsilon): return None
            amin, amax = _safe_min_max(new_feat.values, xp=new_feat.xp)
            if amax - amin < epsilon: return None
            return new_feat.sym_expr, new_feat
    except Exception: pass
    return None

def apply_custom_binary_op(f1, f2, op_def, min_val, max_val):
    """Helper to apply a custom binary operator."""
    epsilon = 1e-7 if f1.xp == torch and f1.values.dtype == torch.float32 else 1e-9
    try:
        op_name = op_def.get('op_name', '?')
        new_feat = f1._apply_binary_op(
            f2, op_def['func'], op_def['sym_func'], op_name,
            unit_op=op_def.get('unit_op'),
            unit_check=op_def.get('unit_check'),
            domain_check=op_def.get('domain_check')
        )
        if new_feat and new_feat.xp.all(new_feat.xp.isfinite(new_feat.values)):
            abs_vals = new_feat.xp.abs(new_feat.values)
            if new_feat.xp.any(abs_vals < min_val) or new_feat.xp.any(abs_vals > max_val): return None
            if new_feat.xp.all(abs_vals < epsilon): return None
            amin, amax = _safe_min_max(new_feat.values, xp=new_feat.xp)
            if amax - amin < epsilon: return None
            return new_feat.sym_expr, new_feat
    except Exception: pass
    return None

# It no longer copies data to the CPU if it's already on the GPU.
def apply_parametric_op(feature, op_def, min_val, max_val):
    """Applies a parametric operator with its initial default parameter values."""
    epsilon = 1e-7 if feature.xp == torch and feature.values.dtype == torch.float32 else 1e-9
    xp = feature.xp
    try:
        p_initial = op_def['p_initial']
        if op_def.get('domain_check') and not op_def['domain_check'](feature): return None
        if feature.ureg and op_def.get('unit_check') and not op_def['unit_check'](feature.unit): return None

        new_values = op_def['func'](feature.values, p_initial, xp)

        if not xp.all(xp.isfinite(new_values)): return None

        abs_vals = xp.abs(new_values)
        if xp.any(abs_vals < min_val) or xp.any(abs_vals > max_val): return None
        if xp.all(abs_vals < epsilon): return None
        amin, amax = _safe_min_max(new_values, xp=xp)
        if amax - amin < epsilon: return None

        p_symbols = [sympy.Symbol(p) for p in op_def['p_names']]
        new_sym_expr = op_def['sym_func'](feature.sym_expr, p_symbols).subs(zip(p_symbols, p_initial))
        new_unit = feature.unit
        if op_def.get('unit_check'): new_unit = 'dimensionless'
        elif op_def['p_names'] == ['p']: new_unit = feature.unit ** p_initial[0]
        op_name = list(PARAMETRIC_OP_DEFS.keys())[list(PARAMETRIC_OP_DEFS.values()).index(op_def)]
        new_name = f"{op_name}(p={p_initial})({feature.name})"

        new_feat = UFeature(new_name, new_values, new_unit, feature.ureg, 
                            new_sym_expr, feature.base_features, feature.xp, 
                            dtype=feature.values.dtype, 
                            torch_device=feature.torch_device)
        return new_feat.sym_expr, new_feat
    except Exception:
        return None

def _canonicalize_expr(expr):
    """Return a canonical SymPy expression for duplicate detection."""
    try:
        e = expr
        def _sq_pow_to_abs(ex):
            if ex.is_Pow and ex.exp == sympy.Rational(1, 2):
                base = ex.base
                if base.is_Pow and base.exp == 2:
                    return sympy.Abs(base.base)
            return ex
        e = e.replace(_sq_pow_to_abs)

        def _x2_over_sqrtx2(ex):
            if ex.is_Mul and len(ex.args) == 2:
                a, b = ex.args
                if a.is_Pow and a.exp == 2 and b.is_Pow and b.exp == -sympy.Rational(1, 2):
                    bb = b.base
                    if bb.is_Pow and bb.exp == 2 and bb.base == a.base:
                        return sympy.Abs(a.base)
            return ex
        e = e.replace(_x2_over_sqrtx2)

        def _abs2_over_abs(ex):
            if ex.is_Mul and len(ex.args) == 2:
                a, b = ex.args
                if a.is_Pow and a.base.func is sympy.Abs and a.exp == 2 and b.is_Pow and b.exp == -1 and b.base.func is sympy.Abs:
                    if a.base.args[0] == b.base.args[0]:
                        return sympy.Abs(a.base.args[0])
            return ex
        e = e.replace(_abs2_over_abs)

        e = sympy.powdenest(sympy.simplify(e), force=True)
        e = sympy.simplify(e)
        return e
    except Exception:
        return expr

def _is_numerically_duplicate(new_feat, existing_feats, tol=1e-12):
    """
    True if *new_feat* is numerically identical (within tol) to any in *existing_feats*.
    This version is device-aware to avoid GPU->CPU data transfer bottlenecks.
    """
    nv = new_feat.values
    xp = new_feat.xp

    for ef in existing_feats:
        ev = ef.values
        if nv.shape != ev.shape:
            continue

        if xp == torch:
            if torch.allclose(nv, ev, rtol=tol, atol=tol, equal_nan=True):
                return True
        else:
            if xp.allclose(nv, ev, rtol=tol, atol=tol, equal_nan=True):
                return True
    return False

# ------------------------------------------------------------------------------
# Core iterative generator (patched sections marked)
# ----------------------------------------------------------------------------

def generate_features_iteratively(X, y, primary_units, depth, n_features_per_sis_iter,
                                  sis_score_degeneracy_epsilon,
                                  n_jobs, use_cache, workdir, op_rules, parametric_ops,
                                  min_abs_feat_val, max_abs_feat_val, interaction_only,
                                  task_type, sis_method, multitask_sis_method,
                                  unary_rungs=1, xp=np, dtype=np.float64, 
                                  torch_device=None):
    """Breadth-first SIS-driven feature generator (feature-space only)."""
    from .scoring import run_SIS  # local import to avoid circular

    # --- Setup -----------------------------------------------------------
    ureg = None
    if PINT_AVAILABLE:
        ureg = pint.UnitRegistry()
        if 'electron_volt' not in ureg:
            ureg.define('electron_volt = 1.602176634e-19 * joule = eV')
    elif primary_units:
        warnings.warn("Pint not available. Generating features without unit checks.")

    clean_names = [f"f{i}" for i in range(len(X.columns))]
    primary_symbols = [sympy.Symbol(s, real=True) for s in clean_names]
    original_symbols = [sympy.Symbol(c, real=True) for c in X.columns]
    clean_to_original_map = dict(zip(primary_symbols, original_symbols))
    X_clean = X.copy(); X_clean.columns = clean_names

    initial_features = []
    if not primary_units: primary_units = {}
    for i, name in enumerate(X_clean.columns):
        original_name = str(original_symbols[i])
        unit = primary_units.get(original_name, 'dimensionless' if ureg else None)
        if original_name not in primary_units and ureg:
            warnings.warn(f"No unit defined for '{original_name}'. Treating as dimensionless.")
        initial_features.append(UFeature(original_name, X_clean[name].values, unit, ureg,
                                         primary_symbols[i], {original_name}, 
                                         xp, dtype, torch_device))

    device_name = "CPU"
    if xp == cp: device_name = "NVIDIA GPU"
    elif xp == torch and torch_device is not None: device_name = "Apple/MPS GPU"
    print(f"\nStarting iterative feature generation on {device_name} to depth {depth}...")
    print(f"  Keeping top {n_features_per_sis_iter} features per iteration.")

    candidate_features_map = {_canonicalize_expr(feat.sym_expr): feat for feat in initial_features}
    last_level_features = initial_features

    BUILTIN_UNARY_NAMES = UFeature.get_all_unary_op_names()
    BUILTIN_BINARY_NAMES = UFeature.get_all_binary_op_names()

    for d in range(1, depth + 1):
        print(f"\n  Depth {d}: Generating new features...")
        newly_added_this_level = []

        # --- Unary ops ---------------------------------------------------------
        apply_unary_now = (not interaction_only) and (d <= unary_rungs)
        if apply_unary_now:
            for f in last_level_features:
                for rule in op_rules:
                    op_name = rule['op']
                    if 'exclude_features' in rule and not f.base_features.isdisjoint(set(rule['exclude_features'])):
                        continue
                    res = None
                    if op_name in BUILTIN_UNARY_NAMES:
                        op_func = f.get_unary_op_func(op_name)
                        if op_func:
                            res = apply_op(f, op_func, min_abs_feat_val, max_abs_feat_val)
                    elif op_name in CUSTOM_UNARY_OP_DEFS:
                        op_def = CUSTOM_UNARY_OP_DEFS[op_name]
                        op_func = lambda: f._apply_unary_op(op_def['func'],
                                                            op_def.get('unit_op', lambda u: u),
                                                            op_def['sym_func'],
                                                            op_def['name_func'],
                                                            op_def.get('unit_check'),
                                                            op_def.get('domain_check'))
                        res = apply_op(f, op_func, min_abs_feat_val, max_abs_feat_val)
                    elif op_name in parametric_ops and op_name in PARAMETRIC_OP_DEFS:
                        res = apply_parametric_op(f, PARAMETRIC_OP_DEFS[op_name],
                                                  min_abs_feat_val, max_abs_feat_val)

                    if res:
                        expr, feat = res
                        simplified_expr = _canonicalize_expr(expr)
                        if not simplified_expr.is_Number:
                            if simplified_expr not in candidate_features_map:
                                candidate_features_map[simplified_expr] = feat
                                newly_added_this_level.append(feat)

        # --- Binary ops --------------------------------------------------
        all_candidate_features_list = list(candidate_features_map.values())
        for f1 in last_level_features:
            for f2 in all_candidate_features_list:
                for rule in op_rules:
                    op_name = rule['op']

                    if f1 is f2 and op_name in ('add', 'sub'):
                        continue

                    if op_name in ('add', 'sub'):
                        union_size = len(f1.base_features | f2.base_features)
                        max_size   = max(len(f1.base_features), len(f2.base_features))
                        if union_size == max_size:
                            continue

                    if 'exclude_features' in rule:
                        excluded = set(rule['exclude_features'])
                        if not f1.base_features.isdisjoint(excluded) or not f2.base_features.isdisjoint(excluded):
                            continue
                    res = None
                    if op_name in BUILTIN_BINARY_NAMES:
                        if op_name in ['add', 'mul'] and f2.name < f1.name:
                            continue
                        res = apply_binary_op(f1, f2, op_name, min_abs_feat_val, max_abs_feat_val)
                    elif op_name in CUSTOM_BINARY_OP_DEFS:
                        if f2.name < f1.name:
                            continue
                        res = apply_custom_binary_op(f1, f2, CUSTOM_BINARY_OP_DEFS[op_name],
                                                     min_abs_feat_val, max_abs_feat_val)

                    if res:
                        expr, feat = res
                        simplified_expr = _canonicalize_expr(expr)
                        if not simplified_expr.is_Number:
                            if simplified_expr not in candidate_features_map:
                                candidate_features_map[simplified_expr] = feat
                                newly_added_this_level.append(feat)

        # It must be run at every depth level.

        print(f"    Generated {len(newly_added_this_level)} new unique features. Total candidates: {len(candidate_features_map)}")
        if not newly_added_this_level:
            print("  No new valid features could be generated. Stopping iteration.")
            break

        # --- SIS screening -------------------------------------------------------------------------------------
        is_gpu_run = xp != np
        candidate_names = [str(_canonicalize_expr(f.sym_expr)) for f in candidate_features_map.values()]
        candidate_sym_map = {name: f.sym_expr for name, f in zip(candidate_names, candidate_features_map.values())}

        if sis_method == 'decision_tree':
            print(f"  Screening and pruning {len(candidate_features_map)} candidates using Decision Tree (Bayesian Apriori)...")
            if is_gpu_run:
                warnings.warn("Decision Tree SIS is CPU-only. Data will be moved from GPU to CPU for this step.")

            # 1. Get numerical data on CPU
            temp_values_dict = {
                name: (cp.asnumpy(feat.values) if isinstance(feat.values, cp.ndarray)
                       else feat.values.cpu().numpy() if isinstance(feat.values, torch.Tensor)
                       else feat.values)
                for name, feat in zip(candidate_names, candidate_features_map.values())
            }
            temp_phi_df = pd.DataFrame(temp_values_dict, index=X.index)

            # 2. Get feature importances (Likelihood)
            print("    - Training decision tree to get feature importances (likelihood)...")
            if task_type in ALL_CLASSIFICATION_TASKS:
                dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            else: # Regression or Multitask
                dt = DecisionTreeRegressor(max_depth=5, random_state=42)
            
            dt.fit(temp_phi_df, y)
            importances = pd.Series(dt.feature_importances_, index=temp_phi_df.columns)

            # 3. Calculate complexity (Prior)
            print("    - Calculating complexity score for each feature (prior)...")
            complexities = pd.Series({
                str(_canonicalize_expr(f.sym_expr)): _calculate_complexity(f.sym_expr)
                for f in candidate_features_map.values()
            })
            # Prior is inverse of complexity. Add 1 to avoid division by zero and smooth the effect.
            priors = 1.0 / (1.0 + complexities)

            # 4. Combine to get final score (Posterior)
            print("    - Combining importance and complexity for final ranking...")
            # Align series before multiplying, fill missing with 0
            aligned_importances, aligned_priors = importances.align(priors, fill_value=0)
            final_scores = aligned_importances * aligned_priors
            
            sorted_scores_series = final_scores.sort_values(ascending=False)
            sis_score_label = "Hybrid Score"

        else: # Default correlation-based SIS
            print(f"  Screening and pruning {len(candidate_features_map)} candidates using correlation...")
            sis_score_label = "SIS Score"
            if is_gpu_run:
                feature_tensors = [feat.values.reshape(-1, 1) for feat in candidate_features_map.values()]
                phi_tensor = xp.hstack(feature_tensors) if xp == cp else torch.cat(feature_tensors, dim=1)
                sorted_scores_series = run_SIS(
                    phi=None, y=y, task_type=task_type, xp=xp, multitask_sis_method=multitask_sis_method,
                    phi_tensor=phi_tensor, phi_names=candidate_names
                )
            else:
                temp_values_dict = {name: feat.values for name, feat in zip(candidate_names, 
                                                        candidate_features_map.values())}
                temp_phi_df = pd.DataFrame(temp_values_dict, index=X.index)
                sorted_scores_series = run_SIS(temp_phi_df, y, task_type, xp=xp, 
                                               multitask_sis_method=multitask_sis_method)

        if not isinstance(sorted_scores_series, pd.Series) or sorted_scores_series.empty:
            print("  SIS screening returned no features. Stopping."); break
        
        print(f"  SIS screening complete. Top features at this depth:")
        top_features_data = []
        y_np = y.values if isinstance(y, pd.Series) else y
        
        for i in range(min(5, len(sorted_scores_series))):
            feat_name = sorted_scores_series.index[i]
            sis_score = sorted_scores_series.iloc[i]
            sym_expr = candidate_sym_map.get(feat_name)
            formula = feat_name
            if sym_expr:
                subbed_expr = sym_expr.subs(clean_to_original_map)
                formula = str(subbed_expr)

            # Calculate RMSE and R2 for the single feature
            rmse = np.nan
            r2 = np.nan
            try:
                # <--- MODIFIED CODE START
                if 'phi_tensor' in locals() and is_gpu_run and sis_method == 'correlation':
                # <--- MODIFIED CODE END
                    feat_idx = candidate_names.index(feat_name)
                    # Get feature values, move to CPU for sklearn
                    if xp == cp:
                        feature_values_np = cp.asnumpy(phi_tensor[:, feat_idx])
                    else: # torch
                        feature_values_np = phi_tensor[:, feat_idx].cpu().numpy()
                else:
                    feature_values_np = temp_phi_df[feat_name].values
                
                X_feat = feature_values_np.reshape(-1, 1)
                model = LinearRegression().fit(X_feat, y_np)
                y_pred = model.predict(X_feat)
                rmse = np.sqrt(mean_squared_error(y_np, y_pred))
                r2 = r2_score(y_np, y_pred)
            except Exception:
                pass

            top_features_data.append((i + 1, sis_score, rmse, r2, formula))

        if top_features_data:
            max_formula_len = max(len(f) for _, _, _, _, f in top_features_data)
            # Header
            rank_w, sis_w, rmse_w, r2_w = 5, 12, 11, 11
            header = f"    {'Rank':<{rank_w}} | {sis_score_label:<{sis_w}} | {'RMSE':<{rmse_w}} | {'R2 Score':<{r2_w}} | Formula"
            separator = f"    {'-'*rank_w}-+-{'-'*sis_w}-+-{'-'*rmse_w}-+-{'-'*r2_w}-+-{'-'*max(7, max_formula_len)}"
            print(header)
            print(separator)
            # Rows
            for rank, sis_score, rmse, r2, formula in top_features_data:
                print(f"    {rank:<{rank_w}} | {sis_score:<{sis_w}.4g} | {rmse:<{rmse_w}.4f} | {r2:<{r2_w}.4f} | {formula}")


        # keep top-k + degeneracy ------------------------------------------------------------------------------
        if len(sorted_scores_series) <= n_features_per_sis_iter:
            top_k_names = sorted_scores_series.index.tolist()
        else:
            score_at_k = sorted_scores_series.iloc[n_features_per_sis_iter - 1]
            score_threshold = score_at_k - sis_score_degeneracy_epsilon
            qualifying_features = sorted_scores_series[sorted_scores_series >= score_threshold]
            top_k_names = qualifying_features.index.tolist()
            if len(top_k_names) > n_features_per_sis_iter:
                print(f"    (Kept {len(top_k_names) - n_features_per_sis_iter} additional features due to score degeneracy)")

        # Within top-k, remove near-numerical duplicates (On GPU if possible) --------------------------------
        if len(top_k_names) > 1:
            print(f"    Filtering {len(top_k_names)} top-scoring features to remove redundancies.")
            # <--- MODIFIED CODE START
            if is_gpu_run and sis_method == 'correlation':
            # <--- MODIFIED CODE END
                # to avoid a costly GPU->CPU transfer of the correlation matrix.
                top_k_indices = [candidate_names.index(n) for n in top_k_names]
                top_k_tensor = phi_tensor[:, top_k_indices]

                top_k_tensor_c = top_k_tensor - top_k_tensor.mean(axis=0)
                cov_matrix = top_k_tensor_c.T @ top_k_tensor_c / (top_k_tensor_c.shape[0] - 1)

                std_args = {'unbiased': True} if xp == torch else {'ddof': 1}
                std_devs = xp.std(top_k_tensor, axis=0, **std_args)

                if std_devs.ndim > 1: std_devs = std_devs.squeeze()

                # Calculate correlation matrix on GPU
                corr_matrix_gpu = cov_matrix / xp.outer(std_devs, std_devs)
                corr_matrix_gpu[xp.isnan(corr_matrix_gpu)] = 0 # Handle potential div by zero

                if xp == torch:
                    corr_matrix_gpu.fill_diagonal_(1.0)
                else: # for numpy and cupy
                    corr_matrix_gpu[xp.diag_indices_from(corr_matrix_gpu)] = 1.0

                corr_matrix = xp.abs(corr_matrix_gpu)
                
                # Greedy filtering on GPU
                to_remove_mask = xp.zeros(len(top_k_names), dtype=bool)
                for i in range(len(top_k_names)):
                    if to_remove_mask[i]:
                        continue
                    # Mark features to the right of 'i' that are highly correlated for removal
                    duplicates_in_slice = corr_matrix[i, i+1:] > 0.999
                    to_remove_mask[i+1:][duplicates_in_slice] = True
                
                # Get final indices to keep and convert back to CPU only for name lookup
                keep_indices = xp.where(~to_remove_mask)[0]
                if xp == torch:
                    keep_indices_cpu = keep_indices.cpu().numpy()
                else: # cupy
                    keep_indices_cpu = cp.asnumpy(keep_indices)

                original_count = len(top_k_names)
                final_top_k_names = [top_k_names[i] for i in keep_indices_cpu]

                if len(final_top_k_names) < original_count:
                    print(f"    Removed {original_count - len(final_top_k_names)} redundant features. Kept {len(final_top_k_names)}.")
                top_k_names = final_top_k_names

            else: # Original CPU path
                top_k_indices = [candidate_names.index(n) for n in top_k_names]
                corr_matrix = temp_phi_df[top_k_names].corr().abs()

                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                keep, seen = [], set()
                for col in top_k_names:
                    if col in seen: continue
                    keep.append(col); seen.add(col)
                    dupes = upper_tri.index[upper_tri.loc[:, col] > 0.999].tolist()
                    seen.update(dupes)
                if len(keep) < len(top_k_names):
                    print(f"    Removed {len(top_k_names) - len(keep)} redundant features. Kept {len(keep)}.")
                top_k_names = keep

        # prune candidate map to survivors ---------------------------------------------------------------------
        pruned_map = {}
        for expr, feat in candidate_features_map.items():
            if str(_canonicalize_expr(expr)) in top_k_names:
                pruned_map[_canonicalize_expr(expr)] = feat

        surviving_new_features = []
        pruned_expressions = {str(e) for e in pruned_map.keys()}
        for feat in newly_added_this_level:
            if str(_canonicalize_expr(feat.sym_expr)) in pruned_expressions:
                surviving_new_features.append(feat)

        candidate_features_map = pruned_map
        last_level_features = surviving_new_features
        print(f"    Pruned feature space to {len(candidate_features_map)} candidates.")
        if not surviving_new_features and d > 0:
            print("  No newly generated features survived this screening round. Stopping iteration."); break
        if d < depth:
            print(f"    {len(last_level_features)} features (survivors from this depth) will form the basis for depth {d+1}.")

    # finalize --------------------------------------------------------------------------------------------------
    print("\nIterative feature generation complete.")
    final_features_list = list(candidate_features_map.values())

    def _to_cpu(arr):
        if CUPY_AVAILABLE and isinstance(arr, cp.ndarray): return cp.asnumpy(arr)
        if TORCH_AVAILABLE and isinstance(arr, torch.Tensor): return arr.cpu().numpy()
        return arr

    values_dict = {str(_canonicalize_expr(feat.sym_expr)): _to_cpu(feat.values) for feat in final_features_list}
    feature_df = pd.DataFrame(values_dict, index=X.index)
    sym_map = {str(_canonicalize_expr(feat.sym_expr)): _canonicalize_expr(feat.sym_expr) for feat in final_features_list}
    return feature_df.dropna(axis=1), sym_map, clean_to_original_map