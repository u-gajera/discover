# -*- coding: utf-8 -*-
"""
This module handles all aspects of feature generation and manipulation.
It includes the unit-aware UFeature class and the core logic for building
the feature space (phi-space) through iterative application of mathematical
operators.
"""
import pandas as pd
import numpy as np
import sympy
from pathlib import Path
import warnings

from joblib import Memory

# Optional CuPy and PyTorch imports for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    class cp:
        ndarray = type(None)
        def asarray(self, arr, **kwargs): return np.asarray(arr, **kwargs)
        def asnumpy(self, arr): return arr
        def get_default_memory_pool(self):
            class DummyPool:
                def free_all_blocks(self): pass
            return DummyPool()
        def finfo(self, dtype): return np.finfo(dtype)
        def abs(self, arr): return np.abs(arr)
        def all(self, arr): return np.all(arr)
        def any(self, arr): return np.any(arr)
        def min(self, arr): return np.min(arr)
        def max(self, arr): return np.max(arr)
        def isfinite(self, arr): return np.isfinite(arr)
        def sqrt(self, arr): return np.sqrt(arr)
        def cbrt(self, arr): return np.cbrt(arr)
        def sign(self, arr): return np.sign(arr)
        def log(self, arr): return np.log(arr)
        def exp(self, arr): return np.exp(arr)
        def sin(self, arr): return np.sin(arr)
        def cos(self, arr): return np.cos(arr)

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
        def tensor(x, **kwargs): return np.array(x, **kwargs)
        @staticmethod
        def finfo(dtype):
            class DummyFinfo:
                max = np.finfo(np.float64).max
                min = np.finfo(np.float64).min
            return DummyFinfo()


# Try to import pint for unit handling
try:
    import pint
    PINT_AVAILABLE = True
    warnings.filterwarnings("ignore", module='pint')
    pint.UnitRegistry().setup_matplotlib()
except ImportError:
    PINT_AVAILABLE = False
    class pint:
        class DimensionalityError(Exception): pass
        class UnitRegistry:
            def __call__(self, unit_str): return 1.0
            def setup_matplotlib(self): pass

# =============================================================================
#  Operator Definitions (Including NEW User-Customizable Operators)
# =============================================================================

# This dictionary defines non-linear operators that include trainable parameters.
PARAMETRIC_OP_DEFS = {
    'exp-': {
        'func': lambda f, p: np.exp(-p[0] * f),
        'sym_func': lambda s, p_sym: sympy.exp(-p_sym[0] * s),
        'p_names': ['p'],
        'p_initial': [1.0],
        'p_bounds': [(1e-4, 1e4)],
        'unit_check': lambda u: u.dimensionless,
    },
    'pow': {
        'func': lambda f, p: np.power(f, p[0]),
        'sym_func': lambda s, p_sym: s**p_sym[0],
        'p_names': ['p'],
        'p_initial': [1.0],
        'p_bounds': [(-3.0, 3.0)],
        'domain_check': lambda f_uf: f_uf.is_positive,
        'unit_check': None,
    },
}

# --- NEW: USER-DEFINED CUSTOM OPERATORS ---
# Define your own custom operators here. They can be enabled by name in the config.
CUSTOM_UNARY_OP_DEFS = {
    # Example 1: A shifted logarithm, log(x+1), useful for features with zeros.
    'log1p': {
        'func': np.log1p,                               # The numpy function
        'sym_func': lambda s: sympy.log(s + 1),         # The symbolic equivalent
        'unit_check': lambda u: u.dimensionless,        # Constraint: must be dimensionless
        'domain_check': lambda f: np.all(f.values > -1),# Constraint: values must be > -1
        'name_func': lambda n: f"log1p({n})",            # How to name the new feature
    },
    # Example 2: A gaussian-like function without parameters
    'gaussian': {
        'func': lambda x: np.exp(-x**2),
        'sym_func': lambda s: sympy.exp(-s**2),
        'unit_check': lambda u: u.dimensionless,
        'domain_check': None,
        'name_func': lambda n: f"gaussian({n})",
    }
}

CUSTOM_BINARY_OP_DEFS = {
    # Example: The harmonic mean of two features
    'harmonic_mean': {
        'func': lambda f1, f2: 2 * f1 * f2 / (f1 + f2),
        'sym_func': lambda s1, s2: 2 * s1 * s2 / (s1 + s2),
        'unit_op': lambda u1, u2: u1, # Result has the same unit as the inputs
        'unit_check': lambda u1, u2: u1.dimensionality == u2.dimensionality, # Must have same units
        'domain_check': lambda f1, f2: not np.any(np.abs(f1.values + f2.values) < 1e-9), # Avoid division by zero
        'op_name': "<H>" # Custom name for the operation
    }
}


# =============================================================================
#  Feature Generation (with Unit-Awareness and Operator Control)
# =============================================================================

class UFeature:
    """A wrapper for features that includes name, values, units, symbolic expression, and base features."""
    def __init__(self, name, values, unit, ureg, sym_expr, base_features, xp=np, dtype=np.float64, torch_device=None):
        self.name = name
        self.xp = xp
        self.torch_device = torch_device
        self.ureg = ureg
        self.sym_expr = sym_expr
        self.base_features = base_features # NEW: Set of original feature names

        vals = values.magnitude if hasattr(values, 'magnitude') else values
        if xp == torch:
            torch_dtype = {np.float32: torch.float32, np.float64: torch.float64}.get(np.dtype(dtype).type)
            self.values = torch.tensor(vals, dtype=torch_dtype, device=self.torch_device)
        else:
            self.values = xp.asarray(vals, dtype=dtype)

        self.unit = ureg(unit) if isinstance(unit, str) and ureg else unit
        self.is_positive = self.xp.all(self.values > 1e-9)
        self.has_zero    = self.xp.any(self.xp.abs(self.values) < 1e-9)
        self.min_val = self.xp.min(self.values)
        self.max_val = self.xp.max(self.values)

    def _apply_binary_op(self, other, op_func, sym_op_func, op_name, unit_op=None, unit_check=None, domain_check=None):
         if self.ureg and unit_check and not unit_check(self.unit, other.unit): return None
         if domain_check and not domain_check(self, other): return None
         
         if not self.ureg:
              try:
                  new_values = op_func(self.values, other.values)
                  new_sym_expr = sym_op_func(self.sym_expr, other.sym_expr)
                  new_name = f"({self.name}{op_name}{other.name})"
                  new_base_features = self.base_features.union(other.base_features)
                  return UFeature(new_name, new_values, None, self.ureg, new_sym_expr, new_base_features, self.xp, dtype=self.values.dtype, torch_device=self.torch_device)
              except: return None
         try:
            new_unit = unit_op(self.unit, other.unit) if unit_op else op_func(self.unit, other.unit)
            new_values = op_func(self.values, other.values)
            new_sym_expr = sym_op_func(self.sym_expr, other.sym_expr)
            new_name = f"({self.name}{op_name}{other.name})"
            new_base_features = self.base_features.union(other.base_features) # NEW
            return UFeature(new_name, new_values, new_unit, self.ureg, new_sym_expr, new_base_features, self.xp, dtype=self.values.dtype, torch_device=self.torch_device)
         except (pint.DimensionalityError, ValueError, TypeError, OverflowError): return None

    def __add__(self, other): return self._apply_binary_op(other, lambda a, b: a + b, lambda a, b: a + b, '+')
    def __sub__(self, other): return self._apply_binary_op(other, lambda a, b: a - b, lambda a, b: a - b, '-')
    def __mul__(self, other): return self._apply_binary_op(other, lambda a, b: a * b, lambda a, b: a * b, '*')
    def __truediv__(self, other):
        if other.has_zero: return None
        return self._apply_binary_op(other, lambda a, b: a / b, lambda a, b: a / b, '/')

    def _apply_unary_op(self, val_op, unit_op, sym_op, name_func, unit_check=None, domain_check=None):
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
            new_base_features = self.base_features # NEW: Unary op doesn't add new base features
            return UFeature(new_name, new_values, new_unit, self.ureg, new_sym_expr, new_base_features, self.xp, dtype=self.values.dtype, torch_device=self.torch_device)
        except (ValueError, TypeError, pint.DimensionalityError, OverflowError): return None

    def power(self, p):
        if p != int(p) and not self.is_positive : return None
        if self.has_zero and p < 0: return None
        return self._apply_unary_op(lambda a: a**p, lambda u: u**p, lambda s: s**p, lambda n: f"({n})**{p}")

    def sqrt(self): return self._apply_unary_op(lambda a: self.xp.sqrt(a), lambda u: u**0.5, sympy.sqrt, lambda n: f"sqrt({n})", domain_check=lambda s: s.is_positive or s.has_zero)
    def cbrt(self): return self._apply_unary_op(lambda a: a.cbrt() if self.xp==torch else self.xp.cbrt(a), lambda u: u**(1/3.), sympy.cbrt, lambda n: f"cbrt({n})")
    def abs(self):  return self._apply_unary_op(self.xp.abs, lambda u: u, sympy.Abs, lambda n: f"Abs({n})")
    def inv(self):  return self._apply_unary_op(lambda a: 1.0/a, lambda u: 1.0/u, lambda s: 1/s, lambda n: f"1/({n})", domain_check=lambda s: not s.has_zero)
    def sign(self): return self._apply_unary_op(self.xp.sign, lambda u: u, sympy.sign, lambda n: f"sign({n})", unit_check=lambda u: u.dimensionless)
    def log(self): return self._apply_unary_op(self.xp.log, lambda u: u, sympy.log, lambda n: f"log({n})", unit_check=lambda u: u.dimensionless, domain_check=lambda s: s.is_positive)
    def exp(self): return self._apply_unary_op(self.xp.exp, lambda u: u, sympy.exp, lambda n: f"exp({n})", unit_check=lambda u: u.dimensionless)
    def exp_neg(self): return self._apply_unary_op(lambda a: self.xp.exp(-a), lambda u: u, lambda s: sympy.exp(-s), lambda n: f"exp(-{n})", unit_check=lambda u: u.dimensionless)
    def sin(self): return self._apply_unary_op(self.xp.sin, lambda u: u, sympy.sin, lambda n: f"sin({n})", unit_check=lambda u: u.dimensionless)
    def cos(self): return self._apply_unary_op(self.xp.cos, lambda u: u, sympy.cos, lambda n: f"cos({n})", unit_check=lambda u: u.dimensionless)

    @staticmethod
    def get_all_unary_op_names():
       return [
           'sqrt', 'cbrt', 'abs', 'log', 'exp', 'exp-', 'sin', 'cos',
           'inv', 'sign', 'sq', 'cb', 'pow6', 'pow-2', 'pow-3',
           'pow0.5', 'pow-0.5'
       ]
    
    def get_unary_op_func(self, op_name):
        """Returns the function for a given unary operator name."""
        # This replaces the buggy implementation and makes it easier to call operators by name
        op_map = {
           'sqrt': self.sqrt, 'cbrt': self.cbrt, 'abs': self.abs, 'log': self.log,
           'exp': self.exp, 'exp-': self.exp_neg, 'sin': self.sin, 'cos': self.cos,
           'inv': self.inv, 'sign': self.sign,
           'sq': lambda: self.power(2), 'cb': lambda: self.power(3), 'pow6': lambda: self.power(6),
           'pow-2': lambda: self.power(-2), 'pow-3': lambda: self.power(-3),
           'pow0.5': self.sqrt, 'pow-0.5': lambda: self.power(-0.5)
        }
        return op_map.get(op_name)

    @staticmethod
    def get_all_binary_op_names():
         return ['add', 'sub', 'mul', 'div']

    @staticmethod
    def get_binary_op_method_name(op_name):
        return {'add': '__add__', 'sub': '__sub__', 'mul': '__mul__', 'div': '__truediv__'}.get(op_name)


def apply_op(f, op_func, min_val, max_val):
    """Helper to apply a unary operator and check for validity."""
    try:
       new_feat = op_func()
       if new_feat and new_feat.xp.all(new_feat.xp.isfinite(new_feat.values)):
           abs_vals = new_feat.xp.abs(new_feat.values)
           if new_feat.xp.any(abs_vals < min_val) or new_feat.xp.any(abs_vals > max_val):
               return None
           if new_feat.xp.all(abs_vals < 1e-9): return None
           if new_feat.xp.max(new_feat.values) - new_feat.xp.min(new_feat.values) < 1e-9: return None
           return new_feat.sym_expr, new_feat
    except Exception: pass
    return None

def apply_binary_op(f1, f2, op_name, min_val, max_val):
    """Helper to apply a binary operator and check for validity."""
    try:
        op_method_name = UFeature.get_binary_op_method_name(op_name)
        if not op_method_name: return None
        new_feat = getattr(f1, op_method_name)(f2)

        if new_feat and new_feat.xp.all(new_feat.xp.isfinite(new_feat.values)):
            abs_vals = new_feat.xp.abs(new_feat.values)
            if new_feat.xp.any(abs_vals < min_val) or new_feat.xp.any(abs_vals > max_val):
                return None
            if new_feat.xp.all(abs_vals < 1e-9): return None
            if new_feat.xp.max(new_feat.values) - new_feat.xp.min(new_feat.values) < 1e-9: return None
            if (op_method_name == '__sub__' and f1.sym_expr == f2.sym_expr) or \
               (op_method_name == '__truediv__' and f1.sym_expr == f2.sym_expr) : return None
            return new_feat.sym_expr, new_feat
    except Exception: pass
    return None

def apply_custom_binary_op(f1, f2, op_def, min_val, max_val):
    """Helper to apply a custom binary operator."""
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
            if new_feat.xp.all(abs_vals < 1e-9): return None
            if new_feat.xp.max(new_feat.values) - new_feat.xp.min(new_feat.values) < 1e-9: return None
            return new_feat.sym_expr, new_feat
    except Exception: pass
    return None


def apply_parametric_op(feature, op_def, min_val, max_val):
    """Applies a parametric operator with its initial default parameter values."""
    try:
        p_initial = op_def['p_initial']
        if op_def.get('domain_check') and not op_def['domain_check'](feature): return None
        if feature.ureg and op_def.get('unit_check') and not op_def['unit_check'](feature.unit): return None
        
        cpu_values = feature.values
        if feature.xp != np:
            cpu_values = cp.asnumpy(feature.values) if CUPY_AVAILABLE and isinstance(feature.values, cp.ndarray) else feature.values.cpu().numpy()

        new_values = op_def['func'](cpu_values, p_initial)
        if not np.all(np.isfinite(new_values)): return None
        abs_vals = np.abs(new_values)
        if np.any(abs_vals < min_val) or np.any(abs_vals > max_val): return None
        if np.all(abs_vals < 1e-9): return None
        if np.max(new_values) - np.min(new_values) < 1e-9: return None

        p_symbols = [sympy.Symbol(p) for p in op_def['p_names']]
        new_sym_expr = op_def['sym_func'](feature.sym_expr, p_symbols).subs(zip(p_symbols, p_initial))
        new_unit = feature.unit
        if op_def.get('unit_check'): new_unit = 'dimensionless'
        elif op_def['p_names'] == ['p']: new_unit = feature.unit ** p_initial[0]
        op_name = list(PARAMETRIC_OP_DEFS.keys())[list(PARAMETRIC_OP_DEFS.values()).index(op_def)]
        new_name = f"{op_name}(p={p_initial})({feature.name})"
        
        new_feat = UFeature(new_name, new_values, new_unit, feature.ureg, new_sym_expr, feature.base_features, feature.xp, dtype=feature.values.dtype, torch_device=feature.torch_device)
        return new_feat.sym_expr, new_feat
    except Exception:
        return None

_cached_feature_generator = None
def get_cached_generator(workdir):
    """Initializes and returns a joblib.Memory object for caching feature generation."""
    global _cached_feature_generator
    if workdir and _cached_feature_generator is None:
        cache_path = Path(workdir) / "feature_cache"
        cache_path.mkdir(exist_ok=True, parents=True)
        print(f"[Cache] Feature generation cache enabled at: {cache_path}")
        memory = Memory(cache_path, verbose=0)
        _cached_feature_generator = memory.cache(generate_features_with_units_core)
    elif _cached_feature_generator:
        return _cached_feature_generator
    return generate_features_with_units_core

def generate_features_with_units(X, primary_units, depth, n_jobs, use_cache, workdir,
                                 op_rules, parametric_ops, min_abs_feat_val, max_abs_feat_val,
                                 interaction_only, xp, dtype, torch_device):
    """
    Wrapper for the core feature generation function that handles caching.
    """
    if use_cache:
        generator = get_cached_generator(workdir)
        # The core function expects positional arguments based on the DataFrame's components.
        X_cols = X.columns
        X_values = tuple(map(tuple, X.values))
        
        primary_units_tuple = tuple(sorted(primary_units.items()))
        # Convert op_rules to a hashable format for caching
        op_rules_tuple = tuple(tuple(sorted(d.items())) for d in op_rules)
        parametric_ops_tuple = tuple(sorted(parametric_ops))

        return generator(X_cols, X_values, primary_units_tuple, depth, n_jobs,
                         op_rules_tuple, parametric_ops_tuple, min_abs_feat_val, max_abs_feat_val,
                         interaction_only, xp, dtype, workdir, torch_device)
    else:
        # The direct call also passes components, not the whole DataFrame.
        return generate_features_with_units_core(X.columns, X.values, tuple(sorted(primary_units.items())),
                                                 depth, n_jobs, op_rules, tuple(sorted(parametric_ops)),
                                                 min_abs_feat_val, max_abs_feat_val,
                                                 interaction_only, xp, dtype, workdir, torch_device)

def _canonicalize_expr(expr):
    """Converts a sympy expression to a canonical form for robust duplicate checking."""
    try:
        return sympy.simplify(expr)
    except (TypeError, ValueError, Exception):
        return expr

def generate_features_with_units_core(X_cols, X_values, primary_units_tuple, depth, n_jobs,
                                      op_rules_tuple, parametric_ops_tuple, min_abs_feat_val, max_abs_feat_val,
                                      interaction_only, xp, dtype, workdir, torch_device):
    """
    The core engine for generating the feature space (phi-space).
    This version includes advanced operator rules and interaction-only mode.
    """
    X = pd.DataFrame(X_values, columns=X_cols)
    primary_units = dict(primary_units_tuple)
    # Convert op_rules back from tuple if it was cached
    op_rules = [dict(rule) for rule in op_rules_tuple] if isinstance(op_rules_tuple[0], tuple) else op_rules_tuple
    parametric_ops = list(parametric_ops_tuple)

    ureg = None
    if PINT_AVAILABLE:
        ureg = pint.UnitRegistry()
        if 'electron_volt' not in ureg: ureg.define('electron_volt = 1.602176634e-19 * joule = eV')
    elif primary_units:
        warnings.warn("Pint not available. Generating features without unit checks.")

    clean_names = [f'f{i}' for i in range(len(X.columns))]
    primary_symbols = [sympy.Symbol(s, real=True) for s in clean_names]
    original_symbols = [sympy.Symbol(c, real=True) for c in X.columns]
    clean_to_original_map = dict(zip(primary_symbols, original_symbols))
    X.columns = clean_names
    
    initial_features = []
    if not primary_units: primary_units = {}

    for i, name in enumerate(X.columns):
        original_name = str(original_symbols[i])
        unit = primary_units.get(original_name, 'dimensionless' if ureg else None)
        if original_name not in primary_units and ureg:
            warnings.warn(f"No unit defined for '{original_name}'. Treating as dimensionless.")
        # NEW: base_features set is initialized with the original feature name
        initial_features.append(UFeature(original_name, X[name].values, unit, ureg, primary_symbols[i], {original_name}, xp, dtype, torch_device))

    device_name = "CPU"
    if xp == cp: device_name = "NVIDIA GPU"
    if xp == torch and torch_device is not None: device_name = "Apple/MPS GPU"
    print(f"\nStarting {'unit-aware' if ureg else ''} feature generation on {device_name} to depth {depth}...")
    print(f"  Using {len(op_rules)} operator rules. Interaction-only mode: {'ON' if interaction_only else 'OFF'}")
    if parametric_ops:
        print(f"  Including in-tree parametric operators: {parametric_ops}")

    canonical_features = {_canonicalize_expr(feat.sym_expr): feat for feat in initial_features}
    last_level_features = initial_features

    # Get all available operator names/definitions
    BUILTIN_UNARY_NAMES = UFeature.get_all_unary_op_names()
    BUILTIN_BINARY_NAMES = UFeature.get_all_binary_op_names()

    for d in range(1, depth + 1):
        print(f"  Depth {d}: Combining {len(last_level_features)} features from previous level...")
        newly_added_this_level = []

        # --- Unary and Parametric Operations ---
        if not interaction_only:
            for f in last_level_features:
                for rule in op_rules:
                    op_name = rule['op']
                    
                    # NEW: Check operator constraints against the feature's base components
                    if 'exclude_features' in rule:
                        if not f.base_features.isdisjoint(set(rule['exclude_features'])):
                            continue # Skip this operator for this feature
                    
                    res = None
                    # Try built-in, custom, and parametric unary operators
                    if op_name in BUILTIN_UNARY_NAMES:
                        op_func = f.get_unary_op_func(op_name)
                        if op_func: res = apply_op(f, op_func, min_abs_feat_val, max_abs_feat_val)
                    elif op_name in CUSTOM_UNARY_OP_DEFS:
                        op_def = CUSTOM_UNARY_OP_DEFS[op_name]
                        op_func = lambda: f._apply_unary_op(op_def['func'], op_def.get('unit_op', lambda u: u), op_def['sym_func'], op_def['name_func'], op_def.get('unit_check'), op_def.get('domain_check'))
                        res = apply_op(f, op_func, min_abs_feat_val, max_abs_feat_val)
                    elif op_name in parametric_ops and op_name in PARAMETRIC_OP_DEFS:
                         res = apply_parametric_op(f, PARAMETRIC_OP_DEFS[op_name], min_abs_feat_val, max_abs_feat_val)

                    if res:
                        expr, feat = res
                        simplified_expr = _canonicalize_expr(expr)
                        if not simplified_expr.is_Number and simplified_expr not in canonical_features:
                            canonical_features[simplified_expr] = feat
                            newly_added_this_level.append(feat)

        # --- Binary Operations ---
        all_features_list = list(canonical_features.values())
        for f1 in last_level_features:
            for f2 in all_features_list:
                for rule in op_rules:
                    op_name = rule['op']
                    
                    # NEW: Check operator constraints
                    if 'exclude_features' in rule:
                        excluded = set(rule['exclude_features'])
                        if not f1.base_features.isdisjoint(excluded) or not f2.base_features.isdisjoint(excluded):
                            continue
                    
                    res = None
                    if op_name in BUILTIN_BINARY_NAMES:
                         if op_name in ['add', 'mul'] and id(f1) < id(f2): continue
                         res = apply_binary_op(f1, f2, op_name, min_abs_feat_val, max_abs_feat_val)
                    elif op_name in CUSTOM_BINARY_OP_DEFS:
                         if id(f1) < id(f2): continue
                         res = apply_custom_binary_op(f1, f2, CUSTOM_BINARY_OP_DEFS[op_name], min_abs_feat_val, max_abs_feat_val)
                    
                    if res:
                        expr, feat = res
                        simplified_expr = _canonicalize_expr(expr)
                        if not simplified_expr.is_Number and simplified_expr not in canonical_features:
                            canonical_features[simplified_expr] = feat
                            newly_added_this_level.append(feat)

        if not newly_added_this_level:
            print(f"  No new unique features generated at depth {d}. Stopping.")
            break
        last_level_features = newly_added_this_level
        print(f"  Depth {d}: Added {len(newly_added_this_level)} new unique features. Total unique: {len(canonical_features)}")

    all_final_features = list(canonical_features.values())
    
    def to_cpu(arr):
        if CUPY_AVAILABLE and isinstance(arr, cp.ndarray): return cp.asnumpy(arr)
        if TORCH_AVAILABLE and isinstance(arr, torch.Tensor): return arr.cpu().numpy()
        return arr

    values_dict = {str(feat.sym_expr): to_cpu(feat.values) for feat in all_final_features}
    feature_df = pd.DataFrame(values_dict)

    print(f"Φ-Space generation complete. Total valid, unique features: {len(all_final_features)}")
    sym_map = {str(feat.sym_expr): feat.sym_expr for feat in all_final_features}
    
    return feature_df.dropna(axis=1), sym_map, clean_to_original_map