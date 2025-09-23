# Contributing to DISCOVER

Thank you for your interest in contributing to DISCOVER! Contributions involving the addition of new features, bug fixes, or documentation improvements are all encouraged. The following guide will be of assistance.

### Code Structure Overview
The project is organized into several top-level modules:

-   `run_discover.py`: The main command-line interface.
-   `discover/models.py`: The core `DiscoverBase` class responsible for controlling the workflow.
-   `discover/features.py`: Oversees all feature space generation considerations and the `UFeature` class for unit awareness.
-   `discover/search.py`: Positions different search strategies (e.g., greedy, SISSO++, MIQP).
-   `discover/scoring.py`: Contains model scoring functions (SIS, CV, regression/classification scores) and GPU-accelerated kernels.
-   `discover/utils.py`: Contains plotting helpers, result savers, and formula formatters.
-   `discover/constants.py`: Defines global constants (e.g., task types).

### Adding a New Mathematical Operator

Adding a custom operator is an easy method to extend DISCOVER's capabilities. Operators are defined in `discover/features.py`.

#### Adding a Custom Unary Operator

1.  Open `discover/features.py`.
2.  Navigate to the `CUSTOM_UNARY_OP_DEFS` dictionary.
3.  Add a new entry with a new name.

The entry must be a dictionary with following keys:
-   `func`: The numerical function (e.g., `lambda x: np.tanh(x)`).
-   `sym_func`: Symphological analogue with `sympy` (i.e., `lambda s: sympy.tanh(s)`).
-   `unit_check` (optional): An optional function that is `True` if the input unit is right. For example, `lambda u: u.dimensionless` for trig functions.
-   `domain_check` (optional): An optional function that is `True` if the input values are appropriate (e.g., `lambda f: np.all(f.values > 0)` for `log`).
-   `name_func`: A function that generates the string name of the new feature (e.g., `lambda n: f"tanh({n})"`).

**Example: Adding a `tanh` operator**
```python
# In discover/features.py
CUSTOM_UNARY_OP_DEFS = {
    #. existing operators
    'tanh': {
        'func': np.tanh,
        'sym_func': lambda s: sympy.tanh(s),
'unit_check': lambda u: u.dimensionless,
        'domain_check': None,
        'name_func': lambda n: f"tanh({n})",
    }
}

#### Adding a Custom Binary Operator

1.  Open `discover/features.py`.
2.  Go to the `CUSTOM_BINARY_OP_DEFS` dictionary.
3.  Add a new entry.

The form is the same as unary operators but with two arguments.

**Example: Adding a `geometric_mean` operator**
```python
# In discover/features.py
CUSTOM_BINARY_OP_DEFS = {
    #. other operators
    'geometric_mean': {
        'func': lambda f1, f2: np.sqrt(f1 * f2),
        'sym_func': lambda s1, s2: sympy.sqrt(s1 * s2),
        'unit_op': lambda u1, u2: u1**0.5 * u2**0.5,
    }}
'unit_check': None, # Or a check for compatible units
        'domain_check': lambda f1, f2: np.all(f1.values >= 0) and np.all(f2.values >= 0),
        'op_name': "<G>"
    }

```

### How to Add a New Search Strategy

1.  Open `discover/search.py`.
2.  Define a new function, typically with `_find_best_models_` prepended to its name. As an example, study `_find_best_models_greedy`.
3.  The function must have the following signature:
    ```python
def _find_best_models_new_strategy(sisso_instance, phi_sis_df, y, D_max, task_type, max_feat_cross_corr, sample_weight, device, torch_device, **kwargs):
```
4.  The function needs to return a dictionary with keys as dimensions (e.g., `1`, `2`, `3`) and values as dictionaries of the model info for that dimension. The model info dictionary should have:
    -   `features`: List of feature names (strings).
    -   `score`: The model's training score.
-   `model`: The fitted model object (e.g., an scikit-learn model).
    -   `coef`: The model coefficients (if available).
    -   `sym_features`: List of the features as `sympy` objects.
    -   `is_parametric`: A flag (`False` for most linear models).
5.  Open `discover/models.py`, locate the `fit` method of the `DiscoverBase` class, and insert your new search strategy into the `if/elif` block that invokes the search functions.
    ```python
    # In discover/models.py in the fit method
    elif self.search_strategy == 'new_strategy':
        search_results = _find_best_models_new_strategy(**search_args)
```

### Coding Style and Conventions

-   **PEP 8:** Conform to the PEP 8 coding style.
-   **Docstrings:** Use docstrings for all modules, classes, and functions with a brief and obvious explanation of their purpose, arguments, and return values.
-   **Type Hinting:** Type hints are encouraged for improved code readability and maintainability.
-   **Clarity:** Make your code readable and clear. Use comments where the logic is not apparent.

### Submitting Changes

1.  **Fork the repository** on GitHub.
2.  **Create a new branch** for your feature or bugfix: `git checkout -b feature/my-new-operator`.
3.  **Make your changes** and commit them with a good commit message.
4.  **Push your branch** to your fork: `git push origin feature/my-new-operator`.
5.  **Open a Pull Request** against the main repository, explaining what you have changed.
