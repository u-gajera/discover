# How-To Guides

This section provides practical, task-oriented guides for users who want to leverage the advanced features of DISCOVER.

---

### How to Use DISCOVER in a Python Script or Notebook

While the `run_discover.py` script is convenient, you can get more flexibility by using the DISCOVER classes directly in Python. This allows for easier integration into larger workflows and custom analysis.

```python
import pandas as pd
from discover import DiscoverRegressor, print_descriptor_formula

# 1. Load your data
data = pd.read_csv("manuel_data.csv")
y = data["E_a"]
X = data.drop(columns=["E_a", "Name", "Delta_E", "E_Ts", "E_KRA", "A-site", "B-site", "X-site"])
X = X.dropna()
y = y[X.index]

# 2. Define configuration as a dictionary
config = {
    "depth": 3,
    "max_D": 2,
    "op_rules": [{"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "div"}, {"op": "sqrt"}],
    "search_strategy": "sisso++",
    "selection_method": "cv",
    "cv": 5,
    "random_state": 42
}

# 3. Initialize and fit the model
model = DiscoverRegressor(**config)
model.fit(X, y)

# 4. Print the best model's formula
print("--- Best Model Found ---")
model.best_model_summary()

# 5. Make predictions
predictions = model.predict(X)
print(f"\nPredictions on training data: {predictions[:5]}")

# 6. Generate and save plots
print("\nGenerating plots...")
fig_parity = model.plot_parity(X, y, save_path="parity_plot.png")
fig_cv = model.plot_cv_results(save_path="cv_scores.png")
print("Plots saved to parity_plot.png and cv_scores.png")
```

### How to Enable GPU Acceleration

If you have a compatible GPU and have installed the necessary libraries (`cupy` for NVIDIA, `pytorch` for Apple Silicon), you can significantly speed up the workflow.

In your `config.json` or Python dictionary, set the `device` parameter:

**For NVIDIA CUDA:**
```json
{
  "device": "cuda",
  "gpu_id": 0,
  "use_single_precision": true
}
```
-   `gpu_id`: The index of the GPU to use (if you have multiple).
-   `use_single_precision`: Set to `true` to use `float32`, which is often much faster on GPUs and typically sufficient for symbolic regression tasks.

**For Apple Silicon (MPS):**
```json
{
  "device": "mps"
}
```
-   Note: MPS acceleration currently works with `float32`. DISCOVER will automatically switch to single precision if you select the `mps` device.

DISCOVER will automatically use the GPU for feature generation, SIS screening, and L2 regression. If a specific operation is not GPU-supported (e.g., Huber loss), it will seamlessly fall back to the CPU for that step.

### How to Configure the Feature Space

You have fine-grained control over the features DISCOVER generates.

**1. Customizing Operators:**
The `op_rules` list determines which mathematical operations are used. The default set is comprehensive, but you can tailor it. For example, to only allow simple arithmetic and powers:

```json
"op_rules": [
  {"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "div"},
  {"op": "sq"}, {"op": "cb"}, {"op": "inv"}
]
```
See `discover/features.py` for a full list of built-in operators, including `abs_diff` and `harmonic_mean`.

**2. Excluding Features from Operators:**
You can prevent certain primary features from being used in specific operations. This is useful for enforcing physical constraints. For example, to prevent `log` from being applied to a feature named `temperature`:

```json
"op_rules": [
  {"op": "log", "exclude_features": ["temperature"]},
  {"op": "add"}
]
```

**3. Interaction-Only Features:**
Set `"interaction_only": true` to prevent unary operators (like `sqrt`, `log`, `sq`) from being applied. This forces the model to find relationships based only on combinations of *different* primary features.

### How to Choose a Search Strategy

The `search_strategy` parameter is one of the most important for balancing speed and the quality of the final model.

| Strategy      | Description                                                                  | Best For                                                        |
|---------------|------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `greedy`      | Fast. Selects the best feature at each step to add to the model.             | Quick initial analysis, very high-dimensional feature spaces.   |
| `omp`         | Orthogonal Matching Pursuit. More robust than simple greedy search.          | Regression tasks where feature orthogonality is a concern.      |
| `sisso++`     | Very fast and efficient breadth-first search using QR decomposition.         | Default choice for most regression tasks; excellent balance.    |
| `rmhc` / `sa` | Heuristic searches (Hill Climbing, Simulated Annealing). Can escape local minima. | Complex problems where greedy methods might fail.               |
| `brute_force` | Exhaustively checks every possible combination. Guarantees the optimal model.| Problems with a small number of candidate features (< 30-40).   |
| `miqp`        | Guarantees the provably optimal L0-norm model via Mixed-Integer Programming. | Regression tasks where optimality is required (needs Gurobi).   |

### How to Perform Classification

DISCOVER supports several classification tasks. To use one, simply change `task_type`.

**1. Logistic Regression or SVM:**
```json
"task_type": "classification_logreg", // or "classification_svm"
```
The model will be a linear combination of descriptors fed into a logistic or SVM classifier. The goal is to find descriptors that create a linearly separable space.

**2. Convex Hull Classification:**
```json
"task_type": "ch_classification"
```
This is a unique geometric approach. The search algorithm tries to find a feature space where the convex hulls of the different classes have minimal overlap. This is powerful for problems where classes are known to occupy distinct domains in some property space.

### How to Use Unit-Aware Feature Generation

To prevent physically nonsensical features (e.g., adding a length to a temperature), you can provide units for your primary features. This requires the `pint` library.

In your configuration, create a `primary_units` dictionary:

```json
"primary_units": {
  "E_a": "electron_volt",
  "A_valence": "dimensionless",
  "R_A": "angstrom",
  "R_B": "angstrom",
  "alpha": "degree",
  "k64_m": "dimensionless"
}
```
DISCOVER will now only generate features that are dimensionally consistent. For example, `R_A + R_B` is allowed, but `R_A + alpha` will be discarded. Functions like `log` or `exp` will only be applied to dimensionless features.

