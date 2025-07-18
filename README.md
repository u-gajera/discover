Here is your content properly formatted in **Markdown**:

# pySISSO: A Python Implementation of Sure Independence Screening and Sparsifying Operator

**pySISSO** is a modern, Python-native implementation of the Sure Independence Screening and Sparsifying Operator (SISSO) method—a powerful machine learning framework for discovering predictive, interpretable models and descriptors from large feature spaces. This implementation is designed for ease of use, extensibility, and performance, incorporating modern best practices and algorithmic enhancements.

This tool is ideal for researchers and data scientists in materials science, chemistry, and other scientific domains who need to find physically meaningful relationships in their data.

![Parity Plot Example](https://i.imgur.com/g8iVv4v.png)

---

## Features

* **Iterative Feature Generation:** Efficiently builds a high-quality feature space by iteratively generating and screening features, avoiding the combinatorial explosion of traditional methods.
* **Multiple Search Strategies:**

  * **Greedy Search:** A fast, iterative approach.
  * **Brute-Force:** Exhaustively finds the best model for low-dimensional problems.
  * **SISSO++:** A highly efficient, breadth-first search using QR decomposition.
  * **Orthogonal Matching Pursuit (OMP):** A robust greedy method for regression.
  * **MIQP (Gurobi):** Provably finds the optimal L0-norm model (requires Gurobi license).
* **Diverse Machine Learning Tasks:**

  * **Regression:** Linear (Ridge), Robust (Huber), and Multi-Task models.
  * **Classification:** Logistic Regression, Support Vector Machines (SVM), and Convex Hull-based classifiers.
* **GPU Acceleration:** Leverages NVIDIA (CUDA) and Apple Silicon (MPS) GPUs for significant speedups in feature screening and model fitting.
* **Advanced Analytics & Visualization:** Includes a suite of tools for model interpretation, such as feature importance, partial dependence plots, and a dedicated script for generating publication-quality parity plots.
* **Unit-Aware Feature Engineering:** Uses the `pint` library to prevent the creation of physically nonsensical features by enforcing unit consistency.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

A `requirements.txt` file is ideal, but you can install the core packages manually:

```bash
pip install pandas numpy sympy scikit-learn matplotlib seaborn joblib
```

### 4. Optional Installations

* **For GPU Acceleration (NVIDIA):**

  ```bash
  # Replace XX with your CUDA version (e.g., 118 for CUDA 11.8)
  pip install cupy-cudaXX
  ```

* **For MIQP Search:**

  ```bash
  pip install gurobipy
  ```

  *Note: Gurobi is a commercial solver and requires a license.*

* **For Unit-Awareness:**

  ```bash
  pip install pint
  ```

---

## Quick Start: Running an Analysis

Running an analysis is straightforward. You modify a single JSON configuration file and then execute the main script.

### Step 1: Prepare Your Data

Ensure your data is in a single **CSV file**. The file should contain all primary features and the target property you want to predict in distinct columns.

### Step 2: Edit the Configuration File

Open `config.json`. This file controls every aspect of the SISSO run. For a quick start, you only need to change these key parameters:

* `"data_file"`: Path to your CSV data file (e.g., `"my_data.csv"`).
* `"property_key"`: Column name containing the target property (e.g., `"Target_U (eV)"`).
* `"workdir"`: Directory where all results will be saved (e.g., `"my_first_run"`).
* `"task_type"`: Type of machine learning task:

  * `"regression"` (predicting a continuous value)
  * `"classification_logreg"` (predicting categories)
* `"max_D"`: Maximum dimension (number of features) for final models. A good starting point is `3`.

#### Example `config.json`

```json
{
  "data_file": "SISSO_Sample_Dataset.csv",
  "property_key": "Target_U (eV)",
  "workdir": "sisso_output",
  "task_type": "regression",
  "max_D": 3,
  "op_rules": [
    {"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "div"},
    {"op": "sqrt"}, {"op": "sq"}
  ],
  "search_strategy": "greedy",
  "selection_method": "cv",
  "cv": 5
}
```