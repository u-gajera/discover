# DISCOVER: A Supplementary Material / Mathematical Guide

## The Core Problem: $L_0$-Regularized Regression

All search methods in DISCOVER are designed to find an approximate or exact solution to the same fundamental problem. Given a large set of $M$ candidate symbolic features ${\Phi} = \{\Phi_1, \Phi_2, \dots, \Phi_M\}$ and a target property vector $\mathbf{y}$, the goal is to find a linear model with at most $D$ features that minimizes the prediction error. This is the $L_0$-norm regularized least-squares problem:

$$
\min_{{\beta}} \left\| \mathbf{y} - {\Phi} {\beta} \right\|_2^2 \quad \text{subject to} \quad \left\| {\beta} \right\|_0 \le D
$$

where $\|{\beta}\|_0$ counts the number of nonzero elements in the coefficient vector ${\beta}$. This problem is NP-hard, necessitating the use of various exact heuristic, approximate, or specialized algorithms.

## Search Algorithms (The Sparsifying Operator)

The following methods are implemented in `search.py` to select the optimal subset of $D$ features from the screened candidates.

### Brute-Force Search (`_find_best_models_brute_force`)

**Methodology**
This is the most straightforward and exhaustive approach. It systematically evaluates every possible combination of $D$ features from the pool of $M$ candidates.

1.  For a given dimension $D$, generate all $\binom{M}{D}$ unique combinations of features.
2.  For each combination, form a feature matrix $X_D$.
3.  Fit an Ordinary Least Squares (OLS) regression model to find the coefficients ${\beta}$:
    $$
    {\beta} = X_D^\dagger \mathbf{y} \quad \text{(pseudo-inverse in case $X_D$ is rank-deficient)}
    $$
4.  Calculate the error of this model, typically the Root Mean Squared Error (RMSE):
    $$
    \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\frac{1}{n} \|\mathbf{y} - X_D{\beta}\|_2^2}
    $$
5.  The combination that yields the minimum RMSE is selected as the best model for dimension $D$.

**Application in Symbolic Regression**
This method guarantees finding the globally optimal set of $D$ symbolic features from the selected screened pool. However, its computational cost, $O(\binom{M}{D})$, makes it practical only for very small $M$ and $D$.

### Greedy Search (`_find_best_models_greedy`)

**Methodology**
This is a forward selection algorithm that builds the model one feature at a time. It is computationally efficient, but may not find the global optimum.

1.  **Dimension 1:** Find the single best feature $\Phi_1^*$ by fitting $M$ separate 1D models. The feature with the lowest error is selected:
    $$
    \Phi_1^* = \underset{\Phi_j \in {\Phi}}{\text{argmin}} \left( \min_{\beta_0, \beta_j} \|\mathbf{y} - (\beta_0 + \beta_j \Phi_j)\|_2^2 \right)
    $$
2.  **Dimension 2:** Keep $\Phi_1^*$ fixed. Search through all remaining $M-1$ features to find the second feature $\Phi_2^*$ that, when combined with $\Phi_1^*$, yields the best 2D model:
    $$
    \Phi_2^* = \underset{\Phi_j \in {\Phi} \setminus \{\Phi_1^*\}}{\text{argmin}} \left( \min_{\beta_0, \beta_1, \beta_j} \|\mathbf{y} - (\beta_0 + \beta_1 \Phi_1^* + \beta_j \Phi_j)\|_2^2 \right)
    $$
3.  **Dimension D:** Continue this process, adding one feature at each step that provides the greatest improvement to the existing model, until a $D$-dimensional model is constructed.

**Application in Symbolic Regression**
The greedy search quickly identifies a good, but not necessarily optimal, combination of symbolic features. In each step, it selects the symbolic expression that best explains the variance *not already explained* by the previously selected features.

### Orthogonal Matching Pursuit (OMP) (`_find_best_models_omp`)

**Methodology**
OMP is a more robust version of the greedy algorithm. Instead of just adding the next best feature, it recalculates the coefficients for all selected features at each step.

1.  Initialize the residual $\mathbf{r}_0 = \mathbf{y}$ and the selected feature set $\mathcal{S}_0 = \emptyset$.
2.  For $k=1, \dots, D$:
    * Find the feature $\Phi_k^*$ from the remaining pool that is most correlated with the current residual $\mathbf{r}_{k-1}$:
        $$
        \Phi_k^* = \underset{\Phi_j \notin \mathcal{S}_{k-1}}{\text{argmax}} \left| \langle \mathbf{r}_{k-1}, \Phi_j \rangle \right|
        $$
    * Add the selected feature to the active set: $\mathcal{S}_k = \mathcal{S}_{k-1} \cup \{\Phi_k^*\}$.
    * Solve a least-squares problem to find the new coefficients ${\beta}_k$ using all features currently in the active set $\mathcal{S}_k$:
        $$
        {\beta}_k = {\Phi}_{\mathcal{S}_k}^\dagger \mathbf{y} \quad \text{(least-squares solution / pseudo-inverse)}
        $$
    * Update the residual for the next iteration:
        $$
        \mathbf{r}_k = \mathbf{y} - {\Phi}_{\mathcal{S}_k} {\beta}_k
        $$

**Application in Symbolic Regression**
OMP provides a more stable path to a solution than a simple greedy search. By re-fitting the entire model at each step, it better accounts for correlations between the selected symbolic features, often leading to a more physically meaningful final equation.

### SISSO++ (Breadth-First QR Search) (`_find_best_models_sisso_pp`)

**Methodology**
This is a highly efficient breadth-first search algorithm that avoids the combinatorial explosion of brute-force by using linear algebra updates. It keeps a "beam" of the best-performing models at each dimension.

1.  **Dimension 1:** Evaluate all 1D models and retain the top $N_{\text{beam}}$ models with the lowest RSS.
2.  **Dimension 2:** For each of the $N_{\text{beam}}$ models from D=1, try adding every other available feature.
3.  **The QR Update:** The key to its efficiency is using QR decomposition. If we have the decomposition for a feature set $X_D = Q_D R_D$, the RSS is easily calculated:
    $$
    \text{RSS}_D = \|\mathbf{y}\|_2^2 - \|Q_D^T \mathbf{y}\|_2^2
    $$
    To find the RSS for a new model with an added feature $\mathbf{x}_{\text{new}}$, we don't refit. Instead, we compute the component of $\mathbf{x}_{\text{new}}$ orthogonal to the space spanned by $Q_D$:
    $$
    \mathbf{w}_{\text{new}} = \mathbf{x}_{\text{new}} - Q_D Q_D^T \mathbf{x}_{\text{new}}
    $$
    The reduction in RSS is then calculated directly:
    $$
    \Delta \text{RSS} = \frac{(\mathbf{r}_D^T \mathbf{w}_{\text{new}})^2}{\|\mathbf{w}_{\text{new}}\|_2^2}
    $$
    where $\mathbf{r}_D = \mathbf{y} - Q_D Q_D^T \mathbf{y}$ is the residual.
4.  A new beam of the top $N_{\text{beam}}$ 2D models is created, and the process repeats for $D=3, \dots, D_{\text{max}}$.

**Application in Symbolic Regression**
SISSO++ explores a much wider range of feature combinations than a simple greedy search without incurring the cost of brute-force. It is highly effective at finding high-quality, low-dimensional symbolic models that might be missed by a purely sequential greedy approach.

### Random Mutation Hill Climbing (RMHC) & Simulated Annealing (SA)

**Methodology**
These are stochastic heuristic search algorithms that explore the solution space by making random changes to a candidate model.

* **RMHC (`_find_best_models_rmhc`):**
    1.  Start with a good initial $D$-dimensional model (e.g., from a greedy search).
    2.  In each iteration, randomly swap one feature in the current model with one feature from the outside pool.
    3.  If the new model has a lower error (is "uphill"), accept the change.
    4.  If the new model is worse, reject the change and keep the old model.
    5.  Repeat for many iterations. Multiple restarts from the best-so-far solution help avoid local optima.

* **SA (`_find_best_models_sa`):**
    SA is similar to RMHC, but with a crucial difference: it can accept "downhill" moves (worse solutions) to escape local optima. The probability of accepting a worse solution with error increase $\Delta E$ is given by the Metropolis criterion:
    $$
    P(\text{accept}) = e^{-\frac{\Delta E}{T}}
    $$
    where $T$ is a "temperature" parameter that starts high (allowing many bad moves) and is gradually decreased (the "annealing schedule"), making the search converge towards a good solution.

**Application in Symbolic Regression**
These methods are excellent for refining a model found by a faster method like a greedy search. They can escape the local optima that greedy methods are prone to, potentially swapping out a feature for another that, while worse on its own, enables a much better overall model when combined with the other $D-1$ features.

### Mixed-Integer Quadratic Programming (MIQP) (`_find_best_models_miqp`)

**Methodology**
This method provides a mathematically rigorous, exact solution to the $L_0$ problem. It recasts the problem into a format that can be solved by specialized solvers like Gurobi.

1.  **Variables:** Define two sets of variables:
    * Continuous variables $\beta_j$ for the coefficients.
    * Binary variables $z_j \in \{0, 1\}$, where $z_j = 1$ if feature $\Phi_j$ is included in the model, and $0$ otherwise.
2.  **Objective Function:** Minimize the RSS, which is a quadratic function of ${\beta}$:
    $$
    \min_{{\beta}, \mathbf{z}} \quad (\mathbf{y} - {\Phi}{\beta})^T (\mathbf{y} - {\Phi}{\beta})  \quad \equiv \quad  \min_{{\beta}, \mathbf{z}} \quad {\beta}^T ({\Phi}^T{\Phi}) {\beta} - 2(\mathbf{y}^T{\Phi}){\beta} + \mathbf{y}^T \mathbf{y}
    $$
3.  **Constraints:**
    * The **Cardinality Constraint** directly enforces the $L_0$ norm:
        $$
        \sum_{j=1}^{M} z_j = D
        $$
    * The **Big-M Constraints** link the binary and continuous variables. They ensure that if a feature is not selected ($z_j=0$), its coefficient must also be zero ($\beta_j=0$):
        $$
        -M \cdot z_j \le \beta_j \le M \cdot z_j \quad \text{for } j=1, \dots, M
        $$
        where $M$ is a sufficiently large constant (an upper bound on any possible coefficient value).

**Application in Symbolic Regression**
When applicable (for regression with $L_2$ loss) and computationally feasible, MIQP is the gold standard. It guarantees that the selected combination of $D$ symbolic features is the provably optimal one from the entire candidate pool, leaving no doubt that a better linear combination of $D$ features exists.

### Geometric Greedy Search (`_find_best_models_ch_greedy`)

**Methodology**
This is a specialized greedy search for the `ch_classification` task. The scoring function is not RMSE but a geometric measure of class separability.

1.  At each dimension $D$, the goal is to add a new feature that minimizes the overlap between the convex hulls of the different classes in the $D$-dimensional descriptor space.
2.  The overlap is estimated using **Monte Carlo integration**:
    * A large number of random points are sampled within the bounding box of the data.
    * For each point, determine how many class hulls it falls inside.
    * The overlap score is the fraction of points inside any hull that are also inside more than one hull:
        $$
        \text{Overlap Fraction} = \frac{\text{Points in } \ge 2 \text{ hulls}}{\text{Points in } \ge 1 \text{ hull}}
        $$
3.  The search proceeds greedily, selecting the feature at each step that results in the lowest overlap fraction.

**Application in Symbolic Regression**
This method finds a set of symbolic features that transform the original data into a new space where the different classes are maximally separable by geometric boundaries (convex hulls). The resulting "formula" is not a single equation for a property, but a set of coordinate transformations (the descriptors) that define this optimal classification space.

## Post-Search: Non-Linear Refinement (`_refine_model_with_nlopt`)

After a linear model has been identified by one of the search methods, this optional step can refine it by introducing non-linear parameters.

* **Methodology:** It takes a discovered linear model, such as $y \approx \beta_0 + \beta_1 \Phi_1 + \beta_2 \Phi_2$, and tests if replacing a descriptor $\Phi_i$ with a parameterized version, like $e^{-p \Phi_i}$ or $\Phi_i^p$, can improve the fit.
* It uses a numerical optimization algorithm (L-BFGS-B) to simultaneously find the optimal values for the linear coefficients ($\beta_i$) and the new non-linear parameters ($p$). The objective function remains the RMSE.
* **Application in Symbolic Regression:** This allows DISCOVER to find even more complex and accurate formulas that are not purely linear combinations of the base descriptors, such as discovering optimal exponents or decay constants within the final equation.