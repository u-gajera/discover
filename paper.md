---
title: 'DISCOVER: A Physics-Informed, GPU-Accelerated Symbolic Regression Framework'
tags:
  - Python
  - symbolic regression
  - physics
  - materials science
  - GPU
authors:
  - name: Udaykumar Gajera
    affiliation: "1, 2"
  - name: Mohsen Sotoudeh
    affiliation: "3, 4"
  - name: Kanchan Sarkar
    affiliation: 2
  - name: Axel Groß
    affiliation: "2, 3"
affiliations:
 - name: Department of Physics and CSMB, Humboldt-Universität zu Berlin, Berlin, Germany
   index: 1
 - name: Institute of Theoretical Chemistry, Ulm University, Oberberghof 7, 89081 Ulm, Germany
   index: 2
 - name: Helmholtz Institute Ulm (HIU), Electrochemical Energy Storage, 89081 Ulm, Germany
   index: 3
 - name: Karlsruhe Institute of Technology (KIT), P.O. Box 3640, D-76021 Karlsruhe, Germany
   index: 4
date: 13 January 2026
bibliography: references.bib
---

# Summary

Symbolic Regression (SR) enables the discovery of interpretable mathematical relationships from experimental and simulation data. Although established approaches such as Sure Independence Screening and Sparsifying Operator (SISSO) have successfully identified low-dimensional descriptors within large feature spaces [@ouyang2018sisso], many existing SR tools integrate poorly with modern Python workflows, offer limited control over the symbolic search space, or struggle with the computational demands of large-scale studies. This paper introduces DISCOVER (Data-Informed Symbolic Combination of Operators for Variable Equation Regression), an open-source symbolic regression package developed to address these challenges through a modular, physics-motivated design. DISCOVER allows users to guide the symbolic search using domain knowledge, constrain the feature space explicitly, and take advantage of GPU acceleration to improve computational efficiency, enabling reproducible and scalable SR workflows. The software is intended for applications in computational physics, computational chemistry, and materials science, where interpretability, physical consistency, and execution time are especially important, and it complements general-purpose SR frameworks by emphasizing the discovery of physically meaningful models [@wang2025symantic].

# Statement of Need

Symbolic regression is widely used in scientific domains where interpretability and physical insight are essential, including physics, chemistry, and materials science. While many SR methods can recover analytical expressions from data [@udrescu2020aifeynman], practical adoption is often limited by several factors: insufficient integration with Python-based scientific workflows, limited mechanisms for incorporating *a priori* physical knowledge, and high computational cost when exploring large symbolic search spaces. These challenges make it difficult for researchers to apply SR methods efficiently and reproducibly in real-world scientific studies.

Existing tools such as SISSO provide powerful, deterministic strategies for identifying sparse descriptors but are not designed to offer fine-grained, user-defined control over the symbolic search or to leverage modern hardware acceleration as a core feature [@ouyang2018sisso; @purcell2023recentadvancessissomethod]. Conversely, more flexible or physics-informed SR approaches (PiSR) may require complex customization or lack scalable performance [@sun2022physics]. As a result, researchers often face trade-offs between interpretability, usability, and computational efficiency.

Recent symbolic regression tools have demonstrated impressive capabilities in recovering analytical expressions from data. For example, AI Feynman [@udrescu2020aifeynman] leverages symbolic manipulation and neural-guided search to rediscover known physical laws, while extensions of the SISSO framework, such as SISSO++ [@purcell2023recentadvancessissomethod], continue to advance large-scale descriptor discovery through efficient sparsity-driven screening. These methods represent important progress in the field; however, they often prioritize either fully automated discovery or highly specialized workflows, and may offer limited flexibility for incorporating fine-grained physical constraints, modern Python integration, or hardware acceleration as first-class features.

DISCOVER addresses this gap by providing a Python-native symbolic regression framework that explicitly supports physics-informed constraints and GPU-accelerated computation. By allowing users to define constraints on operators, feature combinations, and physical consistency through a configuration-based interface, DISCOVER lowers the barrier to incorporating domain knowledge into SR workflows. Its design supports reproducible experimentation, efficient exploration of constrained search spaces, and seamless integration into existing scientific Python ecosystems.

# Software Description

DISCOVER is an open-source symbolic regression package designed for the guided discovery of interpretable mathematical expressions. The software generates candidate symbolic expressions from user-provided features and operator libraries, evaluates them against target data, and identifies parsimonious models that balance accuracy and simplicity. The search process is iterative and incorporates pruning strategies informed by user-defined physical constraints. To support sparse model discovery, DISCOVER provides access to multiple sparsifying search strategies, including heuristic, optimization-based, and stochastic approaches such as Orthogonal Matching Pursuit (OMP) [@omp], Mixed-Integer Quadratic Programming (MIQP) [@miqp], and Simulated Annealing [@sa].

The software architecture is modular and Python-native, enabling straightforward integration with common scientific libraries. Computationally intensive operations such as feature generation and model evaluation are parallelized and executed on hardware accelerators when available. DISCOVER supports GPU acceleration via CUDA on NVIDIA GPUs and Metal Performance Shaders (MPS) on Apple Silicon devices, allowing efficient execution across a wide range of modern computing platforms. This hardware-aware design enables scalable symbolic regression workflows on both high-performance computing systems and local development environments.

![Overview of the DISCOVER workflow, illustrating iterative feature generation, physics-informed screening, and sparse model selection.](discover_workflow.png){#fig:workflow width=10cm}

## Core Optimization Objective

All search strategies implemented in DISCOVER are designed to approximate or solve a common underlying optimization problem. Given a set of $M$ candidate symbolic features $\boldsymbol{\Phi} = \{\Phi_1, \Phi_2, \ldots, \Phi_M\}$ and a target property vector $\mathbf{y}$, the objective is to identify a sparse linear combination of features that accurately models the data. This problem can be expressed as an $L_0$-regularized least-squares regression:

$$
\min_{\boldsymbol{\beta}} \; \left\| \mathbf{y} - \boldsymbol{\Phi} \boldsymbol{\beta} \right\|_2^2
\quad \text{subject to} \quad
\left\| \boldsymbol{\beta} \right\|_0 \le D,
$$

where $\boldsymbol{\beta}$ is the coefficient vector and $\|\boldsymbol{\beta}\|_0$ denotes the number of nonzero entries, enforcing a maximum descriptor dimensionality $D$. This formulation is common in sparse symbolic regression and descriptor discovery and is known to be NP-hard. As a result, DISCOVER offers multiple heuristic, approximate, and specialized search strategies to explore this objective efficiently under user-defined physical and computational constraints.

## Physics-Informed Constraints

A central design goal of DISCOVER is to facilitate the explicit incorporation of domain knowledge into the symbolic regression process. Physical constraints are specified through a configuration-based interface and applied during expression generation and evaluation. Dimensional consistency is enforced through integration with the `pint` unit-handling library, enabling unit-aware symbolic operations and validation of candidate expressions. By tracking physical units throughout the search process, DISCOVER can exclude dimensionally invalid expressions early, reducing the effective search space and promoting the discovery of physically meaningful and interpretable models.

# Design Philosophy and Constraints

A core design goal of DISCOVER is to enable direct incorporation of domain expertise into the symbolic regression process. Rather than relying solely on automated sparsity or heuristic search, DISCOVER allows users to specify constraints via a configuration file without modifying source code. Supported constraints include enforcement of dimensional consistency [@Tenachi_2023], restrictions on allowed operators or expression complexity, and user-defined rules governing variable combinations and functional forms [@bladek2021shape].

These constraints reduce the effective search space, improve interpretability, and help ensure that discovered expressions are physically meaningful. This approach is particularly useful in scientific domains where prior knowledge is well established and model plausibility is as important as predictive accuracy [@keren2025framework].

# Scope and Use Cases

DISCOVER is intended for scientific applications where symbolic regression is used as a tool for model discovery rather than purely predictive performance. Typical use cases include identifying low-dimensional descriptors for physical or chemical properties, such as crystal structure stability [@10.1063/5.0088177] or ion mobility in energy storage materials [@SOTOUDEH2024101494]. The software is especially suited to computational physics, computational chemistry, and materials science workflows that benefit from Python integration and hardware-accelerated computation, spanning from battery cathode discovery [@ziheng49309].

# Limitations

The effectiveness of DISCOVER depends on the quality of the input features and the appropriateness of user-defined constraints. Overly restrictive constraints may exclude valid expressions, while insufficient constraints can lead to large search spaces with increased computational cost. Although GPU acceleration improves performance for many workloads, DISCOVER is not optimized for fully unconstrained searches over extremely large feature spaces compared to specialized low-level implementations such as SISSO [@ouyang2018sisso]. Ongoing development focuses on expanded operator libraries, improved benchmarking, and scalability enhancements.

# Statement of AI Assistance

During the preparation of this work, the authors used large language models to assist in refactoring the source code. Specifically, AI tools were utilized to remove redundant functions, generate explanatory comments for complex logic, and standardize function naming conventions (e.g., renaming legacy short-form functions like `r_dis()` to the more descriptive `run_discover()`).

# Acknowledgements

U.G. acknowledges primary support from the NFDI consortium FAIRmat, funded by the Deutsche Forschungsgemeinschaft (DFG) under project 460197019. Furthermore, this work contributes to research performed at CELEST (Center for Electrochemical Energy Storage Ulm--Karlsruhe) and the Dr. Barbara Mez-Starck Foundation, funded by the DFG under Project ID 390874152 (POLiS Cluster of Excellence).