import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="discover-symbolic-regression",
    version="0.1.0",
    author="Uday Gajera",
    description="Data-Informed Symbolic Combination " \
                "of Operators for Variable Equation Regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    
    # Classifiers help users find your project on PyPI
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    
    python_requires='>=3.8',
    
    # Core dependencies
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "sympy",
        "joblib",
        "matplotlib",
    ],
    
    extras_require={
        "plotting": ["seaborn"],
        "units": ["pint"],
        "miqp": ["gurobipy"],
        # Note: The user must install the correct CuPy version for their CUDA toolkit.
        # e.g., 'cupy-cuda11x' for CUDA 11.x or 'cupy-cuda12x' for CUDA 12.x.
        "gpu": ["torch", "cupy"], 
        "all": ["seaborn", "pint", "gurobipy", "torch", "cupy"],
    }
)