from setuptools import setup, find_packages

setup(
    name='pysisso',
    version='0.1.0',
    description='A Python Implementation of Sure Independence Screening and Sparsifying Operator',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)