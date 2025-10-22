"""
Setup script for LLM training project.
"""

from setuptools import setup, find_packages

setup(
    name="LLM",
    version="0.1.0",
    description="A personal LLM training project",
    author="Quinn Lue",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "tokenizers",
        "tqdm",
        "dlx @ file://../dlx",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    }
)
