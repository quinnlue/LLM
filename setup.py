"""
Setup script for GPT-1 training project.
"""

from setuptools import setup, find_packages

setup(
    name="gpt1",
    version="0.1.0",
    description="A personal GPT-1 training project",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "tokenizers",
        "tqdm",
        "dlx",
        # Add other dependencies as needed
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    }
)
