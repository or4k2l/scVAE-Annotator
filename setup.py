"""Setup script for scVAE-Annotator."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="scvae-annotator",
    version="0.1.0",
    author="scVAE-Annotator Team",
    description="A deep learning-based tool for automated cell type annotation in single-cell RNA-seq data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/or4k2l/scVAE-Annotator",
    project_urls={
        "Bug Reports": "https://github.com/or4k2l/scVAE-Annotator/issues",
        "Source": "https://github.com/or4k2l/scVAE-Annotator",
        "Documentation": "https://github.com/or4k2l/scVAE-Annotator/blob/main/docs/README.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-timeout>=2.1.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pylint>=2.15.0",
            "isort>=5.10.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
            "coverage[toml]>=6.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
            "myst-parser>=0.18.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="single-cell RNA-seq annotation VAE deep-learning bioinformatics",
    entry_points={
        "console_scripts": [
            "scvae-annotate=scvae_annotator.cli:main",
        ],
    },
)
