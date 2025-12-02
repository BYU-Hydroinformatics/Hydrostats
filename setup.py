from setuptools import setup
from pathlib import Path

setup(
    name="hydrostats",
    packages=["hydrostats"],
    version="0.78",
    description=(
        "Tools for use in comparison studies, specifically for use in the field of hydrology"
    ),
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Wade Roberts",
    author_email="waderoberts123@gmail.com",
    url="https://github.com/waderoberts123/hydrostats",
    keywords=[
        "hydrology",
        "error",
        "metrics",
        "comparison",
        "statistics",
        "forecast",
        "observed",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    license="MIT",
    install_requires=[
        "numpy",
        "numba",
        "pandas",
        "matplotlib",
        "scipy",
        "HydroErr",
    ],
)
