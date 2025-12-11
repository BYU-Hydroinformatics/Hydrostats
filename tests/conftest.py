from pathlib import Path

import matplotlib as mpl

# Use non-interactive backend for image tests
mpl.use("Agg")


import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def files_for_tests(tests_dir: Path) -> Path:
    return tests_dir / "Files_for_tests"


@pytest.fixture(scope="session")
def comparison_files(tests_dir: Path) -> Path:
    return tests_dir / "Comparison_Files"


@pytest.fixture(scope="session")
def baseline_plots(tests_dir: Path) -> Path:
    return tests_dir / "baseline_images" / "plot_tests"


@pytest.fixture(scope="session")
def merged_df(files_for_tests: Path) -> pd.DataFrame:
    return pd.read_pickle(files_for_tests / "merged_df.pkl")
