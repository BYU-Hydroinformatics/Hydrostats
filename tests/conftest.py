import pickle
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest


def _mpl_baseline_dir(tests_dir: Path) -> Path:
    major_str, minor_str = mpl.__version__.split(".", maxsplit=2)[:2]
    major_minor = (int(major_str), int(minor_str))
    base_dir = tests_dir / "baseline_images"
    if major_minor >= (3, 11):
        return base_dir / "mpl_3_11"
    return base_dir


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
    return _mpl_baseline_dir(tests_dir)


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("mpl_baseline_path") is None:
        tests_dir = Path(__file__).resolve().parent
        config.option.mpl_baseline_path = str(_mpl_baseline_dir(tests_dir))


@pytest.fixture(scope="session")
def merged_df(files_for_tests: Path) -> pd.DataFrame:
    pickle_path = files_for_tests / "merged_df.pkl"

    try:
        df = pd.read_pickle(pickle_path)  # noqa: S301
    except pickle.UnpicklingError:
        # Git LFS pointer files are plain text and cannot be unpickled.
        lines = pickle_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        first_line = lines[0] if lines else ""
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            pytest.fail(
                "Missing Git LFS test data: install git-lfs and run 'git lfs pull' "
                "to fetch tests/Files_for_tests/merged_df.pkl."
            )
        raise

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The merged_df.pkl file did not contain a valid DataFrame.")
    return df
