import json
import subprocess
import sys

import pytest

from scripts import create_release

REPO = "BYU-Hydroinformatics/Hydrostats"
TARGET = "a" * 40


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.target_version = "2.0.0rc1"
        self.pages: list[list[dict[str, str | bool | None]]] = [
            [
                {"tag_name": "1.0.0rc1", "draft": False, "published_at": "2025-12-22T16:03:07Z"},
                {"tag_name": "1.0.0", "draft": False, "published_at": "2025-12-22T18:26:53Z"},
                {"tag_name": "unpublished", "draft": True, "published_at": None},
            ],
        ]
        self.refs: list[dict[str, str]] = []

    def __call__(self, *command: str) -> str:
        self.calls.append(command)
        if command[:2] == ("git", "rev-parse"):
            return TARGET
        if command[:2] == ("git", "show"):
            return f'[project]\nversion = "{self.target_version}"'
        if command[:2] == ("gh", "api"):
            if command[-1].endswith("releases?per_page=100"):
                return json.dumps(self.pages)
            if "/git/matching-refs/tags/" in command[-1]:
                return json.dumps(self.refs)
            if f"repos/{REPO}/commits/{TARGET}" in command:
                return TARGET
        if command[:3] == ("gh", "release", "create"):
            return "https://github.com/example/draft"
        raise AssertionError(f"Unexpected command: {command!r}")


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> FakeCommands:
    fake = FakeCommands()
    monkeypatch.setattr(create_release, "run", fake)
    monkeypatch.setattr(create_release, "project_metadata", lambda: ("2.0.0rc1", REPO))
    monkeypatch.setattr(sys, "argv", ["create_release.py"])
    return fake


@pytest.mark.parametrize(
    ("version", "channel", "prerelease"),
    [("2.0.0", "stable", False), ("2.0.0rc1", "latest", True)],
)
def test_creates_draft_with_conventional_notes(
    commands: FakeCommands,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    channel: str,
    prerelease: bool,
) -> None:
    commands.target_version = version
    monkeypatch.setattr(create_release, "project_metadata", lambda: (version, REPO))

    assert create_release.main() == 0

    command = next(call for call in commands.calls if call[:3] == ("gh", "release", "create"))
    assert command[3] == version
    assert command[command.index("--title") + 1] == version
    assert command[command.index("--target") + 1] == TARGET
    assert "--draft" in command
    assert ("--prerelease" in command) is prerelease
    notes = command[command.index("--notes") + 1]
    assert f"https://hydrostats.readthedocs.io/en/{channel}/change_log.html" in notes
    assert f"https://github.com/{REPO}/compare/1.0.0...{version}" in notes


def test_dry_run_does_not_create_release(
    commands: FakeCommands, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["create_release.py", "--target", "release", "--dry-run"])

    assert create_release.main() == 0
    assert (
        "git",
        "rev-parse",
        "--verify",
        "--end-of-options",
        "release^{commit}",
    ) in commands.calls
    assert not any(call[:3] == ("gh", "release", "create") for call in commands.calls)
    assert "Prerelease: True" in capsys.readouterr().out


def test_rejects_mismatched_target_version(
    commands: FakeCommands, capsys: pytest.CaptureFixture[str]
) -> None:
    commands.target_version = "1.0.0"

    assert create_release.main() == 1
    assert "selected commit has version '1.0.0'" in capsys.readouterr().err
    assert not any(call[0] == "gh" for call in commands.calls)


def test_rejects_existing_draft(commands: FakeCommands, capsys: pytest.CaptureFixture[str]) -> None:
    commands.pages[0].append(
        {"tag_name": "2.0.0rc1", "draft": True, "published_at": None},
    )

    assert create_release.main() == 1
    assert "A release for 2.0.0rc1 already exists" in capsys.readouterr().err
    assert not any(call[:3] == ("gh", "release", "create") for call in commands.calls)


def test_rejects_existing_tag_without_release(
    commands: FakeCommands, capsys: pytest.CaptureFixture[str]
) -> None:
    commands.refs = [{"ref": "refs/tags/2.0.0rc1"}]

    assert create_release.main() == 1
    assert "The tag 2.0.0rc1 already exists" in capsys.readouterr().err
    assert not any(call[:3] == ("gh", "release", "create") for call in commands.calls)


def test_first_release_has_no_comparison(commands: FakeCommands) -> None:
    commands.pages = [[]]
    commands.refs = [{"ref": "refs/tags/2.0.0rc10"}]

    assert create_release.main() == 0
    command = next(call for call in commands.calls if call[:3] == ("gh", "release", "create"))
    assert "**Full Changelog**" not in command[command.index("--notes") + 1]


@pytest.mark.usefixtures("commands")
def test_command_failure_without_stderr_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*command: str) -> str:
        raise subprocess.CalledProcessError(1, command, stderr="")

    monkeypatch.setattr(create_release, "run", fail)

    assert create_release.main() == 1
    assert "returned non-zero exit status 1" in capsys.readouterr().err
