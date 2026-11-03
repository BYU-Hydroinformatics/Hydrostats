import json
import subprocess

import pytest
from typer.testing import CliRunner

from scripts import create_release

REPO = "BYU-Hydroinformatics/Hydrostats"
TARGET = "a" * 40
USAGE_ERROR_EXIT_CODE = 2


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
    return fake


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize(
    ("version", "channel", "prerelease"),
    [("2.0.0", "stable", False), ("2.0.0rc1", "latest", True)],
)
def test_creates_draft_with_conventional_notes(
    commands: FakeCommands,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    channel: str,
    prerelease: bool,
) -> None:
    commands.target_version = version
    monkeypatch.setattr(create_release, "project_metadata", lambda: (version, REPO))

    result = runner.invoke(create_release.app, [])

    assert result.exit_code == 0, result.output
    assert result.stdout == "https://github.com/example/draft\n"
    assert result.stderr == ""
    assert commands.calls[0] == (
        "git",
        "rev-parse",
        "--verify",
        "--end-of-options",
        "HEAD^{commit}",
    )

    command = next(call for call in commands.calls if call[:3] == ("gh", "release", "create"))
    assert command[3] == version
    assert command[command.index("--title") + 1] == version
    assert command[command.index("--target") + 1] == TARGET
    assert "--draft" in command
    assert ("--prerelease" in command) is prerelease
    notes = command[command.index("--notes") + 1]
    assert f"https://hydrostats.readthedocs.io/en/{channel}/change_log.html" in notes
    assert f"https://github.com/{REPO}/compare/1.0.0...{version}" in notes


@pytest.mark.parametrize(
    ("target_args", "expected_ref"),
    [
        ([], "HEAD"),
        (["--target", "release"], "release"),
        (["--target=release"], "release"),
        (["--target=-release"], "-release"),
    ],
)
def test_dry_run_does_not_create_release(
    commands: FakeCommands, runner: CliRunner, target_args: list[str], expected_ref: str
) -> None:
    result = runner.invoke(create_release.app, [*target_args, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert (
        "git",
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{expected_ref}^{{commit}}",
    ) in commands.calls
    assert not any(call[:3] == ("gh", "release", "create") for call in commands.calls)
    assert f"Repository: {REPO}" in result.stdout
    assert "Version: 2.0.0rc1" in result.stdout
    assert f"Target: {TARGET}" in result.stdout
    assert "Prerelease: True" in result.stdout
    assert "Command: gh release create 2.0.0rc1" in result.stdout


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help_lists_options_without_running_commands(
    commands: FakeCommands, runner: CliRunner, option: str
) -> None:
    result = runner.invoke(create_release.app, [option])

    assert result.exit_code == 0, result.output
    assert "--target" in result.stdout
    assert "--dry-run" in result.stdout
    assert "HEAD" in result.stdout
    assert commands.calls == []


@pytest.mark.parametrize("args", [["--target"], ["--unknown"], ["--no-dry-run"]])
def test_invalid_arguments_return_usage_error(
    commands: FakeCommands, runner: CliRunner, args: list[str]
) -> None:
    result = runner.invoke(create_release.app, args)

    assert result.exit_code == USAGE_ERROR_EXIT_CODE, result.output
    assert "Error" in result.stderr
    assert commands.calls == []


def test_rejects_mismatched_target_version(commands: FakeCommands, runner: CliRunner) -> None:
    commands.target_version = "1.0.0"

    result = runner.invoke(create_release.app, [])

    assert result.exit_code == 1
    assert "selected commit has version '1.0.0'" in result.stderr
    assert not any(call[0] == "gh" for call in commands.calls)


def test_rejects_existing_draft(commands: FakeCommands, runner: CliRunner) -> None:
    commands.pages[0].append(
        {"tag_name": "2.0.0rc1", "draft": True, "published_at": None},
    )

    result = runner.invoke(create_release.app, [])

    assert result.exit_code == 1
    assert "A release for 2.0.0rc1 already exists" in result.stderr
    assert not any(call[:3] == ("gh", "release", "create") for call in commands.calls)


def test_rejects_existing_tag_without_release(commands: FakeCommands, runner: CliRunner) -> None:
    commands.refs = [{"ref": "refs/tags/2.0.0rc1"}]

    result = runner.invoke(create_release.app, [])

    assert result.exit_code == 1
    assert "The tag 2.0.0rc1 already exists" in result.stderr
    assert not any(call[:3] == ("gh", "release", "create") for call in commands.calls)


def test_first_release_has_no_comparison(commands: FakeCommands, runner: CliRunner) -> None:
    commands.pages = [[]]
    commands.refs = [{"ref": "refs/tags/2.0.0rc10"}]

    result = runner.invoke(create_release.app, [])

    assert result.exit_code == 0, result.output
    command = next(call for call in commands.calls if call[:3] == ("gh", "release", "create"))
    assert "**Full Changelog**" not in command[command.index("--notes") + 1]


@pytest.mark.usefixtures("commands")
@pytest.mark.parametrize(
    ("stderr", "expected"),
    [("", "returned non-zero exit status 1"), ("gh: denied\n", "gh: denied")],
)
def test_command_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, stderr: str, expected: str
) -> None:
    def fail(*command: str) -> str:
        raise subprocess.CalledProcessError(1, command, stderr=stderr)

    monkeypatch.setattr(create_release, "run", fail)

    result = runner.invoke(create_release.app, [])

    assert result.exit_code == 1
    assert result.stderr.startswith("Could not create release: ")
    assert expected in result.stderr
