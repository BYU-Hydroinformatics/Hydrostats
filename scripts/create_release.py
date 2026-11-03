"""Create a draft GitHub release using the version in pyproject.toml."""

import argparse
import json
import re
import shlex
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?")
GITHUB_REPOSITORY_PARTS = 2


def run(*command: str) -> str:
    """Run a command and return its standard output."""
    result = subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def project_metadata() -> tuple[str, str]:
    """Read and validate the project version and GitHub repository."""
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)["project"]

    version = project["version"]
    if not VERSION_PATTERN.fullmatch(version):
        msg = (
            f"Unsupported release version {version!r}; expected MAJOR.MINOR.PATCH "
            "with an optional aN, bN, or rcN suffix."
        )
        raise ValueError(msg)

    homepage = urlparse(project["urls"]["Homepage"])
    path_parts = homepage.path.strip("/").removesuffix(".git").split("/")
    if homepage.netloc != "github.com" or len(path_parts) != GITHUB_REPOSITORY_PARTS:
        msg = "The project Homepage must be a GitHub repository URL."
        raise ValueError(msg)

    return version, "/".join(path_parts)


def previous_release(repo: str, version: str) -> str | None:
    """Reject an existing release and return the most recent published tag."""
    pages = json.loads(
        run(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/releases?per_page=100",
        ),
    )
    releases = [release for page in pages for release in page]
    if any(release["tag_name"] == version for release in releases):
        msg = f"A release for {version} already exists."
        raise ValueError(msg)

    published = [
        release for release in releases if not release["draft"] and release["published_at"]
    ]
    published.sort(
        key=lambda release: datetime.fromisoformat(release["published_at"]),
        reverse=True,
    )
    return published[0]["tag_name"] if published else None


def check_new_tag(repo: str, version: str) -> None:
    """Reject an existing remote tag, even if it has no release."""
    refs = json.loads(run("gh", "api", f"repos/{repo}/git/matching-refs/tags/{version}"))
    if any(ref["ref"] == f"refs/tags/{version}" for ref in refs):
        msg = f"The tag {version} already exists."
        raise ValueError(msg)


def check_target(repo: str, version: str, target: str) -> None:
    """Verify the target contains the selected version and is available on GitHub."""
    target_project = tomllib.loads(run("git", "show", f"{target}:pyproject.toml"))
    target_version = target_project["project"]["version"]
    if target_version != version:
        msg = (
            f"The selected commit has version {target_version!r}, but the working "
            f"pyproject.toml has {version!r}. Commit the version change or select another ref."
        )
        raise ValueError(msg)
    run("gh", "api", f"repos/{repo}/commits/{target}", "--jq", ".sha")


def main() -> int:
    """Create a GitHub draft release, or preview it with --dry-run."""
    parser = argparse.ArgumentParser(
        description="Create a draft GitHub release from the project version.",
    )
    parser.add_argument(
        "--target",
        default="HEAD",
        help="Local Git ref or commit to release (default: HEAD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the release details without creating it.",
    )
    args = parser.parse_args()

    try:
        version, repo = project_metadata()
        target = run(
            "git", "rev-parse", "--verify", "--end-of-options", f"{args.target}^{{commit}}"
        )
        check_target(repo, version, target)
        prior_tag = previous_release(repo, version)
        check_new_tag(repo, version)

        is_prerelease = bool(re.search(r"(?:a|b|rc)\d+$", version))
        docs_channel = "latest" if is_prerelease else "stable"
        notes = (
            f"Release Notes: https://hydrostats.readthedocs.io/en/{docs_channel}/change_log.html"
        )
        if prior_tag:
            notes += (
                f"\n\n**Full Changelog**: https://github.com/{repo}/compare/{prior_tag}...{version}"
            )

        command = [
            "gh",
            "release",
            "create",
            version,
            "--repo",
            repo,
            "--title",
            version,
            "--notes",
            notes,
            "--draft",
            "--target",
            target,
        ]
        if is_prerelease:
            command.append("--prerelease")

        if args.dry_run:
            output = [
                f"Repository: {repo}",
                f"Version: {version}",
                f"Target: {target}",
                f"Prerelease: {is_prerelease}",
                f"Notes:\n{notes}",
                f"Command: {shlex.join(command)}",
            ]
            sys.stdout.write("\n".join(output) + "\n")
            return 0

        sys.stdout.write(run(*command) + "\n")
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr or "").strip() or str(error)
        else:
            detail = str(error)
        sys.stderr.write(f"Could not create release: {detail}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
