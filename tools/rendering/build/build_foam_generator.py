#!/usr/bin/env python3
"""Build the pinned headless SPlisHSPlasH FoamGenerator for the render tools.

The script pins the upstream repository and commit, applies the bundled patch
(headless build, explicit ``--seed``, velocity arrays in split VTK output, and
zero-range potential handling), and builds the ``FoamGenerator`` target with
double precision. Every input is a command-line parameter; there is no case
configuration file. Re-running against existing directories is idempotent: an
already applied patch is not applied twice, and an existing build directory is
reused unless ``--clean`` is given.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

DEFAULT_REPO_URL = "https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git"
DEFAULT_COMMIT = "f3f677140761db7637b5443beb54f19f1f835ed4"
BINARY_RELATIVE_PATH = Path("bin") / "FoamGenerator"


def default_patch_path() -> Path:
    return Path(__file__).resolve().with_name("splishsplash_foam_generator.patch")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", required=True, type=Path,
                        help="SPlisHSPlasH checkout (cloned when absent)")
    result.add_argument("--build-dir", required=True, type=Path,
                        help="Out-of-source CMake build directory")
    result.add_argument("--repo-url", default=DEFAULT_REPO_URL,
                        help="Upstream repository (default: pinned 2.18.1 mirror)")
    result.add_argument("--commit", default=DEFAULT_COMMIT,
                        help="Pinned upstream commit")
    result.add_argument("--patch", default=default_patch_path(), type=Path,
                        help="Patch applied to the pinned source")
    result.add_argument("--jobs", default=os.cpu_count() or 4, type=int)
    result.add_argument("--cmake-generator", default="Ninja")
    result.add_argument("--build-type", default="Release")
    result.add_argument("--double-precision", action=argparse.BooleanOptionalAction,
                        default=True, help="USE_DOUBLE_PRECISION=ON (use --no-double-precision otherwise)")
    result.add_argument("--cmake-policy-minimum", default="3.10")
    result.add_argument("--clean", action="store_true",
                        help="Remove the build directory before configuring")
    result.add_argument("--no-fetch", action="store_true",
                        help="Do not fetch from the remote when the pinned commit is missing")
    return result


def check_prerequisites(cmake_generator: str) -> None:
    missing = [tool for tool in ("git", "cmake", cmake_generator.lower())
               if shutil.which(tool) is None]
    if missing:
        raise ValueError("missing build prerequisites on PATH: " + ", ".join(missing) +
                         " (provide them, e.g. via `uv run --with cmake --with ninja`)")


def run(command: list[str], working_directory: Path | None = None) -> None:
    subprocess.run(command, cwd=working_directory, check=True)


def git_output(arguments: list[str], working_directory: Path) -> str:
    completed = subprocess.run(["git", *arguments], cwd=working_directory, text=True,
                               capture_output=True, check=True)
    return completed.stdout.strip()


def commit_present(source: Path, commit: str) -> bool:
    completed = subprocess.run(["git", "cat-file", "-e", commit], cwd=source,
                               capture_output=True)
    return completed.returncode == 0


def patch_state(source: Path, patch: Path) -> str:
    """Return applied, appliable, or mismatch for a patch against a checkout."""
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                             cwd=source, capture_output=True)
    forward = subprocess.run(["git", "apply", "--check", str(patch)],
                             cwd=source, capture_output=True)
    if reverse.returncode == 0 and forward.returncode != 0:
        return "applied"
    if forward.returncode == 0 and reverse.returncode != 0:
        return "appliable"
    return "mismatch"


def ensure_source(source: Path, repo_url: str, commit: str, patch: Path,
                  allow_fetch: bool) -> None:
    if not patch.is_file():
        raise ValueError(f"patch not found: {patch}")
    if not source.exists():
        run(["git", "clone", repo_url, str(source)])
    if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=source,
                      capture_output=True).returncode != 0:
        raise ValueError(f"not a git checkout: {source}")
    if not commit_present(source, commit):
        if not allow_fetch:
            raise ValueError(f"pinned commit {commit} missing and fetching is disabled")
        run(["git", "fetch", "origin"], working_directory=source)
        if not commit_present(source, commit):
            raise ValueError(f"pinned commit {commit} not found after fetching")
    if git_output(["rev-parse", "HEAD"], source) != commit:
        tracked_changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source,
            text=True, capture_output=True, check=True).stdout.strip()
        if tracked_changes:
            raise ValueError("checkout has tracked modifications; clean it before switching "
                             f"to the pinned commit {commit}")
        run(["git", "checkout", commit], working_directory=source)
    state = patch_state(source, patch)
    if state == "appliable":
        run(["git", "apply", str(patch)], working_directory=source)
    elif state != "applied":
        raise ValueError(f"patch does not match the pinned source: {patch}")


def cmake_flags(options: argparse.Namespace) -> list[str]:
    return [
        f"-DCMAKE_BUILD_TYPE={options.build_type}",
        "-DUSE_DOUBLE_PRECISION=" + ("ON" if options.double_precision else "OFF"),
        "-DFOAM_GENERATOR_ONLY=ON",
        f"-DCMAKE_POLICY_VERSION_MINIMUM={options.cmake_policy_minimum}",
    ]


def configure(options: argparse.Namespace) -> None:
    run(["cmake", "-S", str(options.source_dir), "-B", str(options.build_dir),
         "-G", options.cmake_generator, *cmake_flags(options)])


def verify_binary(source: Path) -> Path:
    binary = (source / BINARY_RELATIVE_PATH).resolve()
    if not binary.is_file():
        raise ValueError(f"FoamGenerator binary not found: {binary}")
    completed = subprocess.run([str(binary), "--help"], text=True, capture_output=True,
                               check=True)
    if "--seed" not in completed.stdout:
        raise ValueError("FoamGenerator does not support the explicit --seed patch")
    return binary


def main(arguments: list[str] | None = None) -> Path:
    options = parser().parse_args(arguments)
    options.source_dir = options.source_dir.expanduser()
    options.build_dir = options.build_dir.expanduser()
    options.patch = options.patch.expanduser()
    if options.jobs < 1:
        raise SystemExit("--jobs must be positive")
    check_prerequisites(options.cmake_generator)
    if options.clean and options.build_dir.exists():
        shutil.rmtree(options.build_dir)
    ensure_source(options.source_dir, options.repo_url, options.commit, options.patch,
                  allow_fetch=not options.no_fetch)
    # External dependencies are installed on the first configure; a second
    # configure picks them up before the FoamGenerator target is built.
    configure(options)
    run(["cmake", "--build", str(options.build_dir), "--target",
         "Ext_NeighborhoodSearch", "Ext_PBD", "--parallel", str(options.jobs)])
    configure(options)
    run(["cmake", "--build", str(options.build_dir), "--target", "FoamGenerator",
         "--parallel", str(options.jobs)])
    binary = verify_binary(options.source_dir)
    print(f"FoamGenerator ready: {binary}")
    return binary


if __name__ == "__main__":
    main()
