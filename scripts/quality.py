"""Cross-platform local quality gate matching GitHub Actions."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run(python, "-m", "ruff", "check", "src", "tests", "scripts")
    run(python, "-m", "ruff", "format", "--check", "src", "tests", "scripts")
    run(python, "-m", "pytest")
    run(python, "scripts/validate_artifacts.py")


if __name__ == "__main__":
    main()
