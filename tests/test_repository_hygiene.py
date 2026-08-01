import subprocess
from pathlib import Path


def test_no_generated_junk_is_tracked():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    forbidden_parts = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "tmp", "htmlcov"}
    assert not [path for path in tracked if forbidden_parts.intersection(Path(path).parts)]
    assert not [path for path in tracked if path.endswith((".log", ".pyc", ".coverage"))]
