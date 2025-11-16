from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_src_root() -> Path:
    return get_project_root() / "src"
