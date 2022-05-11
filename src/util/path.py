import os

from pathlib import Path


def get_project_root():
    return str(Path(__file__).parent.parent.parent)


def relative_path(path):
    return str(os.path.relpath(path, get_project_root()))
