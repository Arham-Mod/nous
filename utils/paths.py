from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def project_path(*parts):
    return ROOT_DIR.joinpath(*parts)