from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def get_path(*parts):
    return ROOT.joinpath(*parts)