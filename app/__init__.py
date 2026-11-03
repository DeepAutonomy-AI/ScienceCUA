"""Evidence-grounded PDF/HTML preprocessing and knowledge export."""

from importlib.metadata import version
from pathlib import Path
import tomllib


def _project_version() -> str:
    source_metadata = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if source_metadata.is_file():
        with source_metadata.open("rb") as stream:
            return tomllib.load(stream)["project"]["version"]
    return version("resource-to-skill-dsh")


__version__ = _project_version()

