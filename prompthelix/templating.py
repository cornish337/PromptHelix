"""Utility for loading Jinja2 templates used by the UI."""

from pathlib import Path
from fastapi.templating import Jinja2Templates


# Compute the absolute path to the "templates" directory relative to this file
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Initialize the Jinja2Templates instance with the absolute directory path so
# it works regardless of the current working directory.
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
