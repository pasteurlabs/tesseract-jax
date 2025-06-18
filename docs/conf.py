# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re
import shutil
from pathlib import Path

from tesseract_jax import __version__

here = Path(__file__).parent.resolve()

project = "Tesseract-JAX"
copyright = "2025, Pasteur Labs"
author = "The Tesseract-JAX Team @ Pasteur Labs + OSS contributors"

# The short X.Y version
parsed_version = re.match(r"(\d+\.\d+\.\d+)", __version__)
if parsed_version:
    version = parsed_version.group(1)
else:
    version = "0.0.0"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # This is myst-parser + jupyter notebook support
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    # Copy button for code blocks
    "sphinx_copybutton",
    # OpenGraph metadata for social media sharing
    "sphinxext.opengraph",
]

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "tesseract_core": (
        "https://docs.pasteurlabs.ai/projects/tesseract-core/latest/",
        None,
    ),
}

templates_path = []
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["static"]
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
}
html_css_files = ["custom.css"]


# -- Handle Jupyter notebooks ------------------------------------------------

# Do not execute notebooks during build (just take existing output)
nb_execution_mode = "off"

# Copy example notebooks to docs/examples folder on every build
for example_dir in Path("../examples").glob("*/"):
    # Copy the example directory to the docs folder
    shutil.copytree(
        example_dir, here / "examples" / example_dir.name, dirs_exist_ok=True
    )
