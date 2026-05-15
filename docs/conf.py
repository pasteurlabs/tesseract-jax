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
copyright = "2026, Pasteur Labs"
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
    # Collapsible dropdowns and other UI components
    "sphinx_design",
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

templates_path = ["_templates"]
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
html_js_files = []


# -- Handle Jupyter notebooks ------------------------------------------------

# Do not execute notebooks during build (just take existing output)
nb_execution_mode = "off"

# Copy example notebooks to docs/examples folder on every build
for example_dir in Path("../examples").glob("*/"):
    # Copy the example directory to the docs folder
    shutil.copytree(
        example_dir, here / "examples" / example_dir.name, dirs_exist_ok=True
    )


# ---------------------------------------------------------------------------
# Fetch shared navbar assets from tesseract-core/main
# ---------------------------------------------------------------------------
_CORE_RAW = "https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs"
_SHARED_FILES = {
    here / "_templates" / "page.html": f"{_CORE_RAW}/_templates/page.html",
    here / "static" / "top-nav.css": f"{_CORE_RAW}/static/top-nav.css",
}


def _fetch_shared_nav(_app):  # noqa: ANN001
    import urllib.request

    import sphinx.util.logging

    logger = sphinx.util.logging.getLogger(__name__)

    for dest, url in _SHARED_FILES.items():
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            if dest.exists():
                logger.warning(
                    "could not fetch %s, using existing %s: %s",
                    url,
                    dest.name,
                    exc,
                )
            else:
                raise RuntimeError(
                    f"Cannot fetch {url} and no local copy exists: {exc}"
                ) from exc


html_css_files.append("top-nav.css")


def setup(app):  # noqa: ANN001, ANN201, D103
    app.connect("builder-inited", _fetch_shared_nav)
