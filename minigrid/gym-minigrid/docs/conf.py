from __future__ import annotations

import os
import sys

import minigrid

project = "MiniGrid"
copyright = "2023 Farama Foundation"
author = "Farama Foundation"


release = minigrid.__version__

sys.path.insert(0, os.path.abspath("../.."))


extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_github_changelog",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


templates_path = ["_templates"]


exclude_patterns = []


napoleon_use_ivar = True
napoleon_use_admonition_for_references = True

napoleon_custom_sections = [("Returns", "params_style")]


html_theme = "furo"
html_title = "MiniGrid Documentation"
html_baseurl = "https://minigrid.farama.org/"
html_copy_source = False
html_favicon = "_static/img/minigrid-favicon.png"
html_theme_options = {
    "light_logo": "img/minigrid.svg",
    "dark_logo": "img/minigrid-white.svg",
    "image": "img/minigrid-github.png",
    "description": "Minigrid contains simple and easily configurable grid world environments to conduct Reinforcement Learning research. This library was previously known as gym-minigrid.",
    "gtag": "G-FBXJQQLXKD",
    "versioning": True,
    "source_repository": "https://github.com/Farama-Foundation/Minigrid/",
    "source_branch": "master",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
html_css_files = []


sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
