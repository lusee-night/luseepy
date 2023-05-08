# Configuration file for the Sphinx documentation builder.

import os
import sys

from unittest import mock

sys.path.insert(0, os.path.abspath('../..'))
os.system("apt-get upgrade openssl")

# -- Project information

project = 'luseepy'
copyright = '2022, LuSEE-Night'
author = 'M.Potekhin'

release = '0.2'
version = '0.2'


# Mock imports, because it fails to build in readthedocs
MOCK_MODULES = ["pyshtools","pyshtools.legendre"]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

import lusee

# -- General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
