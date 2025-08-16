# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'TMMax'
copyright = '2025, Bahrem S. Danis'
author = 'Bahrem S. Danis'

release = '0.1.14'
version = '0.1.14'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_favicon',   
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


html_static_path = ["_static"]

favicons = [
    {"rel": "icon", "sizes": "16x16", "href": "favicon-16x16.png"},
    {"rel": "icon", "sizes": "32x32", "href": "favicon-32x32.png"},
    {"rel": "apple-touch-icon", "sizes": "180x180", "href": "apple-touch-icon.png"},
]