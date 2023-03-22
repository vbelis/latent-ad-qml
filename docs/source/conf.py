# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("../..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qad"
copyright = "2023, Vasilis Belis, Ema Puljak, Kinga Anna Wozniak"
author = "Vasilis Belis, Ema Puljak, Kinga Anna Wozniak"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.linkcode"
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

# master_doc = "index"
# html_additional_pages = {'index': 'index.html'}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_member_order = "bysource"
# nbsphinx_allow_errors = True
# nbsphinx_execute = 'never'
html_show_sourcelink = True
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_context = {
    "display_github": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "tensorflow": ("https://www.tensorflow.org/", None),
    "qiskit": ("https://qiskit.org/", None),
    "qiskit_machine_learning": (
        "https://github.com/Qiskit/qiskit-machine-learning",
        None,
    ),
    "typing": ("https://docs.python.org/3/library/typing.html", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/", None),
}

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/vbelis/latent-ad-qml/blob/main/%s.py" % filename
            