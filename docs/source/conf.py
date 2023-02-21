# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
#sys.path.append(os.path.abspath('../..'))
print(sys.path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'qad'
copyright = '2023, Vasilis Belis, Ema Puljak, Kinga Anna Wozniak'
author = 'Vasilis Belis, Ema Puljak, Kinga Anna Wozniak'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx"
]
master_doc = "contents"
html_additional_pages = {'index': 'index.html'}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_member_order = 'bysource'
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
html_show_sourcelink = True
add_module_names = False

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'tensorflow': ('https://www.tensorflow.org/', None),
    'qiskit': ('https://qiskit.org/', None),
    'qiskit_machine_learning': ('https://github.com/Qiskit/qiskit-machine-learning', None),
    'typing': ('https://docs.python.org/3/library/typing.html', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/', None)
}
