Anomaly detection with quantum machine learning for particle physics data
*************************************************************************

This repository has the code we developed for the paper **Quantum anomaly detection in the latent space of proton collision events at the LHC** `[1] <https://arxiv.org/abs/2301.10780>`. In this work, we investigate unsupervised quantum machine learning algorithms for anomaly detection tasks in particle physics data. 

The `qad` package associated with this work was created for reproducibility of the results and ease-of-use in future studies.
.. image:: https://github.com/vbelis/latent-ad-qml/blob/main/docs/Pipeline_QML.png
    :width: 800px

The figure above, taken from `[1] <https://arxiv.org/abs/2301.10780>`_, depicts the _quantum\-classical pipeline for detecting (anomalous) new-physics events in proton collisions at the LHC. Our strategy, implemented in `qad`, combines a data compression scheme with unsupervised quantum machine learning models to assist in scientific discovery at high energy physics experiments.

Documentation
=============
The documentation for is available on `readthedocs <https://latent-ad-qml.readthedocs.io/en/latest/>`_

How to install
==============
The package can be installed with Python's ``pip`` package manager. We recommend installing the dependencies and the package within a dedicated environment. 
You can directly install ``qad`` by running:

.. code-block:: bash

    pip install https://github.com/vbelis/latent-ad-qml/archive/main.zip

or by first cloning the repo locally and then installing the package:
.. code-block:: bash

    git clone https://github.com/vbelis/latent-ad-qml.git
    cd latent-ad-qml
    pip install .

Usage
=====
Examples on how to run the code and use `qad` to reproduce results and plots from the paper can be found in the `scripts <https://github.com/vbelis/latent-ad-qml/tree/main/scripts>`_.


References
==========
**[1]** K. A. Woźniak<sup>\*</sup>, V. Belis<sup>\*</sup>, E. Puljak<sup>\*</sup>, P. Barkoutsos, G. Dissertori, M. Grossi, M. Pierini, F. Reiter, I. Tavernelli, S. Vallecorsa , _Quantum anomaly detection in the latent space of proton collision events at the LHC_, `arXiv:2301.10780 https://arxiv.org/abs/2301.10780`.



Dimensionality Reduction
------------------------
.. toctree::
   :maxdepth: 1
   :caption: Dimensionality Reduction

   autoencoder

Algorithms
----------
.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   kernel_machines
   kmedians
   kmeans

Analysis
--------
.. toctree::
   :maxdepth: 1
   :caption: Analysis

   analysis