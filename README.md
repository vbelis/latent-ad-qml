# Anomaly detection with quantum machine learning for particle physics data

[![Email: vasilis](https://img.shields.io/badge/email-vasileios.belis%40cern.ch-blue?style=flat-square&logo=minutemailer)](mailto:vbelis@phys.ethz.ch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)
[![Python: version](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![License: version](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](https://github.com/QML-HEP/ae_qml/blob/main/LICENSE)

This repository has the code we developed for the paper _"Quantum anomaly detection in the latent space of proton collision events at the LHC"_ [[1]](https://arxiv.org/abs/2301.10780). In this work, we investigate unsupervised quantum machine learning algorithms for anomaly detection tasks in particle physics data. 

The `qad` package associated with this work was created for reproducibility of the results and ease-of-use in future studies.
<p align="center">
<img src="https://github.com/vbelis/latent-ad-qml/blob/docs-reformat/docs/Pipeline_QML.png?raw=true" alt="Sublime's custom image"/>
</p>

The figure above, taken from [[1]](https://arxiv.org/abs/2301.10780), depicts the _quantum\-classical pipeline_ for detecting (anomalous) new-physics events in proton collisions at the LHC. Our strategy, implemented in `qad`, combines a data compression scheme with unsupervised quantum machine learning models to assist in scientific discovery at high energy physics experiments.

## Documentation 
The documentation for can be consulted in the readthedocs [page](https://latent-ad-qml.readthedocs.io/en/latest/).
## How to install
The package can be installed with Python's `pip` package manager. We recommend installing the dependencies and the package within a dedicated environment. 
You can directly install `qad` by running:

```
pip install https://github.com/vbelis/latent-ad-qml/archive/main.zip
```
or by first cloning the repo locally and then installing the package:
```bash
git clone https://github.com/vbelis/latent-ad-qml.git
cd latent-ad-qml
pip install .
```

## Usage
Examples on how to run the code and use `qad` to reproduce results and plots from the paper can be found in the [scripts](https://github.com/vbelis/latent-ad-qml/tree/main/scripts).


# References
[1] K. A. Woźniak<sup>\*</sup>, V. Belis<sup>\*</sup>, E. Puljak<sup>\*</sup>, P. Barkoutsos, G. Dissertori, M. Grossi, M. Pierini, F. Reiter, I. Tavernelli, S. Vallecorsa , _Quantum anomaly detection in the latent space of proton collision events at the LHC_, [arXiv:2301.10780](https://arxiv.org/abs/2301.10780). 
