# Anomaly detection with quantum machine learning for particle physics data

This repository has the code we developed for the paper _"Quantum anomaly detection in the latent space of proton collision events at the LHC"_ [[1]](https://arxiv.org/abs/2301.10780). In this work, we investigate unsupervised quantum machine learning algorithms for anomaly detection tasks in particle physics data. 

The `qad` package associated with this work was created for reproducibility of the results and ease-of-use in future studies.
<p align="center">
<img src="https://github.com/vbelis/latent-ad-qml/blob/docs-reformat/docs/Pipeline_QML.png?raw=true" alt="Sublime's custom image"/>
Quantum-classical pipeline for detecting new-physics events in proton collisions at the LHC. Taken from [1].
</p>

## Documentation 
The documentation for can be consulted in the readthedocs page: **TODO**

## How to install
The package can be installed with Python's `pip` package manager. We recommend installing the dependencies and the package within a dedicated environment. 
**TODO**: try directly `pip install .zip` after the repo is made public. 
```
git clone https://github.com/vbelis/latent-ad-qml.git
cd latent-ad-qml
pip install .
```

## Usage
Examples on how to run the code and use the `qad` to reproduce results and plots from the paper can be found in the [scripts](https://github.com/vbelis/latent-ad-qml/tree/main/scripts).


# References
[1] K. A. Woźniak<sup>\*</sup>, V. Belis<sup>\*</sup>, E. Puljak<sup>\*</sup>, P. Barkoutsos, G. Dissertori, M. Grossi, M. Pierini, F. Reiter, I. Tavernelli, S. Vallecorsa , _Quantum anomaly detection in the latent space of proton collision events at the LHC_, [arXiv:2301.10780](https://arxiv.org/abs/2301.10780). 
