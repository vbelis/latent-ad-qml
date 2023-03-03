from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="qad",
    version="1.0.4",
    description="Quantum anomaly detection in the latent space of proton collision events",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://latent-ad-qml.readthedocs.io/en/latest/",
    author="Vasilis Belis, Ema Puljak, Kinga Anna Wozniak",
    packages=find_packages(exclude=["scripts"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "h5py>=3.6.0",
        "joblib>=1.2.0",
        "matplotlib>=3.5",
        "mplhep>=0.3.26",
        "mplhep-data==0.0.3",
        "numpy>=1.21",
        "pandas>=1.4.0",
        "scikit-learn==1.1.1",
        "scipy>=1.9",
        "qibo==0.1.10",
        "qiskit==0.36.2",
        "qiskit-aer>=0.10.4",
        "qiskit-ibmq-provider==0.19.1",
        "qiskit-ignis==0.7.1",
        "qiskit-machine-learning==0.4.0",
        "qiskit-terra==0.20.2",
        "tensorflow>=2.6",
        "pylatexenc==2.10",
        "triple_e @ https://github.com/vbelis/triple_e/archive/master.zip"
    ],
    extras_require={
        "docs": [
            "sphinx>=3.0",
            "sphinx-autoapi",
            "numpy>=1.21",
            "pandas>=1.4.0"
        ]
    }
)
