from setuptools import setup, find_packages

setup(
    name='qad',
    version='1.0',
    description='Quantum anomaly detection in the latent space of proton collision events',
    author='Vasilis Belis, Ema Puljak, Kinga Anna Wozniak',
    packages=find_packages(exclude=['scripts']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        'h5py>=3.6.0',
        'joblib>=1.2.0',
        'matplotlib>=3.5',
        'mplhep>=0.3.26',
        'mplhep-data==0.0.3',
        'numpy>=1.21',
        'pandas>=1.4.0',
        'scikit-learn==1.1.1',
        'scipy>=1.9',
        'qibo==0.1.10',
        'qiskit==0.36.2',
        'qiskit-aer==0.10.4',
        'qiskit-ibmq-provider==0.19.1',
        'qiskit-ignis==0.7.1',
        'qiskit-machine-learning==0.4.0',
        'qiskit-terra==0.20.2',
        'triple_e @ https://github.com/vbelis/triple_e/archive/master.zip#egg=triple_e-0.1.3',
        'tensorflow>=2.6',
        'pylatexenc==2.10'
    ],
    extra_requires={
        'docs': [
            'sphinx>=3.0',
        ]
    }
)
