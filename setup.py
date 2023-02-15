from setuptools import setup, find_packages

setup(
    name='qad',
    version='1.0',
    description='Quantum anomaly detection in the latent space of proton collision events',
    author='Vasilis Belis, Ema Puljak, Kinga Anna Wozniak',
    packages=find_packages(exclude=['scripts']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)