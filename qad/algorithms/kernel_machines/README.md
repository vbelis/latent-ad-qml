[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# Unsupervised quantum kernel machine for anomaly detection

Notes:
- To save `sklearn.svm.SVC` and `QSVM` models joblib package is used. Serialization and de-serialization of objects is python-version sensitive.  
- In `test.py` the QSVM and SVM models are directly loaded from the file. For the QSVM
this means that the QuantumInstance and the backend are the same as the ones during training.
- The both quantum and classical SVM models (supervised) are all saved after training in a folder named trained_qsvms/.
