# Quantum and classical Support Vector Machine for supervised anomaly detection.

## TODOs

Notes:
- ntrain/ntest/nvalid arguments refer to the total  signal+background samples. The data sets are always balanced, i.e., signal (50%) + background (50%).
- To save `sklearn.svm.SVC` and `QSVM` models joblib package is used. Serialization and de-serialization of objects is python-version sensitive.  
- In `test.py` the QSVM and SVM models are directly loaded from the file. For the QSVM
this means that the QuantumInstance and the backend are the same as the ones during training.
The current implementation doesn't allow the possibility to change the configuration of the 
backend and the simulaters for inference. 
- The both quantum and classical SVM models (supervised) are all saved after training in a folder named trained_qsvms/.