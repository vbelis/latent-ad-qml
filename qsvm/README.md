# (Quantum) Support Vector Machine for supervised anomaly detection.

## TODOs
* make it qsvm/svm agnostic. just specify whether is classical or quantum
 with an argument flag. qsvm -> model. util.initialise_model
* environment only for the (Q)SVM code.
* TODO: - json file that saves all the arguments and models hyper-parameters from `qsvm.main.py` to transfer them automatically to `qsvm.test.py` . This ensures that the seeds and training/testing data will be consistent between the two automatically.

Notes:
- ntrain/ntest/nvalid arguments refer to the total  signal+background samples. The data sets are always balanced, i.e., signal (50%) + background (50%).
- To save `sklearn.svm.SVC` and `QSVN` models joblib package is used. Serialization and de-serialization of objects is python-version sensitive.  
- The both quantum and classical SVM models (supervised) are all saved after training in a folder named trained_qsvms/.