# Module that defines the main object Quantum Support Vector Machine. 
# Based on the kernel machine sklearn.svm.SVC implementation.

from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend
from qiskit.providers.ibmq import IBMQBackend
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np
from time import perf_counter
from typing import Union

import feature_map_circuits as fm
import util
from terminal_enhancer import tcols

class QSVM(SVC):
    """
    Quantum Support Vector Machine (QSVM) class. The SVM optimisation 
    objective function is optimised on a classical device, using convex 
    optimisation libraries utilised by sklearn. The quantum part is the 
    kernel.

    Attributes:
        Same as SVC. TODO
        _kernel_matrix_train: The kernel matrix computed during training/
        _train_data: The training data of the model.
        Note: Saving the last two variables introduces more state to the object,
        however reduces the computation time by half, since the number of quantum
        circuits that need to be simulated is reduced in half (dublicates), 
        when we want to compute accuracy and scores during testing.
    """
    def __init__(self, hpars: dict):
        """
        Args:
            hpars: Hyperparameters of the model and configuration parameters
                   for the training.
        """
        super().__init__(kernel="precomputed", C=hpars["c_param"])
        
        self._nqubits = hpars["nqubits"]
        exec("self._feature_map = fm." + hpars["feature_map"]
             + "(nqubits=self._nqubits)")

        self._quantum_instance, self._backend = util.configure_quantum_instance(
            ibmq_api_config=hpars["ibmq_api_config"],
            run_type=hpars["run_type"],
            backend_name=hpars["backend_name"],
            **hpars["config"],
        )
        self._quantum_kernel = QuantumKernel(
            self._feature_map,
            quantum_instance=self._quantum_instance,
        )
        self._kernel_matrix_train = None
        self._train_data = None 


    @property
    def kernel_matrix_train(self):
        """Returns the kernel matrix elements produces by the training data"""
        return self._kernel_matrix_train

    @property
    def backend(self) -> Union[Backend, IBMQBackend, None]:
        """Returns the backend that the QSVM runs on. If it's an ideal 
        simulations, it returns None.
        """
        return self._backend

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance object that the QSVM uses for the 
        simulations, or hardware runs.
        """
        return self._quantum_instance

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the QuantumCircuit that implements the quantum feature map."""
        return self._feature_map
    
    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns the QuantumKernel object of the QSVM model."""
        return self._quantum_kernel

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray): 
        """
        Train the QSVM model. In the case of QSVM where `kernel=precomputed`
        the kernel_matrix elements from the inner products of training data
        vectors need to be passed to fit. Thus, the quantum kernel matrix 
        elements are first evaluated and then passed to the SVC.fit appropriately.
        
        The method also saved the kernel matrix elements of the training data 
        for later use, such as score calculation.
        
        Args:
            train_data: The training data vectors array, 
                        of shape (ntrain, n_features).
            train_labels: The labels of training data vectors, 1 (signal) and 0
                          or -1 (background), of shape (ntrain,).
        """
        self._train_data = train_data
        print("Calculating the quantum kernel matrix elements... ", end="")
        train_time_init = perf_counter()
        self._kernel_matrix_train = self._quantum_kernel.evaluate(train_data)
        train_time_fina = perf_counter()
        print(
        tcols.OKGREEN + f"Done in: {train_time_fina-train_time_init:.2e} s"
        + tcols.ENDC
        )
        super().fit(self._kernel_matrix_train, train_labels)

    def score(self, x: np.ndarray, y: np.ndarray, train_data: bool = False,
              sample_weight: np.ndarray=None) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        Need to compute the corresponding kernel matrix elements and then pass
        to the SVC.score.
        
        Args: 
            x: Training data set of shape (ntrain, nfeatures)
            y: Target (ground truth) labels of the x data array OR of x_test
               if not None, of shape (ntrain,)
            train_data: Flag that specifies whether the score is computed on
                        the training data or new dataset (test). The reason
                        behind this flag is to not compute the kernel matrix
                        on the training data more than once, since it is the
                        computationally expensive task in training the QSVM.
            sample_weight: Weights of the testing samples, of shape (ntrain,)
        Returns: 
            The accuracy of the model on the given dataset x.
        """
        if train_data:
            # FIXME if QSVM.fit is not called the self._kernel_matrix_train 
            # would be None. Hence something might break in test.py when/if
            # using score on train data, without having the prior fit.
            return super().score(self._kernel_matrix_train, y, sample_weight)

        kernel_matrix_test = self._quantum_kernel.evaluate(
            x_vec=x, 
            y_vec=self._train_data,
        )
        return super().score(kernel_matrix_test, y, sample_weight)
        

    def decision_function(self, x_test: np.ndarray, x_train: np.ndarray) \
                          -> np.ndarray:
        """
        Computes the score value (test statistic) of the QSVM model. It computes
        the displacement of the data vector x from the decision boundary. If the
        sign is positive then the predicted label of the model is +1 and -1 
        (or 0) otherwise.
        
        Args: 
            x_test: Array of data vectors of which the scores we want to 
                    compute.
            x_train: Array of data vectors with which the QSVM was trained.
        Returns:
            The corresponding array of scores of x.
        """
        test_kernel_matrix = self._quantum_kernel.evaluate(x_vec=x_test, 
                                                           y_vec=x_train)
        return super().decision_function(test_kernel_matrix)
