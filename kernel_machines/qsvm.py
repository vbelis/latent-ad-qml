# Module that defines the main object Quantum Support Vector Machine.
# Based on the kernel machine sklearn.svm.SVC implementation.

import joblib
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit.providers.ibmq import IBMQBackend
from qiskit.visualization import plot_circuit_layout
import matplotlib.pyplot as plt
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
    """

    def __init__(self, hpars: dict):
        """
        Args:
            hpars: Hyperparameters of the model and configuration parameters
                   for the training.
        Attributes: TODO
        Methods: TODO?
        """
        super().__init__(kernel="precomputed", C=hpars["c_param"])

        self._nqubits = hpars["nqubits"]
        self._feature_map_name = hpars["feature_map"]
        exec(
            "self._feature_map = fm."
            + self._feature_map_name
            + "(nqubits=self._nqubits)"
        )

        self._backend_config = hpars["config"]
        self._quantum_instance, self._backend = util.configure_quantum_instance(
            ibmq_api_config=hpars["ibmq_api_config"],
            run_type=hpars["run_type"],
            backend_name=hpars["backend_name"],
            **self._backend_config,
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
    def backend_config(self) -> dict:
        """Returns the backend configuration specified during the QSVM training."""
        return self._backend_config

    @property
    def nqubits(self) -> int:
        """Returns the number of qubits of the QSVM circuit."""
        return self._nqubits

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
    def feature_map_name(self) -> str:
        """Returns the quantum feature map name."""
        return self._feature_map_name

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

        The method also, times the kernel matrix element calculation and saves
        the matrix for later use, such as score calculation.

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
            tcols.OKGREEN
            + f"Done in: {train_time_fina-train_time_init:.2e} s"
            + tcols.ENDC
        )
        super().fit(self._kernel_matrix_train, train_labels)

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_data: bool = False,
        sample_weight: np.ndarray = None,
    ) -> float:
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
            return super().score(self._kernel_matrix_train, y, sample_weight)

        kernel_matrix_test = self._quantum_kernel.evaluate(
            x_vec=x,
            y_vec=self._train_data,
        )
        return super().score(kernel_matrix_test, y, sample_weight)

    def decision_function(self, x_test: np.ndarray) -> np.ndarray:
        """
        Computes the score value (test statistic) of the QSVM model. It computes
        the displacement of the data vector x from the decision boundary. If the
        sign is positive then the predicted label of the model is +1 and -1
        (or 0) otherwise.

        Args:
            x_test: Array of data vectors of which the scores we want to
                    compute.
        Returns:
            The corresponding array of scores of x.
        """
        test_kernel_matrix = self._quantum_kernel.evaluate(
            x_vec=x_test,
            y_vec=self._train_data,
        )
        return super().decision_function(test_kernel_matrix)

    def get_transpiled_kernel_circuit(
        self,
        path: str,
        output_format: str = "mpl",
        **kwargs: dict,
    ) -> QuantumCircuit:
        """
        Save the transpiled quantum kernel circuit figure.

        Args:
             quantum_kernel: QuantumKernel object used in the
                                                QSVM training.
             path: Path to save the output figure.
             output_format: The format of the image. Formats:
                            'text', 'mlp', 'latex', 'latex_source'.
             kwargs: Keyword arguemnts for QuantumCircuit.draw().

        Returns:
                Transpiled QuantumCircuit that represents the quantum kernel.
                i.e., the circuit that will be executed on the backend.
        """
        print("\nCreating the quantum kernel circuit...")
        n_params = self._quantum_kernel.feature_map.num_parameters
        feature_map_params_x = ParameterVector("x", n_params)
        feature_map_params_y = ParameterVector("y", n_params)
        qc_kernel_circuit = self._quantum_kernel.construct_circuit(
            feature_map_params_x, feature_map_params_y
        )
        qc_transpiled = self._quantum_instance.transpile(qc_kernel_circuit)[0]

        path += "/quantum_kernel_circuit_plot"
        print(tcols.OKCYAN + "Saving quantum kernel circuit in:" + tcols.ENDC, path)
        qc_transpiled.draw(
            output=output_format,
            filename=path,
            **kwargs,
        )
        plt.clf()
        return qc_transpiled

    def save_circuit_physical_layout(self, circuit: QuantumCircuit, save_path: str):
        """
        Plot and save the quantum circuit and its physical layout on the backend.

        Args:
             circuit: Circuit to plot on the backend.
             save_path: Path to save figure.
        """
        fig = plot_circuit_layout(circuit, self._backend)
        save_path += "/circuit_physical_layout"
        print(
            tcols.OKCYAN + "Saving physical circuit layout in:" + tcols.ENDC, save_path
        )
        fig.savefig(save_path)

    def save_backend_properties(self, path: str):
        """
        Saves a dictionary to file using Joblib. The dictionary contains quantum
        hardware properties, or noisy simulator properties, when the QSVM is not
        trained with ideal simulation.

        Args:
            backend: IBM Quantum computer backend from which we save the
                     calibration data.
            path: String of full path to save the model in.
        """
        properties_dict = self._backend.properties().to_dict()
        path += "/backend_properties_dict"
        joblib.dump(properties_dict, path)
        print(
            tcols.OKCYAN + "Quantum computer backend properties saved in Python"
            " dictionary format in:" + tcols.ENDC,
            path,
        )
