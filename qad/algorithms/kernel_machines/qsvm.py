# Definition of the supervised Quantum Support Vector Machine.
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

import qad.algorithms.kernel_machines.feature_map_circuits as fm
import qad.algorithms.kernel_machines.backend_config as bc
from qad.algorithms.kernel_machines.terminal_enhancer import tcols


class QSVM(SVC):
    """Quantum support vector machine. Supervised SVM model equipped with a
    quantum feature map, implemented by a data encoding circuit.

    Attributes
    ----------
    _nqubits: int
        Number of qubits of the data encoding circuit.
    _feature_map_name: str
        Name of the designed quantum circuit. As defined in :class:`qad.algorithms.kernel_machines.feature_map_circuits`
    _backend_config: dict
        Configuration of the IBMQ backend, e.g. number of shots, qubit layout.
    _quantum_instance: :class:`qiskit.utils.QuantumInstance`
        `QuantumInstance` object required for execution using qiskit.
    _quantum_kernel: :class:`qiskit_machine_learning.kernels.QuantumKernel`
        Quantum kernel function constructed from the data encoding circuit.
    _kernel_matrix_train: :class:`numpy.ndarray`
        Kernel matrix constructed using the training dataset. Saved for computational
        efficiency.
    _train_data: :class:`numpy.ndarray`
        Training dataset. Also saved for computational efficiency, since we don't go
        above a training size of approx 6k.
    """

    def __init__(self, hpars: dict):
        """Initialise the quantum feature map, the quantum instance and quantum kernel.

        Parameters
        ----------
        hpars : dict
            Hyperparameters of the model and configuration parameters for the training.
            This dictionary is defined through `argparse`.
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
        self._quantum_instance, self._backend = bc.configure_quantum_instance(
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
        """Returns the backend that the `QSVM` runs on. If it's an ideal
        simulations, it returns `None`.
        """
        return self._backend

    @property
    def backend_config(self) -> dict:
        """Returns the backend configuration specified during the `QSVM` training."""
        return self._backend_config

    @property
    def nqubits(self) -> int:
        """Returns the number of qubits of the :class:`qad.algorithms.kernel_machines.qsvm.QSVM` circuit."""
        return self._nqubits

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance object that the :class:`qad.algorithms.kernel_machines.qsvm.QSVM` uses for the
        simulations, or hardware runs.
        """
        return self._quantum_instance

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the :class:`qiskit.circuit.QuantumCircuit` that implements the quantum feature map."""
        return self._feature_map

    @property
    def feature_map_name(self) -> str:
        """Returns the quantum feature map name."""
        return self._feature_map_name

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns the :class:`qiskit_machine_learning.kernels.QuantumKernel` object of the QSVM model."""
        return self._quantum_kernel

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray):
        """Train the `QSVM` model. In the case of `QSVM` where `kernel=precomputed`
        the kernel_matrix elements from the inner products of training data
        vectors need to be passed to fit. Thus, the quantum kernel matrix
        elements are first evaluated and then passed to the `SVC.fit` appropriately.

        The method also, times the kernel matrix element calculation and saves
        the matrix for later use, such as score calculation.

        Parameters
        ----------
        train_data : :class:`numpy.ndarray`
            The training data vectors array of shape (ntrain, n_features).
        train_labels : :class:`numpy.ndarray`
            The labels of training data vectors, 1 (signal) and 0
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
        """Returns the mean accuracy on the given test data and labels.
        Need to compute the corresponding kernel matrix elements and then pass
        to the `sklearn.svm.SVC.score`.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Training dataset of shape (ntrain, nfeatures)
        y : :class:`numpy.ndarray`
            Target (ground truth) labels of the x_train or of x_test data arrays
        train_data : bool, optional
            Flag that specifies whether the score is computed on
            the training data or new dataset (test). The reason
            behind this flag is to not compute the kernel matrix
            on the training data more than once, since it is the
            computationally expensive task in training the `QSVM`, by default `False`
        sample_weight : :class:`numpy.ndarray`, optional
            Weights of the testing samples, of shape (ntrain,), by default `None`

        Returns
        -------
        float
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
        """Computes the score value (test statistic) of the `QSVM` model. It computes
        the displacement of the data vector x from the decision boundary. If the
        sign is positive then the predicted label of the model is +1 and -1
        (or 0) otherwise.

        Parameters
        ----------
        x_test : :class:`numpy.ndarray`
            Test dataset data vectors.

        Returns
        -------
        :class:`numpy.ndarray`
            Score array of x.
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
        """Construct, save, and return the transpiled quantum kernel circuit figure.

        Parameters
        ----------
        path : str
            Path for the output figure
        output_format : str, optional
            Output image file format, by default "mpl"

        Returns
        -------
        :class:`qiskit.circuit.QuantumCircuit`
            Transpiled `QuantumCircuit` that represents the quantum kernel.
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
        """Plot and save the quantum circuit and its physical layout on the backend.
        Used only for hardware or noisy simulation runs.

        Parameters
        ----------
        circuit : :class:`qiskit.circuit.QuantumCircuit`
            Circuit to map to the physical qubits of the backend.
        save_path : str
            Path to save the figure.
        """
        fig = plot_circuit_layout(circuit, self._backend)
        save_path += "/circuit_physical_layout"
        print(
            tcols.OKCYAN + "Saving physical circuit layout in:" + tcols.ENDC, save_path
        )
        fig.savefig(save_path)

    def save_backend_properties(self, path: str):
        """Saves a dictionary to file using `joblib` package. The dictionary contains quantum
        hardware properties, or noisy simulator properties, when the QSVM is not
        trained with ideal simulation.

        Parameters
        ----------
        path : str
            Output path.
        """
        properties_dict = self._backend.properties().to_dict()
        path += "/backend_properties_dict"
        joblib.dump(properties_dict, path)
        print(
            tcols.OKCYAN + "Quantum computer backend properties saved in Python"
            " dictionary format in:" + tcols.ENDC,
            path,
        )
