# Utility methods for the qsvm.

import os
import joblib
from typing import Tuple, Union
from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_circuit_layout
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.backends import AerSimulator
from qiskit.providers import Backend, BaseBackend
from qiskit.providers.ibmq import IBMQBackend
from qiskit_machine_learning.kernels import QuantumKernel
import re
import numpy as np
from sklearn.svm import SVC

from . import qdata as qd
from .terminal_colors import tcols


def print_accuracy_scores(test_acc: float, train_acc: float):
    """
    Prints the train and test accuracies of the model. 
    Args:
        test_acc: The accuracy of the trained model on the test dataset.
        train_acc: The accuracy of the trained model on the train dataset.
    """
    print(tcols.OKGREEN + f"Training accuracy = {train_acc}")
    print(f"Testing accuracy     = {test_acc}" + tcols.ENDC)


def create_output_folder(args: dict, qsvm: SVC) -> str:
    """
    Creates output folder for the qsvm and returns the path (str)
    @args (dict)         :: The argument dictionary defined in the qsvm_launch
                            script.
    @qsvm (sklearn.SVC)  :: QSVM object.
    Returns:
            @out_path (str), the path where all files relevant to the qsvm
            will be saved.
    """
    args["output_folder"] = (
        args["output_folder"] + f"_c={qsvm.C}" + f"_{args['run_type']}"
    )
    if args["backend_name"] is not None:
        # For briefness remove the "ibmq" prefix from the backend_name for the
        # output folder:
        backend_name = re.sub("ibmq?_", "", args["backend_name"])
        args["output_folder"] += f"_{backend_name}"
    out_path = "trained_qsvms/" + args["output_folder"]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def save_qsvm(model: SVC, path: str):
    """
    Saves the qsvm model to a certain path.
    @model :: vqc model object.
    @path  :: String of full path to save the model in.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    joblib.dump(model, path)
    print("Trained model saved in: " + path)


def load_qsvm(path: str) -> SVC:
    """
    Load model from pickle file, i.e., deserialisation.
    @path  :: String of full path to load the model from.

    returns :: Joblib object that can be loaded by qiskit.
    """
    return joblib.load(path)


def get_data(args: dict) -> Tuple:
    """
    Loads the appropriate data for the qsvm training. Either a dataset specified
    by a path (to a np.array object) or a latent representation of a dataset passed
    through a pre-trained Autoencoder.

    Args:
        args: Dictionary with all the specification and parameters for the qsvm
              training.
    Returns:
        The AE dataset or an external dataset. The dataset is a tuple !!!!!
    """
    if args["model_path"] is None: 
        print(tcols.OKCYAN + "No AE model is loaded. Loading"
              " a dataset directly..." + tcols.ENDC)
        return get_external_data(args)
    return get_ae_data(args)

def get_external_data(args: dict) -> Tuple:
    """
    Loads an external dataset for the qsvm training. 
    Args:
        args: Dictionary with all the specification and parameters for the qsvm
              training.
    Returns:
        A tuple with the data "loaders" (inspired by PyTorch jargon). Each loader
        consists of the data vectors (features) and the corresponding labels.
    """
    print(tcols.OKCYAN + "Loading training and testing datasets..."
          + tcols.ENDC)
    train_features = np.load(f'{args["data_folder"]}x_data_train.npy')
    train_labels = np.load(f'{args["data_folder"]}y_data_train.npy')
    test_features = np.load(f'{args["data_folder"]}x_data_test.npy')
    test_labels = np.load(f'{args["data_folder"]}y_data_test.npy')

    print("Loaded training and testing datasets of shapes: "
          f"{train_features.shape} and {test_features.shape} respectively.\n")
    train_loader = [train_features[:args["ntrain"]], 
                    train_labels[:args["ntrain"]]]
    test_loader = [test_features[:args["ntest"]], test_labels[:args["ntest"]]]

    return train_loader, test_loader


def save_backend_properties(backend: Union[Backend, BaseBackend], path: str):
    """
    Saves a dictionary to file using Joblib. The dictionary contains quantum
    hardware properties, or noisy simulator properties, when the QSVM is not 
    trained with ideal simulation.

    Args:
        backend: IBM Quantum computer backend from which we save the 
                 calibration data.
        path: String of full path to save the model in.
    """
    properties_dict = backend.properties().to_dict()
    joblib.dump(properties_dict, path)
    print(
        tcols.OKCYAN + "Quantum computer backend properties saved in Python"
        " dictionary format in:" + tcols.ENDC,
        path,
    )

def print_model_info(model: SVC):
    """
    Print information about the trained model, such as the C parameter value, 
    number of support vectors, number of training and testing samples.
    Args:
        model: The trained (Q)SVM model.
    """
    print("\n-------------------------------------------")
    print(
        #f"ntrain = {len(qdata.ae_data.trtarget)}, "
        #f"ntest = {len(qdata.ae_data.tetarget)}, "
        f"C = {model.C}\n"
        f"For classes: {model.classes_}, the number of support vectors for "
        f"each class are: {model.n_support_}"
    )
    print("-------------------------------------------\n")


def get_quantum_kernel_circuit(
    quantum_kernel: QuantumKernel, 
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
    print("\nCreating the quanntum kernel circuit...")
    n_params = quantum_kernel.feature_map.num_parameters
    feature_map_params_x = ParameterVector("x", n_params)
    feature_map_params_y = ParameterVector("y", n_params)
    qc_kernel_circuit = quantum_kernel.construct_circuit(
        feature_map_params_x, feature_map_params_y
    )
    qc_transpiled = quantum_kernel.quantum_instance.transpile(qc_kernel_circuit)[0]

    path += "/quantum_kernel_circuit_plot"
    print(tcols.OKCYAN + "Saving quantum kernel circuit in:" + tcols.ENDC, path)
    qc_transpiled.draw(
        output=output_format,
        filename=path,
        **kwargs,
    )
    return qc_transpiled


def save_circuit_physical_layout(circuit: QuantumCircuit, backend, save_path):
    """
    Plot and save the quantum circuit and its physical layout on the backend.

    Args:
         @circuit (QuantumCircuit) :: Circuit to plot on the backend.
         @backend                  :: The physical quantum computer backend.
         @save_path (str)          :: Path to save figure.
    """
    fig = plot_circuit_layout(circuit, backend)
    save_path += "/circuit_physical_layout"
    print(tcols.OKCYAN + "Saving physical circuit layout in:" + tcols.ENDC, save_path)
    fig.savefig(save_path)


def connect_quantum_computer(ibmq_api_config:dict, backend_name: str) -> IBMQBackend:
    """
    Load a IBMQ-experience backend using a token (IBM-CERN hub credentials)
    This backend (i.e. quantum computer) can either be used for running on
    the real device or to load the calibration (noise/error info). With the
    latter data we can do a simulation of the hardware behaviour.

    Args:
        ibmq_api_config: Configuration file for the IBMQ API token
                                   and provider information.
        backend_name: Quantum computer name.
    Returns:
        IBMQBackend qiskit object.
    """
    print("Enabling IBMQ account using provided token...", end="")
    IBMQ.enable_account(ibmq_api_config["token"])
    provider = IBMQ.get_provider(
        hub=ibmq_api_config["hub"],
        group=ibmq_api_config["group"],
        project=ibmq_api_config["project"],
    )
    try:
        quantum_computer_backend = provider.get_backend(backend_name)
    except QiskitBackendNotFoundError:
        raise AttributeError(
            tcols.FAIL + "Backend name not found in provider's" " list" + tcols.ENDC
        )
    print(tcols.OKGREEN + " Loaded IBMQ backend: " + backend_name + "." + tcols.ENDC)
    return quantum_computer_backend


def get_backend_configuration(backend) -> Tuple:
    """
    Gather backend configuration and properties from the calibration data.
    The output is used to build a noise model using the qiskit aer_simulator.

    Args:
    @backend :: IBMQBackend object representing a a real quantum computer.
    Returns:
            @noise_model from the 1-gate, 2-gate (CX) errors, thermal relaxation,
            etc.
            @coupling_map: connectivity of the physical qubits.
            @basis_gates: gates that are physically implemented on the hardware.
            the transpiler decomposes the generic/abstract circuit to these
            physical basis gates, taking into acount also the coupling_map.
    """
    noise_model = NoiseModel.from_backend(backend)
    coupling_map = backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    return noise_model, coupling_map, basis_gates


def ideal_simulation(**kwargs) -> QuantumInstance:
    """
    Defines QuantumInstance for an ideal (statevector) simulation (no noise, no
    sampling statistics uncertainties).

    Args:
         Keyword arguments of the QuantumInstance object.
    """

    print(tcols.BOLD + "\nInitialising ideal (statevector) simulation." + tcols.ENDC)
    quantum_instance = QuantumInstance(
        backend=Aer.get_backend("aer_simulator_statevector"), **kwargs
    )
    # None needed to specify that no backend device is loaded for ideal sim.
    return quantum_instance, None


def noisy_simulation(ibmq_api_config, backend_name, **kwargs) -> Tuple:
    """
    Prepare a QuantumInstance object for simulation with noise based on the
    real quantum computer calibration data.

    Args:
        @ibmq_api_config (dict) :: Configuration file for the IBMQ API token
                                   and provider information.
        @backend_name (str)     :: Name of the quantum computer,
                                   form ibm(q)_<city_name>.
        @kwargs                 :: Keyword arguments for the QuantumInstance.
    Returns:
            @QuantumInstance object to be used for the simulation.
            @backend on which the noisy simulation is based.
    """
    print(tcols.BOLD + "\nInitialising noisy simulation." + tcols.ENDC)
    quantum_computer_backend = connect_quantum_computer(ibmq_api_config, backend_name)
    backend = AerSimulator.from_backend(quantum_computer_backend)

    quantum_instance = QuantumInstance(backend=backend, **kwargs)
    return quantum_instance, quantum_computer_backend


def hardware_run(backend_name, ibmq_api_config, **kwargs) -> Tuple:
    """
    Configure QuantumInstance based on a quantum computer. The circuits will
    be sent as jobs to be exececuted on the specified device in IBMQ.

    Args:
         @backend_name (str) :: Name of the quantum computer, form ibmq_<city_name>.
         @ibmq_api_config (dict) :: Configuration file for the IBMQ API token
                                    and provider information.
    Returns:
            @QuantumInstance object with quantum computer backend.
            @The quantum computer backend object.
    """
    print(tcols.BOLD + "\nInitialising run on a quantum computer." + tcols.ENDC)
    quantum_computer_backend = connect_quantum_computer(ibmq_api_config, backend_name)
    quantum_instance = QuantumInstance(backend=quantum_computer_backend, **kwargs)
    return quantum_instance, quantum_computer_backend


def configure_quantum_instance(
    ibmq_api_config, run_type, backend_name=None, **kwargs
) -> Tuple:
    """
    Gives the QuantumInstance object required for running the Quantum kernel.
    The quantum instance can be configured for a simulation of a backend with
    noise, an ideal (statevector) simulation or running on a real quantum
    device.
    Args:
         @ibmq_api_config (dict) :: Configuration file for the IBMQ API token
                                    and provider information.

         @run_type (string)      :: Takes values the possible values {ideal,
                                    noisy, hardware} to specify what type of
                                    backend will be provided to the quantum
                                    instance object.
         @backend_name (string)  :: Name of the quantum computer to run or base
                                    the noisy simulation on. For ideal runs it
                                    can be set to "none".
         @**kwargs     (dict)    :: Dictionary of keyword arguments for the
                                    QuantumInstance.
    Returns:
            @QuantumInstance object to be used in the QuantumKernel training.
            @backend that is being used. None if an ideal simulation is initi-
             ated.
    """
    if (run_type == "noisy" or run_type == "hardware") and (backend_name is None):
        raise TypeError(
            tcols.FAIL + "Need to specify backend name ('ibmq_<city_name>')"
            " when running a noisy simulation or running on hardware!" + tcols.ENDC
        )

    switcher = {
        "ideal": lambda: ideal_simulation(**kwargs),
        "noisy": lambda: noisy_simulation(
            ibmq_api_config=ibmq_api_config, backend_name=backend_name, **kwargs
        ),
        "hardware": lambda: hardware_run(
            backend_name=backend_name, ibmq_api_config=ibmq_api_config, **kwargs
        ),
    }

    quantum_instance, backend = switcher.get(run_type, lambda: None)()
    if quantum_instance is None:
        raise TypeError(
            tcols.FAIL + "Specified programme run type does not" "exist!" + tcols.ENDC
        )
    return quantum_instance, backend


def get_run_type():
    # switcher stuff # FIXME follow the ae_qml project design.
    pass
