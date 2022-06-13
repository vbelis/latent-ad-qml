# Utility methods for the qsvm.

import os
import joblib
import re
import json
from time import perf_counter
from typing import Tuple, Union, Callable
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
from sklearn.svm import SVC

from qsvm import QSVM
from terminal_enhancer import tcols


def print_accuracy_scores(test_acc: float, train_acc: float):
    """
    Prints the train and test accuracies of the model. 
    Args:
        test_acc: The accuracy of the trained model on the test dataset.
        train_acc: The accuracy of the trained model on the train dataset.
    """
    print(tcols.OKGREEN + f"Training accuracy = {train_acc}")
    print(f"Testing accuracy = {test_acc}" + tcols.ENDC)


def create_output_folder(args: dict, model: Union[SVC, QSVM]) -> str:
    """
    Creates output folder for the model and returns the path (str).
    
    Args:
        args:The argument dictionary defined in the run_training script.
        model: QSVM or SVC object.
    Returns:
            The path where all files relevant to the model will be saved.
    """
    out_path = args["output_folder"] + f"_c={model.C}" 
    if args["quantum"]:
        out_path = out_path+ f"_{args['run_type']}"
        if args["backend_name"] is not None and args["backend_name"] != "none":
            # For briefness remove the "ibmq" prefix for the output folder:
            backend_name = re.sub("ibmq?_", "", args["backend_name"])
            out_path += f"_{backend_name}"
    out_path = "trained_qsvms/" + out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def save_model(model: Union[SVC, QSVM], path: str):
    """
    Saves the qsvm model to a certain path.
    
    Args:
        model: Kernel machine model that we want to save.
        path: Path to save the model in.
    """
    joblib.dump(model, path + "/model")
    print("Trained model saved in: " + path)


def load_model(path: str) -> SVC:
    """
    Load model from pickle file, i.e., deserialisation.
    @path  :: String of full path to load the model from.

    returns :: Joblib object that can be loaded by qiskit.
    """
    return joblib.load(path)

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

def print_model_info(model: Union[SVC, QSVM]):
    """
    Print information about the trained model, such as the C parameter value, 
    number of support vectors, number of training and testing samples.
    Args:
        model: The trained (Q)SVM model.
    """
    print("\n-------------------------------------------")
    print(
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


def get_backend_configuration(backend: Backend) -> Tuple:
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

def time_and_exec(func: Callable, *args) -> float:
    """
    Executes the given function with its arguments, times and returns the
    execution time. Typically used for timing training, and testing tasks
    of the models.

    Args:
        func: Function to execute.
        *args: Arguments of the function.
    Returns:
        The output of the function and the runtime.
    """
    train_time_init = perf_counter()
    func(*args)
    train_time_fina = perf_counter()
    exec_time = train_time_fina-train_time_init
    return exec_time

    
def init_kernel_machine(args:dict, path:str = None) -> Union[SVC, QSVM]:
    """
    Initialises the kernel machine. Depending on the flag, this will be 
    a SVM or a QSVM.
    Args:
        args: 
    """
    if args["quantum"]: 
        print(tcols.OKCYAN + "\nConfiguring the Quantum Support Vector"
              " Machine." + tcols.ENDC)
        return QSVM(args)
    
    print(tcols.OKCYAN + "\nConfiguring the Classical Support Vector"
          " Machine..." + tcols.ENDC)
    return SVC(kernel="rbf", C=args["c_param"], gamma=args["gamma"])
    
def overfit_xcheck(model: Union[QSVM, SVC], train_data, train_labels, test_data, test_labels):
    """
    Computes the training and testing accuracy of the model to cross-check for
    overtraining if the two values are far way from eachother. The execution of
    this function is also timed.
    """
    print("Computing the test dataset accuracy of the models, quick check"
          " for overtraining...")
    test_time_init = perf_counter()
    if isinstance(model, QSVM):
        train_acc = model.score(train_data, train_labels, train_data=True)
    elif isinstance(model, SVC):
        train_acc = model.score(train_data, train_labels)
    else: 
        raise TypeError(tcols.FAIL + "The model should be either a SVC or "
                        "a QSVM object." + tcols.ENDC)
    test_acc = model.score(test_data, test_labels)
    test_time_fina = perf_counter()
    exec_time = test_time_fina - test_time_init  
    print(f"Completed in: {exec_time:.2e} sec. or "f"{exec_time/60:.2e} min. "
          + tcols.ROCKET)
    print_accuracy_scores(test_acc, train_acc)


def export_hyperparameters(outdir):
    """
    Saves the hyperparameters of the model to a json file.
    @outdir :: Directory where to save the json file.
    """
    file_path = os.path.join(outdir, "hyperparameters.json")
    print(file_path)
    #params_file = open(file_path, "w")
    #json.dump(self.hp, params_file)
    #params_file.close()

