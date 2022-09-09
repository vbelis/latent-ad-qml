# Utility methods for the SVM and QSVM training and testing.

import os
import joblib
import re
import json
import numpy as np
from time import perf_counter
from typing import Tuple, Union, Callable
from qiskit import IBMQ
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.backends import AerSimulator
from qiskit.providers import Backend
from qiskit.providers.ibmq import IBMQBackend
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from qsvm import QSVM
from one_class_svm import CustomOneClassSVM
from one_class_qsvm import OneClassQSVM
from terminal_enhancer import tcols


def print_accuracy_scores(test_acc: float, train_acc: float, is_unsup: bool):
    """ FIXME: takes y_scores not accuracies
    Prints the train and test accuracies of the model.
    Args:
        is_unsup: Flag if the model is unsupervised. The printing is slighlty
                  different if so.
        test_acc: The accuracy of the trained model on the test dataset.
        train_acc: The accuracy of the trained model on the train dataset.
    """
    if is_unsup:
        print(tcols.OKGREEN + f"Fraction of outliers in the traning set = {train_acc}")
    else:
        print(tcols.OKGREEN + f"Training accuracy = {train_acc}")
    print(f"Testing accuracy = {test_acc}" + tcols.ENDC)


def create_output_folder(
    args: dict, model: Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM]
) -> str:
    """
    Creates output folder for the model and returns the path (str).

    Args:
        args: The argument dictionary defined in the run_training script.
        model: QSVM or SVC object.
    Returns:
            The path where all files relevant to the model will be saved.
    """
    if args["unsup"]:
        out_path = args["output_folder"] + f"_nu={model.nu}"
    else:
        out_path = args["output_folder"] + f"_c={model.C}"
    if args["quantum"]:
        out_path = out_path + f"_{args['run_type']}"
        if args["backend_name"] is not None and args["backend_name"] != "none":
            # For briefness remove the "ibmq" prefix for the output folder:
            backend_name = re.sub("ibmq?_", "", args["backend_name"])
            out_path += f"_{backend_name}"
    out_path = "trained_qsvms/" + out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def save_model(model: Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM], path: str):
    """
    Saves the qsvm model to a certain path.

    Args:
        model: Kernel machine model that we want to save.
        path: Path to save the model in.
    """
    if isinstance(model, QSVM) or isinstance(model, OneClassQSVM):
        np.save(path + "/train_kernel_matrix.npy", model.kernel_matrix_train)
        qc_transpiled = model.get_transpiled_kernel_circuit(path)
        if model.backend is not None:
            model.save_circuit_physical_layout(qc_transpiled, path)
            model.save_backend_properties(path)
    joblib.dump(model, path + "/model")
    print("Trained model and plots saved in: " + tcols.OKCYAN + path + tcols.ENDC)


def load_model(path: str) -> Union[QSVM, SVC, CustomOneClassSVM, OneClassQSVM]:
    """
    Load model from pickle file, i.e., deserialisation.

    Args:
        path: String of full path to load the model from.
    Returns:
        Joblib object of the trained QSVM or SVM model.
    """
    return joblib.load(path)


def print_model_info(model: Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM]):
    """
    Print information about the trained model, such as the C parameter value,
    number of support vectors, number of training and testing samples.
    Args:
        model: The trained (Q)SVM model.
    """
    print("\n-------------------------------------------")
    if isinstance(model, SVC):  # Check if it is a supervised SVM or a QSVM
        print(
            f"C = {model.C}\n"
            f"For classes: {model.classes_}, the number of support vectors for "
            f"each class are: {model.n_support_}"
        )
    else:
        print(f"nu = {model.nu}\n" f"Number of support vectors: {model.n_support_}")
    print("-------------------------------------------\n")


def connect_quantum_computer(ibmq_api_config: dict, backend_name: str) -> IBMQBackend:
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
    exec_time = train_time_fina - train_time_init
    return exec_time


def init_kernel_machine(args: dict) -> Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM]:
    """
    Initialises the kernel machine. Depending on the flag, this will be
    a SVM or a QSVM.
    Args:
        args: The argument dictionary defined in the training script.
    """
    # TODO maybe do a switcher for more neatness?
    if args["quantum"]:
        if args["unsup"]:
            print(
                tcols.OKCYAN + "\nConfiguring the one-class Quantum Support Vector"
                " Machine." + tcols.ENDC
            )
            return OneClassQSVM(args)
        print(
            tcols.OKCYAN + "\nConfiguring the Quantum Support Vector"
            " Machine." + tcols.ENDC
        )
        return QSVM(args)

    if args["unsup"]:
        print(
            tcols.OKCYAN + "\nConfiguring the one-class Classical Support Vector"
            " Machine..." + tcols.ENDC
        )
        return CustomOneClassSVM(kernel="rbf", nu=args["nu_param"], gamma=args["gamma"])
    print(
        tcols.OKCYAN + "\nConfiguring the Classical Support Vector"
        " Machine..." + tcols.ENDC
    )
    return SVC(kernel="rbf", C=args["c_param"], gamma=args["gamma"])


def eval_metrics(
    model: Union[QSVM, SVC, CustomOneClassSVM, OneClassQSVM],
    train_data,
    train_labels,
    test_data,
    test_labels,
    out_path,
):
    """
    For the supervised models, it computes the training and testing accuracy of
    the model to cross-check for overtraining. In the unsupervised case, the fraction
    of training datapoints that have been flagged as anomalies is computed.
    TODO mention ROC plot, PR plot.
    The execution of this function is also timed.
    """
    print(
        "Computing the test dataset accuracy of the models, quick check"
        " for overtraining..."
    )
    test_time_init = perf_counter()
    train_acc = None
    if (
        isinstance(model, QSVM)
        or isinstance(model, OneClassQSVM)
        or isinstance(model, CustomOneClassSVM)
    ):
        train_acc = model.score(train_data, train_labels, train_data=True)
    elif isinstance(model, SVC):
        train_acc = model.score(train_data, train_labels)
    else:
        raise TypeError(
            tcols.FAIL
            + "The model should be either a SVC or a QSVM or a OneClassSVM or"
            " a OneClassQSVM object." + tcols.ENDC
        )
    y_score = model.decision_function(test_data)
    compute_roc_pr_curves(test_labels, y_score, out_path)
    
    y_score[y_score>0.] = 1
    y_score[y_score<0.] = 0
    test_acc = accuracy_score(test_labels, y_score)
    print_accuracy_scores(test_acc, train_acc, isinstance(model, OneClassSVM))
    
    test_time_fina = perf_counter()
    exec_time = test_time_fina - test_time_init
    print(
        f"Completed evaluation in: {exec_time:.2e} sec. or "
        f"{exec_time/60:.2e} min. " + tcols.ROCKET
    )    


def compute_roc_pr_curves(test_labels: np.ndarray, y_score: np.ndarray, out_path: str):
    """
    Computes the ROC and Precision-Recall (PR) curves and saves them in the model
    out_path. Also, prints the 1/FPR value around a TPR working point, default=0.8.

    Args:
        test_labels: The test dataset truth labels.
        y_score: The model scores on the test dataset.
        out_path: Path to save the the plots to. Same as the trained model path.
    """
    fpr, tpr, thresholds = roc_curve(y_true=test_labels, y_score=y_score)
    auc = roc_auc_score(test_labels, y_score)
    plt.plot(tpr, 1./fpr, label=f"AUC: {auc:.3f}")
    plt.yscale("log")
    plt.xlabel("TPR")
    plt.ylabel("FPR")
    plt.legend()
    plt.savefig(out_path + "/roc.pdf")
    plt.clf()
    get_fpr_around_tpr_point(fpr, tpr)

    p, r, thresholds = precision_recall_curve(test_labels, probas_pred=y_score)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(out_path +"/pr.pdf")
    print("\nComputed ROC and PR curves " + tcols.SPARKS)

def get_fpr_around_tpr_point(fpr: np.ndarray, tpr: np.ndarray, tpr_working_point: float = 0.8):
    """
    Computes the mean 1/FPR value that corresponds to a small window aroun a given
    TPR working point (default: 0.8). If there are no values in the window, it widened 
    sequentially until it includes some values around the working point.

    Args:
        fpr: The false positive rate values of the model.
        tpr: The true positive rate values of the model.
        tpr_working_point: True positive rate working point, typical values {0.4, 0.6, 0.8}
    """
    ind = np.array([])
    low_bound = tpr_working_point*0.999
    up_bound = tpr_working_point*1.001
    while len(ind) == 0:
        ind = np.where(np.logical_and(tpr>=low_bound, tpr<=up_bound))[0]
        low_bound *= 0.99 # open the window by 1%
        up_bound *= 1.01
    print(f"\nTPR values around {tpr_working_point} window with lower bound {low_bound}"
          f" and upper bound: {up_bound}")
    print(f"Corresponding mean 1/FPR value in that window: {np.mean(1./fpr[ind]):.3f} ± " 
          f"{np.std(1./fpr[ind]):.3f}")


def export_hyperparameters(
    model: Union[QSVM, SVC, CustomOneClassSVM, OneClassQSVM], outdir: str
):
    """
    Saves the hyperparameters of the model to a json file. QSVM and SVM have
    different hyperparameters.

    Args:
        outdir: Directory where to save the json file, same as the saved model.
    """
    file_path = os.path.join(outdir, "hyperparameters.json")
    if isinstance(model, QSVM):
        hp = {
            "C": model.C,
            "nqubits": model.nqubits,
            "feature_map_name": model.feature_map_name,
            "backend_config": model.backend_config,
        }
    else:
        hp = {"C": model.C}
    params_file = open(file_path, "w")
    json.dump(hp, params_file)
    params_file.close()
