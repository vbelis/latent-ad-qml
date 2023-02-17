from time import perf_counter
from typing import Tuple, Callable
from qiskit import IBMQ
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.backends import AerSimulator
from qiskit.providers import Backend
from qiskit.providers.ibmq import IBMQBackend

from qad.algorithms.kernel_machines.terminal_enhancer import tcols


def ideal_simulation(**kwargs) -> QuantumInstance:
    """Defines QuantumInstance for an ideal (statevector) simulation (no noise, no
    sampling statistics uncertainties).

    Returns
    -------
    :class:`qiskit.utils.QuantumInstance`
        Object used for the execution of quantum kernel machines using :class:`qiskit`.
    """
    print(tcols.BOLD + "\nInitialising ideal (statevector) simulation." + tcols.ENDC)
    quantum_instance = QuantumInstance(
        backend=Aer.get_backend("aer_simulator_statevector"), **kwargs
    )
    # None needed to specify that no backend device is loaded for ideal sim.
    return quantum_instance, None


def noisy_simulation(
    ibmq_api_config: dict, backend_name: str, **kwargs
) -> Tuple[QuantumInstance, Backend]:
    """Prepare a :class:`qiskit.utils.QuantumInstance` object for simulation with noise based on the
    real quantum computer calibration data.

    Parameters
    ----------
    ibmq_api_config : dict
        Configuration file with API token and private configuration for IBMQ connection.
    backend_name : str
        Name of the quantum computer, form ibm(q)_<city_name>.
    kwargs:
        Keyword arguments for the :class:`qiskit.utils.QuantumInstance`.

    Returns
    -------
    Tuple
        quantum_instance: :class:`qiskit.utils.QuantumInstance`
            `Quantum instance` object need to execute the quantum models in qiskit.
        quantum_computer_backend: :class:`qiskit.providers.Backend`
            `Backend` object representing a quantum computer from which a noisy
            simulation is based.
    """
    print(tcols.BOLD + "\nInitialising noisy simulation." + tcols.ENDC)
    quantum_computer_backend = connect_quantum_computer(ibmq_api_config, backend_name)
    backend = AerSimulator.from_backend(quantum_computer_backend)

    quantum_instance = QuantumInstance(backend=backend, **kwargs)
    return quantum_instance, quantum_computer_backend


def connect_quantum_computer(ibmq_api_config: dict, backend_name: str) -> IBMQBackend:
    """Load a IBMQ-experience backend using a token (IBM-CERN hub credentials)
    This backend (i.e. quantum computer) can either be used for running on
    the real device or to load the calibration (noise/error info). With the
    latter data we can do a simulation of the hardware behaviour.

    Parameters
    ----------
    ibmq_api_config : dict
        Configuration file for the `IBMQ` API token and provider information.
    backend_name : str
        Quantum computer name

    Returns
    -------
    :class:`qiskit.providers.ibmq.IBMQBackend`
        Backend object used for executing the quantum models in :class:`qiskit`.

    Raises
    ------
    AttributeError
        When a quantum computer name that doesn't exist is given.
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


def configure_quantum_instance(
    ibmq_api_config: dict, run_type: str, backend_name: str = None, **kwargs
) -> Tuple[QuantumInstance, Backend]:
    """Gives the QuantumInstance object required for running the Quantum kernel.
    The quantum instance can be configured for a simulation of a backend with
    noise, an ideal (statevector) simulation or running on a real quantum
    device.

    Parameters
    ----------
    ibmq_api_config : dict
        Configuration file for the IBMQ API token and provider information.
    run_type : str
        Takes values the possible values {ideal,noisy, hardware} to specify
        what type of backend will be provided to the quantum instance object.
    backend_name : str, optional
        Name of the quantum computer to run or base the noisy simulation on.
        For ideal runs it can be set to "none"., by default `None`.
    kwargs:
        Dictionary of keyword arguments for the :class:`qiskit.utils.QuantumInstance`.
    Returns
    -------
    quantum_instance: :class:`qiskit.utils.QuantumInstance`
        Object with quantum computer backend.
    backend: :class:`qiskit.providers.Backend`
        The quantum computer backend object.

    Raises
    ------
    TypeError
        When `run_type` is not in {ideal, noisy, hardware}.
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


def get_backend_configuration(backend: Backend) -> Tuple:
    """Gather backend configuration and properties from the calibration data.
    The output is used to build a noise model using the qiskit aer_simulator.

    Parameters
    ----------
    backend : :class:`qiskit.providers.Backend`
        IBMQBackend object representing a a real quantum computer.

    Returns
    -------
    Tuple
        noise_model:  From the 1-gate, 2-gate (CX) errors, thermal relaxation,
            etc.
        coupling_map: list
            Connectivity of the physical qubits.
        basis_gates: list
            Gates that are physically implemented on the hardware.
            the transpiler decomposes the generic/abstract circuit to these
            physical basis gates, taking into acount also the coupling_map.
    """
    noise_model = NoiseModel.from_backend(backend)
    coupling_map = backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    return noise_model, coupling_map, basis_gates


def hardware_run(
    backend_name: str, ibmq_api_config: dict, **kwargs
) -> Tuple[QuantumInstance, Backend]:
    """Configure :class:`qiskit.utils.QuantumInstance` based on a quantum computer. The circuits will
    be sent as jobs to be exececuted on the specified device in `IBMQ`.

    Parameters
    ----------
    backend_name : str
        Name of the quantum computer, form ibmq_<city_name>.
    ibmq_api_config : dict
        Configuration file for the `IBMQ` API token and provider information.

    Returns
    -------
    Tuple
        quantum_instance: :class:`qiskit.utils.QuantumInstance`
            Object with quantum computer backend.
        backend: :class:`qiskit.providers.Backend`
            The quantum computer backend object.
    """
    print(tcols.BOLD + "\nInitialising run on a quantum computer." + tcols.ENDC)
    quantum_computer_backend = connect_quantum_computer(ibmq_api_config, backend_name)
    quantum_instance = QuantumInstance(backend=quantum_computer_backend, **kwargs)
    return quantum_instance, quantum_computer_backend


def time_and_exec(func: Callable, *args) -> float:
    """Executes the given function with its arguments, times and returns the
    execution time. Typically used for timing training, and testing tasks
    of the models.

    Parameters
    ----------
    func : Callable
        Function to execute.
    args: Arguments of the function.

    Returns
    -------
    float
        The runtime.
    """
    train_time_init = perf_counter()
    func(*args)
    train_time_fina = perf_counter()
    exec_time = train_time_fina - train_time_init
    return exec_time
