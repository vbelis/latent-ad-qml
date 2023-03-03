# Compute the expressibility, entanglement capability, or kernel variance metrics
# of a given circuit. Expressibility and entanglement capability are computed in
# data-dependent setting.

from time import perf_counter
import numpy as np
import pandas as pd
import h5py
from itertools import combinations
import argparse
from typing import Tuple, List, Union
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from triple_e import expressibility
from triple_e import entanglement_capability
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel

from qad.algorithms.kernel_machines.terminal_enhancer import tcols
from qad.algorithms.kernel_machines.feature_map_circuits import (
    u_dense_encoding as u_dense,
)

import warnings  # For pandas frame.append method

warnings.simplefilter(action="ignore", category=FutureWarning)

seed = 42
np.random.seed(seed)


def main(args: dict):
    """Computes the different metrics of the given encoding quantum circuit based
    on the `argparse` options below.

    Parameters
    ----------
    args : dict
        Configuration dictionary with the following parameters.
    n_qubits: int
        Number of qubits for the feature map circuit.
    n_shots: int
        How many fidelity samples to generate per expressibility and entanglement
        capability evaluation.
    n_exp: int
        Number of evaluations ('experiments') of the expressibility and
        entanglement capability. To estimate the mean and std of around
        the true value.
    out_path: str
        Output dataframe to be used for plotting.
    data_path: str
        Path to signal dataset (background or signal .h5 file) to be used in
        expr. calculation. Multiple datasets can also be given for the `expr_vs_qubit`
        data-dependent computation.
    compute: str
        Run different calculations: compute expressibility and entanglement
        capability of different circuits, compute expressibility as a function of the "
        number of qubits, and variance of the kernel as a function of qubits.
        choices=["expr_ent_vs_circ", "expr_vs_nqubits", "var_kernel_vs_nqubits"]
    data_dependent: bool
        Compute the expressibility as a data-dependent quantity.

    Raises
    ------
    TypeError
        If the given computation type is not one from:
        ["expr_ent_vs_circ", "expr_vs_nqubits", "var_kernel_vs_nqubits"].
    """
    circuit_list_expr_ent, circuit_labels = prepare_circs(args)
    data = None
    if args["data_dependent"]:
        data = get_data(args["data_path"])

    switcher = {
        "expr_ent_vs_circ": lambda: compute_expr_ent_vs_circuit(
            args, circuit_list_expr_ent, circuit_labels, data=data
        ),
        "expr_vs_nqubits": lambda: expr_vs_nqubits(args=args, data=data),
        "var_kernel_vs_nqubits": lambda: var_kernel_vs_nqubits(args=args, data=data),
    }
    df_results = switcher.get(args["compute"], lambda: None)()
    if df_results is None:
        raise TypeError(
            tcols.FAIL + "Given computation run does not exist!" + tcols.ENDC
        )


def prepare_circs(args: dict) -> Tuple[List, List]:
    """Prepares the list of circuit needed for evaluation along with their names.
    Following the convention of the paper:

    circuit_names = ["NE_0", "NE_1", "L=1", "L=2", "L=3", "L=4", "L=5", "L=6", "FE"]

    Parameters
    ----------
    args : dict
        Argument dictionary, here used for the number of qubits.

    Returns
    -------
    `Tuple`
        circuit_list_expr_ent: `List`
            Circuit lambda-function callables.
        circuit_labels: `List`
            Corresponding labels as defined in the paper.
    """
    circuit_labels = [
        r"NE$_0$",
        r"NE$_1$",
        "L=1",
        "L=2",
        "L=3",
        "L=4",
        "L=5",
        "L=6",
        "FE",
    ]

    circuit_list_expr_ent = [
        lambda x, rep=rep: u_dense_encoding(x, nqubits=args["n_qubits"], reps=rep)
        for rep in range(1, 7)
    ]
    # Add the two no-entanglement circuits (NE_0, NE_1) at the beginning of the list
    circuit_list_expr_ent.insert(
        0,
        lambda x: u_dense_encoding_no_ent(x, nqubits=args["n_qubits"], reps=1, type=1),
    )
    circuit_list_expr_ent.insert(
        0,
        lambda x: u_dense_encoding_no_ent(x, nqubits=args["n_qubits"], reps=1, type=0),
    )
    # Add the all-to-all CNOT rep3
    circuit_list_expr_ent.append(
        lambda x: u_dense_encoding_all(x, nqubits=args["n_qubits"], reps=3)
    )
    return circuit_list_expr_ent, circuit_labels


def compute_expr_ent_vs_circuit(
    args: dict,
    circuits: List[callable],
    circuit_labels: List[str],
    data: np.ndarray = None,
) -> pd.DataFrame:
    """Computes the expressibility and entanglement capability of a list of circuits,
    in the conventional (uniformly sampled parameters from [0, 2pi])
    and data-dependent manner.

    Parameters
    ----------
    args : dict
        Argparse configuration arguments
    circuits : List[`Callable`]
        List of circuits to compute.
    circuit_labels : List[str]
        Corresponding list of circuit names.
    data : :class:`numpy.ndarray`, optional
        Data distribution. If `None` the circuit parameters are sampled from the
        uniform distribution, by default `None`.

    Returns
    -------
    :class:`pandas.DataFrame`
        Pandas Dataframe containing the circuit name and its computed
        expressibility and entanglement capability, along with their uncertainty.
    """
    print(
        "\nComputing expressibility and entanglement capability of the circuits, "
        f"for {args['n_exp']} evaluations and n_shots = {args['n_shots']}."
        f"\nCircuits: {circuit_labels}"
    )

    n_params = 2 * args["n_qubits"]
    df_expr_ent = pd.DataFrame(
        columns=["circuit", "expr", "expr_err", "ent", "ent_err"]
    )

    # Expr. and Ent. as a function of the circuit depth and amount of CNOT gates
    train_time_init = perf_counter()
    for i, circuit in enumerate(circuits):
        print(f"\nFor circuit {circuit_labels[i]}...")
        expr = []
        # Compute entanglement cap. only once, it fluctuates less than 0.1%
        # and does not require that high statistics
        if circuit_labels[i] in ["NE$_0$", "NE$_1$"]:
            val_ent = 0  # by construction
        else:
            val_ent = entanglement_capability(
                circuit,
                n_params,
                n_shots=1000 if args["n_shots"] > 1000 else args["n_shots"],
                data=data,
            )

        for _ in range(args["n_exp"]):
            val_ex = expressibility(
                circuit,
                n_params,
                method="full",
                n_shots=args["n_shots"],
                n_bins=75,
                data=data,
            )
            expr.append(val_ex)
        expr = np.array(expr)
        ent = np.array(val_ent)
        print(f"expr = {np.mean(expr)} ± {np.std(expr)}")
        print(f"ent = {np.mean(ent)} ± {np.std(ent)}")

        d = {
            "circuit": circuit_labels[i],
            "expr": np.mean(expr),
            "expr_err": np.std(expr),
            "ent": np.mean(ent),
            "ent_err": np.std(ent),
        }
        df_expr_ent = df_expr_ent.append(d, ignore_index=True)

    train_time_fina = perf_counter()
    exec_time = train_time_fina - train_time_init
    print(
        "\nFull computation completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
        f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS
    )
    print("\nResults: \n", df_expr_ent)
    df_expr_ent.to_csv(f"{args['out_path']}.csv")
    print(f"Saving in {args['out_path']}.csv " + tcols.ROCKET)
    return df_expr_ent


def expr_vs_nqubits(
    args: dict,
    rep: int = 3,
    n_exp: str = 20,
    data: Union[np.ndarray, List[np.ndarray]] = None,
) -> pd.DataFrame:
    """Computes the (data-dependent) expressibility of a data embedding circuit as a
    function of the qubit number. Saves the output in a dataframe (.h5) with the
    computed mean values and uncertainties.

    Parameters
    ----------
    args : dict
        Argparse configuration arguments
    rep : int, optional
        Number of repetitions of the data encoding circuit, by default 3
    n_exp : str, optional
        Number of repetitions of the computation (experiments) to assess
        the uncertainty of the stochastically calculated metrics, by default 20
    data : Union[:class:`numpy.ndarray`, List[:class:`numpy.ndarray`]], optional
        List of the datasets available for n_qubits = 4, 8, 16 or None for
        computation with uniformly sampled circuit parameters [0, 2pi]., by default None

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing the computed expressibility as a function of `n_qubits`.
    """
    print(
        f"Computing expressibility as a function of the qubit number, "
        f"for rep = {rep} of the data encoding circuit ansatz."
    )
    df_expr = pd.DataFrame(columns=["expr", "expr_err"])
    n_qubits = range(2, 11)
    if data is not None:
        # n_qubits = [4, 8, 16]
        n_qubits = [8, 16]

    train_time_init = perf_counter()
    for idx, n in enumerate(n_qubits):
        print(f"For n_qubits = {n}: ", end="")
        n_params = 2 * n
        expr = []
        for _ in range(args["n_exp"]):
            val = expressibility(
                lambda x: u_dense_encoding(x, nqubits=n, reps=rep),
                method="full",
                n_params=n_params,
                n_shots=args["n_shots"],
                n_bins=75,
                data=None if data is None else data[idx],
            )
            expr.append(val)

        expr = np.array(expr)
        print(f"expr = {np.mean(expr)} ± {np.std(expr)}")
        d = {
            "expr": np.mean(expr),
            "expr_err": np.std(expr),
        }
        df_expr = df_expr.append(d, ignore_index=True)

    train_time_fina = perf_counter()
    exec_time = train_time_fina - train_time_init
    print(
        "\nFull computation completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
        f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS
    )
    print("\nResults: \n", df_expr)
    df_expr.to_csv(f"{args['out_path']}.csv")
    print(f"Saving in {args['out_path']}.csv " + tcols.ROCKET)
    return df_expr


def var_kernel_vs_nqubits(args: dict, data: np.ndarray, rep: int = 3) -> pd.DataFrame():
    """Computes the variance of the quantum kernel matrix as a function of the number
    of qubits.

    Parameters
    ----------
    args : dict
        Argparse configuration arguments.
    data: :class:`numpy.ndarray`
        Dataset from which to sample.
    rep : int, optional
        Number of repetitions of the data encoding circuit, by default `3`.

    Returns
    -------
    :class:`pandas.DataFrame`
        Variance of the kernel and its correpsonding qubit number.
    """
    n_qubits = [4, 8, 16]
    print(
        f"\n Computing variance of the kernel matrix elements for rep = {rep}"
        " as a function of n_qubits."
    )
    df_var = pd.DataFrame(columns=["n_qubits", "var"])

    train_time_init = perf_counter()
    for idx, n in enumerate(n_qubits):
        quantum_instance = QuantumInstance(
            backend=Aer.get_backend("aer_simulator_statevector")
        )
        quantum_kernel = QuantumKernel(
            u_dense(nqubits=n, reps=3), quantum_instance=quantum_instance
        )
        kernel_matrix_elements = []
        for _ in range(args["n_exp"]):
            data_samples = data[idx][
                np.random.choice(data[idx].shape[0], size=args["n_shots"])
            ]
            kernel_matrix_elements.append(quantum_kernel.evaluate(data_samples))
        kernel_matrix_elements = np.array(kernel_matrix_elements)
        print(f"For n_qubits={n}:  var(K_ij)= {np.var(kernel_matrix_elements)}")
        d = {
            "n_qubits": n,
            "var": np.var(kernel_matrix_elements),
        }
        df_var = df_var.append(d, ignore_index=True)

    train_time_fina = perf_counter()
    exec_time = train_time_fina - train_time_init
    print(
        "\nFull computation completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
        f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS
    )
    print("\nResults: \n", df_var)
    df_var.to_csv(f"{args['out_path']}.csv")
    print(f"Saving in {args['out_path']}.csv " + tcols.ROCKET)
    return df_var


def get_data(
    data_path: Union[str, List[str]], mult_qubits: bool = False
) -> Tuple[np.ndarray]:
    """Loads the data, signal or background, given a path and returns the scaled to
    (0, 2pi) numpy arrays.

    Parameters
    ----------
    data_path : Union[str, List[str]]
        Path to the .h5 dataset, or list of paths for multiple dataset loading.
    mult_qubits : bool, optional
        If True the specified dataset (in data_path) is loaded for
        different qubit numbers, i.e., latent dimensions (4, 8, 16)
        for the kernel machine training/testing., by default False

    Returns
    -------
    Tuple[:class:`numpy.ndarray`]
        The loaded dataset or list of the `numpy` datasets.
    """
    print(tcols.BOLD + f"\nLoading the datasets: {data_path} " + tcols.ENDC)
    h5_data = [h5py.File(dataset_path, "r") for dataset_path in data_path]
    data = [np.asarray(h5_dataset.get("latent_space")) for h5_dataset in h5_data]
    data = [np.reshape(dataset, (len(dataset), -1)) for dataset in data]

    print(f"Loaded {len(data)} datasets array of shapes: ", end="")
    for dataset in data:
        print(f"{dataset.shape} ", end="")
        # rescaling to (0, 2pi) needed for the expr. and ent. computations.
        dataset *= np.pi
        dataset += np.pi
    print()

    if len(data) == 1:
        data = data[0]
    return data


def u_dense_encoding_no_ent(
    x: np.ndarray, nqubits: int = 8, reps: int = 3, type: int = 0
) -> Statevector:
    """Constructs a feature map based on u_dense_encoding but removing entanglement.
    The 'type' argument corresponds the two No-Entanglement (NE) circuits in the paper.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Values of the circuit parameters.
    nqubits : int, optional
        Number of qubits, by default 8
    reps : int, optional
        Repetition of the data encoding circuit, by default 3
    type : int, optional
        Flag to differenciate which type of "Non-entanglement" circuit is used,
        by default 0

    Returns
    -------
    :class:`qiskit.quantum_info.Statevector`
        State vector :class:`qiskit` object corresponding to the state generated by the circuit.
    """
    nfeatures = 2 * nqubits
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            # qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)

        if type == 1:
            for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
                qc.u(x[feature], x[feature + 1], 0, qubit)

    return Statevector.from_instruction(qc)


def u_dense_encoding(
    x: np.ndarray,
    nqubits=8,
    reps=1,
) -> Statevector:
    """Designed feature map circuit for the paper.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Values of the circuit parameters.
    nqubits : int, optional
        Number of qubits for the circuit, by default 8
    reps : int, optional
        Repetitions of the data encoding ansatz, by default 1

    Returns
    -------
    :class:`qiskit.quantum_info.Statevector`
        State vector `qiskit` object corresponding to the state generated by the circuit.
    """
    nfeatures = 2 * nqubits
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            # qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
        for i in range(nqubits):
            if i == nqubits - 1:
                break
            qc.cx(i, i + 1)

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return Statevector.from_instruction(qc)


def u_dense_encoding_all(x: np.ndarray, nqubits: int = 8, reps: int = 3) -> Statevector:
    """Data encoding circuit with all-to-all entanglement gates.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Values of the circuit parameters.
    nqubits : int, optional
        Number of qubits for the circuit, by default 8
    reps : int, optional
        Repetitions of the data encoding circuit, by default 3

    Returns
    -------
    :class:`qiskit.quantum_info.Statevector`
        State vector `qiskit` object corresponding to the state generated by the circuit.
    """
    nfeatures = 2 * nqubits
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            # qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)

        for qpair in list(combinations(range(nqubits), 2)):
            qc.cx(qpair[0], qpair[1])

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return Statevector.from_instruction(qc)


def get_arguments() -> dict:
    """Parses command line arguments and gives back a dictionary.

    Returns
    -------
    dict
        Dictionary with the argparse arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=8,
        help="Number of qubits for feature map circuit.",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        required=True,
        help="How many fidelity samples to generate per expressibility and entanglement "
        "capability evaluation.",
    )
    parser.add_argument(
        "--n_exp",
        type=int,
        required=True,
        help="Number of evaluations ('experiments') of the expressibility and "
        "entanglement capability. To estimate the mean and std of around "
        "the true value",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output dataframe to be used for plotting.",
    )
    parser.add_argument(
        "--data_path",
        nargs="+",
        help="Path to signal dataset (background or signal .h5 file) to be used in "
        "expr. calculation. Multiple datasets can also be given for the `expr_vs_qubit` "
        "data-dependent computation.",
    )
    parser.add_argument(
        "--compute",
        type=str,
        required=True,
        choices=["expr_ent_vs_circ", "expr_vs_nqubits", "var_kernel_vs_nqubits"],
        help="Run different calculations: compute expressibility and entanglement "
        "capability of different circuits, compute expressibility as a function of the "
        "number of qubits, and TODO",
    )
    parser.add_argument(
        "--data_dependent",
        action="store_true",
        help="Compute the expressibility as a data-dependent quantity",
    )
    args = parser.parse_args()
    args = {
        "n_qubits": args.n_qubits,
        "n_shots": args.n_shots,
        "n_exp": args.n_exp,
        "out_path": args.out_path,
        "data_path": args.data_path,
        "compute": args.compute,
        "data_dependent": args.data_dependent,
    }

    return args


if __name__ == "__main__":
    args = get_arguments()
    main(args)
