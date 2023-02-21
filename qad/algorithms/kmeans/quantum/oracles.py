import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator, Pauli


def create_threshold_oracle_operator(nn, idcs_to_mark):
    """Threshold oracle operator for Grover circuit.

    Parameters
    ----------
    nn : int
        number of qubits
    idcs_to_mark : int
        number of elements to mark

    Returns
    -------
    :class:`qiskit.quantum_info.operators.Operator`
        Threshold oracle.
    """

    oracle_matrix = np.identity(2**nn)

    for idx in idcs_to_mark:
        oracle_matrix[idx, idx] = -1

    return Operator(oracle_matrix)


def get_indices_to_mark(dist_arr, threshold):
    """Utility function that finds indices to be marked.

    Parameters
    ----------
    dist_arr : :class:`numpy.ndarray`
        array of distances.
    threshold: float
        current threshold

    Returns
    -------
    :class:`numpy.ndarray`
        the marked indices
    """
    (idx,) = np.nonzero(dist_arr < threshold)
    if idx.size == 0:  # handle minimum element
        idx = [np.argmin(dist_arr)]
    return idx


def create_threshold_oracle_set(dist_arr):
    """Create set of threshold oracles for linear combination.

    Parameters
    ----------
    dist_arr : :class:`numpy.ndarray`
        array of distances.

    Returns
    -------
    List
        List of threshold oracle `Operators`
    """

    cluster_n = len(dist_arr)
    nn = int(math.floor(math.log2(cluster_n) + 1))
    oracles = []

    for threshold in dist_arr:
        idcs_to_mark = get_indices_to_mark(
            dist_arr, threshold
        )  # what for empty set (min ele)?
        oracles.append(create_threshold_oracle_operator(nn, idcs_to_mark))

    return oracles


def create_oracle_lincombi(threshold, dist_arr, oracles):
    """Create linear combination of threshold oracles for Grover circuit.

    Parameters
    ----------
    threshold : float
        current threshold value
    dist_arr : :class:`numpy.ndarray`
        array of distances.
    oracles: List
        list of oracle `Operators`

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        The linear combination threshold oracle circuit.
    """

    idx = np.where(dist_arr == threshold)[0][0]
    delta_coeff = signal.unit_impulse(len(dist_arr), idx)

    oracle_sum = delta_coeff[0] * oracles[0]
    for c, oracle in zip(delta_coeff[1:], oracles[1:]):
        oracle_sum += c * oracle

    nn = int(math.floor(math.log2(len(dist_arr)) + 1))

    qc = QuantumCircuit(nn, name="combi_oracle")

    qc.unitary(oracle_sum, range(nn))

    return qc
