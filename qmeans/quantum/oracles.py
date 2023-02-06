import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator, Pauli


def create_threshold_oracle_operator(nn, idcs_to_mark):

    # create the identity matrix on n qubits
    oracle_matrix = np.identity(2**nn)
    # add the -1 phase to marked elements
    for idx in idcs_to_mark:
        oracle_matrix[idx, idx] = -1

    # convert oracle_matrix into an operator, and add it to the quantum circuit
    return Operator(oracle_matrix)


def get_indices_to_mark(dist_arr, threshold):
    (idx,) = np.nonzero(dist_arr < threshold)
    if idx.size == 0:  # handle minimum element
        idx = [np.argmin(dist_arr)]
    return idx


def create_threshold_oracle_set(dist_arr):

    """
    create set of threshold oracles {O_1, ..., O_m} for m possible thresholds
    where oracle_t marks all indices i for which f(i) < threshold t
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

    """join oracles from oracle set with delta function for a given threshold (pick oracles to use)"""

    # create delta-coefficients for linear combi sum_a delta(y - a) * O_a
    idx = np.where(dist_arr == threshold)[0][
        0
    ]  # taking first match if equal distances present
    delta_coeff = signal.unit_impulse(len(dist_arr), idx)

    oracle_sum = delta_coeff[0] * oracles[0]
    for c, oracle in zip(delta_coeff[1:], oracles[1:]):
        oracle_sum += c * oracle

    nn = int(math.floor(math.log2(len(dist_arr)) + 1))

    # create a quantum circuit on nn qubits
    qc = QuantumCircuit(nn, name="combi_oracle")

    # import ipdb; ipdb.set_trace()
    # convert oracle_matrix into an operator, and add it to the quantum circuit
    qc.unitary(oracle_sum, range(nn))

    return qc
