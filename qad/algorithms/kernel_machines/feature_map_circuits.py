# Definitions of data encoding circuits investigated in https://arxiv.org/abs/2301.10780
# with linear, all-to-all, and no entanglement.

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from itertools import combinations
import numpy as np


def u_dense_encoding(nqubits: int = 8, reps: int = 3) -> QuantumCircuit:
    """Constructs the data encoding circuit that was designed for the
    paper: https://arxiv.org/abs/2301.10780

    Parameters
    ----------
    nqubits: `int`
        Number of qubits for the circuit.
    reps: `int`


    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Quantum circuit to be used as a feature map for quantum the kernel machine.
    """
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)

        for i in range(nqubits):
            if i == nqubits - 1:
                break
            qc.cx(i, i + 1)

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return qc


def u_dense_encoding_all(nqubits: int = 8, reps: int = 3) -> QuantumCircuit:
    """Data encoding circuit with all-to-all entanglement.

    Parameters
    ----------
    nqubits : int, optional
        Number of qubits for the circuit, by default 8
    reps : int, optional
        Number of repetition of the data encoding circuit, by default 3

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Quantum circuit to be used as a feature map for quantum the kernel machine.
    """
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            # qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)

        for qpair in list(combinations(range(nqubits), 2)):
            qc.cx(qpair[0], qpair[1])

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return qc


def u_dense_encoding_no_ent(
    nqubits: int = 8, reps: int = 3, type: int = 0
) -> QuantumCircuit:
    """Data encoding circuit without entanglement. The 'type' argument
    corresponds the two No-Entanglement (NE) circuits in the paper.

    Parameters
    ----------
    nqubits : int, optional
        Number of qubits for the circuit, by default 8
    reps : int, optional
        Number of repetition of the data encoding circuit, by default 3

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Quantum circuit to be used as a feature map for quantum the kernel machine.
    """
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)

        if type == 1:
            for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
                qc.u(x[feature], x[feature + 1], 0, qubit)
    return qc
