import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator, Pauli


def diffuser(n):
    """Subcircuit for Diffuser of Grover algorithm.

    Parameters
    ----------
    n : int
        number of qubits

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        circuit for Grover diffuser
    """

    qc = QuantumCircuit(n)

    for q in range(n):
        qc.h(q)
        qc.x(q)

    qc.h(n - 1)
    qc.mct(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    for q in range(n):
        qc.x(q)
        qc.h(q)

    # Convert diffuser to gate
    diff_gate = qc.to_gate()
    diff_gate.name = "diffuser"

    return diff_gate


def grover_circuit(n, oracle, marked_n=1):
    """Full Grover quantum circuit.

    Parameters
    ----------
    n : int
        number of qubits
    oracle : :class:`qiskit.circuit.QuantumCircuit`
        the preconstructed oracle circuit
    marked_n : int, optional
        number of marked entries, default 1

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Grover circuit.
    """

    qc = QuantumCircuit(n, n)

    r = int(np.floor(np.pi / 4 * np.sqrt(2**n / marked_n))) if marked_n else 0

    for q in range(n):
        qc.h(q)

    for _ in range(r):
        # add oracle
        qc.append(oracle, range(n))
        # add diffuser
        qc.append(diffuser(n), range(n))

    qc.measure(range(n), range(n))

    return qc
