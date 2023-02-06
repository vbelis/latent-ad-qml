import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator, Pauli


def diffuser(n):

    qc = QuantumCircuit(n)

    # Apply a H-gate to all qubits (transformation |s> -> |00..0>)
    # Followed by X-gate (transformation |00..0> -> |11..1>)
    for q in range(n):
        qc.h(q)
        qc.x(q)

    # Do multi-controlled-Z gate
    qc.h(n - 1)
    qc.mct(list(range(n - 1)), n - 1)  # multi-controlled-toffoli
    qc.h(n - 1)

    # Apply transformation |11..1> -> |00..0>
    # Apply transformation |00..0> -> |s>
    for q in range(n):
        qc.x(q)
        qc.h(q)

    # Convert diffuser to gate
    diff_gate = qc.to_gate()
    diff_gate.name = "diffuser"

    return diff_gate


def grover_circuit(n, oracle, marked_n=1):

    qc = QuantumCircuit(n, n)

    # Determine r
    r = int(np.floor(np.pi / 4 * np.sqrt(2**n / marked_n))) if marked_n else 0

    # Apply a H-gate to all qubits
    for q in range(n):
        qc.h(q)

    for _ in range(r):
        # add oracle
        qc.append(oracle, range(n))
        # add diffuser
        qc.append(diffuser(n), range(n))

    # step 3: measure all qubits
    qc.measure(range(n), range(n))

    return qc
