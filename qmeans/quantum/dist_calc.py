import numpy as np

# import Qiskit
from qiskit import Aer, IBMQ, execute, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

shots_n = 1000


def normalize(v):
    return v / np.linalg.norm(v)


def calc_z(a, b) -> float:
    """z = |a|**2 + |b|**2"""
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    return a_mag**2 + b_mag**2


def psi_amp(a, b):
    """prepare amplitudes for state psi"""

    a_norm = normalize(a)
    b_norm = normalize(b)

    # import ipdb; ipdb.set_trace()

    return np.hstack([a_norm, b_norm]) * (1 / np.sqrt(2))


def phi_amp(a, b):
    """prepare amplitudes for state phi"""

    z = calc_z(a, b)
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)

    return np.hstack([a_mag, -b_mag]) / np.sqrt(z)


def psi_circuit(a, b):

    amp = psi_amp(a, b)  # 2*n amplitudes 1/sqrt(2) (a0, ..., an, b0, ..., bn)
    sz = int(np.log2(len(amp)))

    qc = QuantumCircuit(sz)  # 2 qubits if a,b in R^2

    qc.initialize(amp, range(sz))

    return qc


def phi_circuit(a, b) -> QuantumCircuit:
    """prepare subcircuit for state phi"""

    amp = phi_amp(a, b)  # 2 amplitudes 1/sqrt(z) (|a|, |b|)
    sz = 1  # always 2 amplitudes

    qc = QuantumCircuit(sz)  # 2 qubits if a,b in R^2

    qc.initialize(amp, [0])

    return qc


def overlap_circuit(a, b) -> QuantumCircuit:
    """
    full overlap circuit < phi | psi >
    a,b: real inputs
    """
    n = len(a)
    if not ((n & (n - 1) == 0) and n != 0):
        raise ValueError("size of input vectors must be power of 2 but is " + str(n))

    psi = psi_circuit(a, b)
    phi = phi_circuit(a, b)

    anc = QuantumRegister(1, "ancilla")
    qr_psi = QuantumRegister(psi.width(), "psi")
    qr_phi = QuantumRegister(phi.width(), "phi")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(anc, qr_psi, qr_phi, cr)

    qc.append(psi, qr_psi[:])
    qc.append(phi, qr_phi[:])

    qc.barrier()

    qc.h(0)
    qc.cswap(0, qr_psi[-1], qr_phi[0])  # perform test on psi ancilla alone
    qc.h(0)

    qc.measure(0, 0)

    return qc


def run_circuit(qc):
    simulator = Aer.get_backend("qasm_simulator")
    return execute(qc, backend=simulator, shots=shots_n).result().get_counts(qc)


def calc_overlap(answer, state="0"):
    """calculate overlap from experiment measurements"""

    shots = answer[state] if len(answer) == 1 else answer["0"] + answer["1"]
    return np.abs(answer[state] / shots_n - 0.5) * 2


def calc_dist(answer, z, state="0"):
    """calculate distance proportional to |a-b|**2"""
    return calc_overlap(answer, state) * 2 * z
