import numpy as np
from qiskit import Aer, IBMQ, execute, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

shots_n = 1000


def normalize(v):
    """Normalize real vector.

    Parameters
    ----------
    v : :class:`numpy.ndarray`
        real vector

    Returns
    -------
    :class:`numpy.ndarray`
        normalized vector
    """
    return v / np.linalg.norm(v)


def calc_z(a, b) -> float:
    """Calculates magnitude as :math: `{z = |a|^2 + |b|^2}`.

    Parameters
    ----------
    a : float
        first classic input
    b : float
        second classic input

    Returns
    -------
    float
        magnitude
    """

    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    return a_mag**2 + b_mag**2


def psi_amp(a, b):
    """Prepares amplitudes for encoding state :math:`{\psi}`.

    Parameters
    ----------
    a : float
        first classic input
    b : float
        second classic input

    Returns
    -------
    List
        list of amplitudes (floats)
    """

    a_norm = normalize(a)
    b_norm = normalize(b)

    # import ipdb; ipdb.set_trace()

    return np.hstack([a_norm, b_norm]) * (1 / np.sqrt(2))


def phi_amp(a, b):
    """Prepares amplitudes for encoding state :math:`{\phi}`.

    Parameters
    ----------
    a : float
        first classic input
    b : float
        second classic input

    Returns
    -------
    List
        list of amplitudes (floats)
    """

    z = calc_z(a, b)
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)

    return np.hstack([a_mag, -b_mag]) / np.sqrt(z)


def psi_circuit(a, b) -> QuantumCircuit:
    """Subcircuit for state :math:`{\psi}`.

    Parameters
    ----------
    a : float
        first classic input
    b : float
        second classic input

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Quantum circuit for state :math:`{\psi}`
    """

    amp = psi_amp(a, b)
    sz = int(np.log2(len(amp)))

    qc = QuantumCircuit(sz)

    qc.initialize(amp, range(sz))

    return qc


def phi_circuit(a, b) -> QuantumCircuit:
    """Subcircuit for state :math:`{\phi}`.

    Parameters
    ----------
    a : float
        first classic input
    b : float
        second classic input

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Quantum circuit for state :math:`{\phi}`
    """

    amp = phi_amp(a, b)
    sz = 1
    qc = QuantumCircuit(sz)

    qc.initialize(amp, [0])

    return qc


def overlap_circuit(a, b) -> QuantumCircuit:
    """Full overlap circuit <:math:`{\phi} | {\psi}`>.

    Parameters
    ----------
    a : float
        first classic input
    b : float
        second classic input

    Returns
    -------
    :class:`qiskit.circuit.QuantumCircuit`
        Quantum circuit that calculates overlap of encoded quantum states.
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
    qc.cswap(0, qr_psi[-1], qr_phi[0])
    qc.h(0)

    qc.measure(0, 0)

    return qc


def run_circuit(qc):
    """Utility function that runs quantum circuit

    Parameters
    ----------
    qc : :class:`qiskit.circuit.QuantumCircuit`
        the quantum circuit

    Returns
    -------
    List
        counts of measurements
    """
    simulator = Aer.get_backend("qasm_simulator")
    return execute(qc, backend=simulator, shots=shots_n).result().get_counts(qc)


def calc_overlap(answer, state="0"):
    """Utility function that calculates overlap from measurement results.

    Parameters
    ----------
    answer : List
        counts of measurements.
    state: string, optional
        state that captures distance, by default 0

    Returns
    -------
    float
        the overlap
    """
    shots = answer[state] if len(answer) == 1 else answer["0"] + answer["1"]
    return np.abs(answer[state] / shots_n - 0.5) * 2


def calc_dist(answer, z, state="0"):
    """Utility function that calculates the distance from overlap proportional to :math:`{|a-b|^2}`.

    Parameters
    ----------
    answer : List
        counts of measurements
    z : float
        magnitude
    state : string, optional
        state that captures distance, by default 0

    Returns
    -------
    float
        the distance
    """
    return calc_overlap(answer, state) * 2 * z
