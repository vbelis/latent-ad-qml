# Module where all tested quantum circuits are defined using qiskit.

from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np

def u_dense_encoding(nqubits=8) -> QuantumCircuit:
    """
    Constructs a feature map, inspired by the dense encoding method.
    @nqubits   :: Int number of qubits used.

    returns :: The quantum circuit object form qiskit.
    """
    # TODO test with Hadamards in the beggining.
    nfeatures = 2*nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
        qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
    for i in range(nqubits):
        if i == nqubits - 1:
            break
        qc.cx(i, i + 1)
    for feature, qubit in zip(range(2 * nqubits, nfeatures, 2), range(nqubits)):
        qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)

    for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
        qc.u(x[feature], x[feature + 1], 0, qubit)

    return qc
