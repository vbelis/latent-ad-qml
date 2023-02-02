# Module where all tested quantum circuits are defined using qiskit.

from qiskit.circuit import QuantumCircuit, ParameterVector
from itertools import combinations
import numpy as np


def u_dense_encoding(nqubits: int = 8, reps: int = 3) -> QuantumCircuit:
    """
    Constructs a feature map, inspired by the dense encoding and data
    re-uploading methods.

    Args:
        nqubits: Int number of qubits used.

    Returns: The quantum circuit qiskit object used in the QSVM algorithm.
    """
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            #qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
        
        for i in range(nqubits):
            if i == nqubits - 1:
                break
            qc.cx(i, i + 1)

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return qc


def u_dense_encoding_all(nqubits: int = 8, reps: int = 3) -> QuantumCircuit:
    """
    Constructs a feature map, inspired by the dense encoding and data
    re-uploading methods.

    Args:
        nqubits: Int number of qubits used.

    Returns: The quantum circuit qiskit object used in the QSVM algorithm.
    """
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            #qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
        
        for qpair in list(combinations(range(nqubits), 2)):
            qc.cx(qpair[0], qpair[1])

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return qc

def u_dense_encoding_no_ent(nqubits: int = 8, reps: int = 3) -> QuantumCircuit:
    """
    Constructs a feature map, inspired by the dense encoding and data
    re-uploading methods.

    Args:
        nqubits: Int number of qubits used.

    Returns: The quantum circuit qiskit object used in the QSVM algorithm.
    """
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
        
        # Comment the below if only one layer is needed for the no entanglement test
        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return qc


def ry_rz(nqubits=8, lin_entanglement=True, reps=1):
    """TODO"""
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            qc.ry(theta=x[feature], qubit=qubit,)
            qc.rz(phi=x[feature + 1], qubit=qubit)
        if lin_entanglement:
            for i in range(nqubits):
                if i == nqubits - 1:
                    break
                qc.cx(i, i + 1)
    return qc

def ry_rz_ent(nqubits=8, reps=1):
    """TODO"""
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
        qc.ry(theta=x[feature], qubit=qubit,)
        qc.rz(phi=x[feature + 1], qubit=qubit)
    
    for i in range(nqubits):
        if i == nqubits - 1:
            break
        qc.cx(i, i + 1)
    
    for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
        qc.ry(theta=x[feature + 1], qubit=qubit)
        qc.rz(phi=x[feature], qubit=qubit)
    return qc

def ry_rx_ent(nqubits=8, reps=1):
    """TODO"""
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
        qc.ry(theta=x[feature], qubit=qubit,)
        qc.rx(theta=x[feature + 1], qubit=qubit)
    
    for i in range(nqubits):
        if i == nqubits - 1:
            break
        qc.cx(i, i + 1)
    
    for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
        qc.ry(theta=x[feature + 1], qubit=qubit)
        qc.rx(theta=x[feature], qubit=qubit)
    return qc

def ry_rz_seq(nqubits=8,):
    """TODO"""
    nfeatures = 2 * nqubits
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for i_feature_qubit in range(nqubits):
        qc.ry(theta=x[i_feature_qubit], qubit=i_feature_qubit)
        
    for i in range(nqubits):
        if i == nqubits - 1:
            break
        qc.cx(i, i + 1)
    
    for i_feature_qubit in range(nqubits):
        # encode the rest of the features 
        qc.rz(phi=x[nqubits + i_feature_qubit], qubit=i_feature_qubit)
    return qc
