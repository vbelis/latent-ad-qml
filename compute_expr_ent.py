# Compute the expressibility and entanglement capability metrics of a given circuit

from time import perf_counter
import numpy as np
import pandas as pd
from itertools import combinations
import argparse
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from triple_e import expressibility
from triple_e import entanglement_capability

from kernel_machines.terminal_enhancer import tcols
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # For pandas frame.append method

np.random.seed(42)


def main(args):
    n_params = 2*args["n_qubits"]
    df_expr_ent = pd.DataFrame(columns=['circuit', 'expr', 'expr_err', 
                               'ent', 'ent_err'])
    circuit_list_expr_ent = [lambda x, rep=rep: u_dense_encoding(x, nqubits=\
                             args["n_qubits"], reps=rep) for rep in range(1,7)]
    # Add the two no-entanglement circuits (NE_0, NE_1) at the beginning of the list
    circuit_list_expr_ent.insert(
        0,
        lambda x: u_dense_encoding_no_ent(x, nqubits=args["n_qubits"], reps=1, type=1)
    )
    circuit_list_expr_ent.insert(
        0,
        lambda x: u_dense_encoding_no_ent(x, nqubits=args["n_qubits"], reps=1, type=0)
    )
    # Add the full ent rep3 (FE)
    circuit_list_expr_ent.insert(
        -1,
        lambda x: u_dense_encoding_all(x, nqubits=args["n_qubits"], reps=3)
    )
    circuit_label = ['NE_0', 'NE_1', 'L=1', 'L=2', 'L=3', 'L=4', 'L=5', 'L=6', 'FE']
    print("\nComputing expressibility and entanglement capability of the circuits, "
          f"for {args['n_exp']} evaluations and n_shots = {args['n_shots']}." 
          f"\nCircuits: {circuit_label}")
    
    # Expr. and Ent. as a function of the circuit depth and amount of CNOT gates
    train_time_init = perf_counter()
    for i, circuit in enumerate(circuit_list_expr_ent):
        print(f"\nFor circuit {circuit_label[i]}...")
        expr = []
        
        # Compute entanglement cap. only once, it fluctuates less than 0.1%
        # and does not require that high statistics
        if circuit_label[i] in ['NE_0', 'NE_1']:
            val_ent = 0 # by construction
        else:
            val_ent = entanglement_capability(circuit, n_params, n_shots=1000)
        
        for _ in range(args["n_exp"]):
            val_ex = expressibility(circuit, n_params, method='full', n_shots=args['n_shots'], n_bins=75)
            expr.append(val_ex)
        expr = np.array(expr)
        ent = np.array(val_ent)
        print(f"expr = {np.mean(expr)} ± {np.std(expr)}")
        print(f"ent = {np.mean(ent)} ± {np.std(ent)}")
        
        d = {
            "circuit": circuit_label[i],
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
    # TODO Expr. vs. n_qubits. (For exponential concentration)
    # TODO variance of the kernel matrix vs. n_qubits (For exponential concentration)


def u_dense_encoding_no_ent(x, nqubits: int = 8, reps: int = 3, type: int = 0
) -> QuantumCircuit:
    """
    Constructs a feature map based on u_dense_encoding but removing entanglement.
    The 'type' argument corresponds the two No-Entanglement (NE) circuits in the paper.

    Args:
        nqubits: Int number of qubits used.

    Returns: The quantum circuit qiskit object used in the QSVM algorithm.
    """
    nfeatures = 2 * nqubits
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            #qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
        
        # Comment the below if only one layer is needed for the no entanglement test
        if type == 0: 
            for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
                qc.u(x[feature], x[feature + 1], 0, qubit)
        
    return Statevector.from_instruction(qc)


def u_dense_encoding(x, nqubits=8, reps=1,):
    nfeatures = 2 * nqubits
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
    return Statevector.from_instruction(qc)


def u_dense_encoding_all(x, nqubits: int = 8, reps: int = 3) -> QuantumCircuit:
    """
    Constructs a feature map, inspired by the dense encoding and data
    re-uploading methods.

    Args:
        nqubits: Int number of qubits used.

    Returns: The quantum circuit qiskit object used in the QSVM algorithm.
    """
    nfeatures = 2 * nqubits
    qc = QuantumCircuit(nqubits)
    for rep in range(reps):
        for feature, qubit in zip(range(0, nfeatures, 2), range(nqubits)):
            #qc.h(qubit)
            qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
        
        for qpair in list(combinations(range(nqubits), 2)):
            qc.cx(qpair[0], qpair[1])

        for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
            qc.u(x[feature], x[feature + 1], 0, qubit)
    return Statevector.from_instruction(qc)

'''
print("computing entanglement capability")
for rep in range(20):
    if rep == 0: continue
    ent = []
    print(f"L={rep}: ", end='')
    for i in range(6):
        val = entanglement_capability(lambda x : u_dense_encoding(x, nqubits=n_qubits, reps=rep), 
                    n_params, n_shots=n_shots)
        ent.append(val)
        print(f"{i}: expr = {val}")
    ent = np.array(ent)
    print(f"ent = {np.mean(ent)} ± {np.std(ent)}")   

    for n in range(1,12):
        print(f"For n = {n}: ", end='')
        n_params = 2*n
        val, hist = expressibility(lambda x : u_dense_encoding(x, nqubits=n, reps=reps), 
                             n_params, n_shots=n_shots, n_bins=75, return_histogram=True)
        expr.append(val)
        print(f"expr = {val}")
    expr = np.array(expr)
    np.save("expr_qubits.npy", expr)
    print(expr)
    print(f"histogram values: {hist}")
n_qubits = 8
reps = 2
n_params = 2*n_qubits
n_shots = int(10000)

#print("computing expressibility")
#for rep in range(7):
#    if rep == 0: continue
#    expr = []
#    print(f"L={rep}: ", end='')
#    for i in range(20):
#        val = expressibility(lambda x : u_dense_encoding(x, nqubits=n_qubits, reps=rep), 
#                    n_params, n_shots=n_shots, n_bins=75)
#        expr.append(val)
#        print(f"{i}: expr = {val}")
#    expr = np.array(expr)
#    print(f"expr = {np.mean(expr)} ± {np.std(expr)}")
#
#print("computing entanglement capability")
#for rep in range(20):
#    if rep == 0: continue
#    ent = []
#    print(f"L={rep}: ", end='')
#    for i in range(6):
#        val = entanglement_capability(lambda x : u_dense_encoding(x, nqubits=n_qubits, reps=rep), 
#                    n_params, n_shots=n_shots)
#        ent.append(val)
#        print(f"{i}: expr = {val}")
#    ent = np.array(ent)
#    print(f"ent = {np.mean(ent)} ± {np.std(ent)}")   

#rep=3
#print(f"Computing expressibility as a function of the qubit number, for rep = {rep} of the ansatz...")
#expr = []
#hist = 0
#for n in range(16,17):
#    print(f"For n = {n}: ", end='')
#    n_params = 2*n
#    val, hist = expressibility(lambda x : u_dense_encoding(x, nqubits=n, reps=rep), 
#                         n_params, n_shots=n_shots, n_bins=75, return_histogram=True)
#    expr.append(val)
#    print(f"expr = {val}")
#expr = np.array(expr)
#np.save("expr_qubits.npy", expr)
#print(expr)
#print(f"histogram values: {hist}")
'''

def get_arguments() -> dict:
    """
    Parses command line arguments and gives back a dictionary.
    Returns: Dictionary with the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    args = parser.parse_args()
    args = {
        "n_qubits": args.n_qubits,
        "n_shots": args.n_shots,
        "n_exp": args.n_exp,
        "out_path": args.out_path,
    }

    return args

if __name__ == "__main__":
    args = get_arguments()
    main(args)
