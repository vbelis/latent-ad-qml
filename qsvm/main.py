# Main script for the training qsvm.

from time import perf_counter
from typing import Callable

from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np

from sklearn.svm import SVC

from terminal_colors import tcols

import util
import test
import preprocessing
from feature_map_circuits import u_dense_encoding

seed = 12345 
algorithm_globals.random_seed = seed


def main(args):
    train_loader, test_loader = preprocessing.get_data(args)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]

    feature_map = u_dense_encoding(nqubits=args["nqubit"])
    quantum_instance, backend = util.configure_quantum_instance(
        ibmq_api_config=args["ibmq_api_config"],
        run_type=args["run_type"],
        backend_name=args["backend_name"],
        **args["config"],
    )
    kernel = QuantumKernel(feature_map, quantum_instance)
    
    print("Calculating the quantum kernel matrix elements... ", end="")
    train_time_init = perf_counter()
    quantum_kernel_matrix = kernel.evaluate(x_vec=train_features)
    train_time_fina = perf_counter()
    print(
        tcols.OKGREEN + f"Done in: {train_time_fina-train_time_init:.2e} s" + tcols.ENDC
    )

    qsvm = SVC(kernel="precomputed", C=args["c_param"])
    out_path = util.create_output_folder(args, qsvm)
    np.save(out_path + "/kernel_matrix_elements", quantum_kernel_matrix)
    
    time_and_train(qsvm.fit, quantum_kernel_matrix, train_labels)
    util.print_model_info(qsvm)
    
    print("Computing the test dataset accuracy of the models, quick check"
          " for overtraining...")
    test_time_init = perf_counter()
    kernel_matrix_test = kernel.evaluate(x_vec=test_features, 
                                     y_vec=train_features)
    train_acc = qsvm.score(quantum_kernel_matrix, train_labels)
    test_acc = qsvm.score(kernel_matrix_test, test_labels)
    test_time_fina = perf_counter()
    exec_time = test_time_fina - test_time_init  
    print(f"Completed in: {exec_time:.2e} sec. or "f"{exec_time/60:.2e} min. "
          + tcols.ROCKET)
    util.print_accuracy_scores(test_acc, train_acc)
    util.save_qsvm(qsvm, out_path)
    qc_transpiled = util.get_quantum_kernel_circuit(kernel, out_path)
    
    #if args["compute_kfolds"]: 
    #    test.main()
    # TODO do the k-folding here in one go -> .h5 for ROC plotting with Kinga's script.

    if backend is not None:
        util.save_circuit_physical_layout(qc_transpiled, backend, out_path)
        util.save_backend_properties(backend, out_path + "/backend_properties_dict")

def time_and_train(fit: Callable, *args):
    """
    Trains and computes the training runtime of the qsvm model.
    Args:
        fit: The training function object of the QSVM.
        *args: Arguments required by the `fit` method.
    """
    print("Training the QSVM... ", end="")
    train_time_init = perf_counter()
    fit(*args)
    train_time_fina = perf_counter()
    exec_time = train_time_fina-train_time_init
    print(tcols.OKGREEN +  f"Training completed in: " + tcols.ENDC +
          f"{exec_time:.2e} sec. or {exec_time/60:.2e} min.")
    