# Main script for the training qsvm.

from time import perf_counter
from typing import Callable

from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np

from sklearn.svm import SVC

from terminal_enhancer import tcols

import util
import test
import data_processing
from feature_map_circuits import u_dense_encoding

seed = 12345 
algorithm_globals.random_seed = seed


def main(args):
    train_loader, test_loader = data_processing.get_data(args)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]

    model = util.init_kernel_machine(args)
    out_path = util.create_output_folder(args, model)
    
    time_and_train(model.fit, train_features, train_labels)
    util.print_model_info(model)
    
    print("Computing the test dataset accuracy of the models, quick check"
          " for overtraining...")
    test_time_init = perf_counter()
    if args["quantum"]: # TODO could this be more elegant?
        np.save(out_path + "/kernel_matrix_elements", model.kernel_matrix_train)
        train_acc = model.score(train_features, train_labels, train_data=True)
    else: 
        train_acc = model.score(train_features, train_labels)
    test_acc = model.score(test_features, test_labels)
    test_time_fina = perf_counter()
    exec_time = test_time_fina - test_time_init  
    print(f"Completed in: {exec_time:.2e} sec. or "f"{exec_time/60:.2e} min. "
          + tcols.ROCKET)
    util.print_accuracy_scores(test_acc, train_acc)
    util.save_qsvm(model, out_path)
   
    """
    #FIXME
    util.export_hyperparameters() to include the functions below and the
    np.save(kernel)
    qc_transpiled = util.get_quantum_kernel_circuit(kernel, out_path)
        
    if backend is not None:
        util.save_circuit_physical_layout(qc_transpiled, backend, out_path)
        util.save_backend_properties(backend, out_path + "/backend_properties_dict")
    """

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
    print("Training completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
          f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS)
    