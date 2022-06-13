# Main script for the training QSVM and SVM models.

from time import perf_counter
from typing import Callable
from qiskit.utils import algorithm_globals
import numpy as np

import util
import data_processing
from terminal_enhancer import tcols

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
    
    util.overfit_xcheck(model, train_features, train_labels, 
                        test_features, test_labels)
    util.save_model(model, out_path)
   
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
    