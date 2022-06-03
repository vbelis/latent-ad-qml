from time import perf_counter
import h5py
import numpy as np
from sklearn.svm import SVC
from qiskit_machine_learning.kernels import QuantumKernel
from typing import Tuple
from sklearn import metrics

import util
import data_processing
from terminal_colors import tcols
from feature_map_circuits import u_dense_encoding

def main(args):
    train_loader, test_loader = data_processing.get_data(args)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]
    sig_fold, bkg_fold = data_processing.get_kfold_data(test_features,
                                                        test_labels,)
    qsvm = util.load_qsvm(args["model"] + "model")
    # TODO would be nice in to pass the feature map as an argument as well and
    # save it as a hyperparameter of the QSVM model in the .json file.
    # model = util.load_model(path)
    feature_map = u_dense_encoding(nqubits=args["nqubits"])

    quantum_instance, backend = util.configure_quantum_instance(
        ibmq_api_config=args["ibmq_api_config"],
        run_type=args["run_type"],
        backend_name=args["backend_name"],
        **args["config"],
    )
    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    print("\nFor the signal folds: ")
    score_sig = compute_qsvm_scores(qsvm, kernel, train_features, sig_fold,)
    print("For the background folds: ")
    score_bkg = compute_qsvm_scores(qsvm, kernel, train_features, bkg_fold,)
    output_path = args["model"] 
    #+ args["output"]
    print(f"Saving the sig and bkg in the folder: " + tcols.OKCYAN 
          + f"{output_path}" + tcols.ENDC)
    np.save(output_path + "sig_scores.npy", score_sig)
    np.save(output_path + "bkg_scores.npy", score_bkg)
    

def compute_qsvm_scores(
    model:SVC, 
    kernel: QuantumKernel, 
    x_train: np.ndarray, 
    data_folds: np.ndarray, 
) -> np.ndarray:
    """
    Computing the model scores on all the test data folds to construct
    performance metrics of the model, e.g., ROC curve and AUC.

    Args:
        model: The qsvm model to compute the score for.
        kernel: The quatum kernel of the QSVM.
        x_train: Training data array of the saved QSVM.
        data_folds: Numpy array of kfolded data.

    Returns:
        Array of the qsvm scores obtained.
    """
    print("Computing QSVM scores... ", end="")
    scores_time_init = perf_counter()
    model_scores = np.array(
        [
            model.decision_function(kernel.evaluate(x_vec=fold, y_vec=x_train))
            for fold in data_folds
        ]
    )
    scores_time_fina = perf_counter()
    exec_time = scores_time_fina - scores_time_init
    print(
        tcols.OKGREEN
        + "Completed in: "
        + tcols.ENDC
        + f"{exec_time:2.2e} sec. or {exec_time/60:2.2e} min. " + tcols.ROCKET
    )
    return model_scores

def compute_svm_scores(
    model:SVC, 
    data_folds: np.ndarray, 
    output_folde:str =None,
) -> np.ndarray:
    """
    Compute the scores for the SVM model for the classical benchmark.
    Args:
        model: The trained classical SVM model.
        data_folds: An array of shape (k, n_test) containg the k-folded test
                    dataset.
        output_folder: Path to save the model scores in .npy format.
    Returns: 
        An numpy array of the scores with shape (k, n_test).
    """
    print("\nComputing SVM scores...")
    scores_time_init = perf_counter()
    scores_time_fina = perf_counter()
    exec_time = scores_time_fina - scores_time_init
    pass #TODO

def accuracy_from_scores(
    scores: np.ndarray, 
    truth_labels: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """ # TODO will make use of it if computing the decision function during training
    and computing the k-folds in one go.

    Given the decision_function scores of a model it produces the predicted 
    label based on the sign of the score (one side for the decision boundary
    vs. the other). Then, using the predictions it calculates the accuracy.

    Args:
        scores: 1d array with the scores to be transformed.
    Returns:
        A tuple of an array with 1's (sig) and 0's (bkg), if the score is positive and
        negative, respectively, and the accuracy of the model. The former return object
        is equivalent to SVC.predict.
    """
    scores[scores>0] = 1
    scores[scores<0] = 0
    accuracy = metrics.accuracy_score(truth_labels, scores)
    return accuracy, scores
