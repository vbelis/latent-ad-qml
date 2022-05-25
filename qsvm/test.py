from time import perf_counter

from terminal_colors import tcols
import numpy as np
import plot
import util
from sklearn.svm import SVC
from qiskit_machine_learning.kernels import QuantumKernel
from typing import Tuple
from sklearn import metrics

from feature_map_circuits import u2Reuploading

def main(args):
    train_loader, test_loader = util.get_data(args)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]
    test_folds = [test_features] # FIXME implement code to do k-folds from .h5

    qsvm = util.load_qsvm(args["qsvm_model"] + "model")
    # TODO would be nice in to pass the feature map as an argument as well and
    # save it as a hyperparameter of the QSVM model in the .json file.
    feature_map = u2Reuploading(nqubits=8, nfeatures=args["feature_dim"])

    quantum_instance, backend = util.configure_quantum_instance(
        ibmq_api_config=args["ibmq_api_config"],
        run_type=args["run_type"],
        backend_name=args["backend_name"],
        **args["config"],
    )
    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    scores = compute_qsvm_scores(
        qsvm, kernel, train_features, test_folds, args["qsvm_model"]
    )
    #plot.roc_plot(
    #    scores, qdata, test_folds_labels, args["qsvm_model"], args["display_name"]
    #)


def compute_qsvm_scores(
    model, kernel, x_train, data_folds, output_folder=None,
) -> np.ndarray:
    """
    Computing the model scores on all the test data folds to construct
    performance metrics of the model, e.g., ROC curve and AUC.

    @model (svm.SVC)         :: The qsvm model to compute the score for.
    @kernel (QuantumKernel)  :: The quatum kernel of the QSVM.
    @x_train (np.ndarray)    :: Training data array of the saved QSVM.
    @data_folds (np.ndarray) :: Numpy array of kfolded data.
    @output_folder :: The folder where the results are saved.

    returns :: Array of the qsvm scores obtained.
    """
    print("\nComputing QSVM scores...")
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
        + f"{exec_time:2.2e} sec. or {exec_time/60:2.2e} min."
    )

    if output_folder is not None:
        path = output_folder + "/y_score_list.npy"
        print("Saving model scores array in: " + path)
        np.save(path, model_scores)

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
    truth_labels,
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
        negative, respectively, and the accuracy of the model.
    """
    scores[scores>0] = 1
    scores[scores<0] = 0
    accuracy = metrics.accuracy_score(truth_labels, scores)
    return accuracy, scores
