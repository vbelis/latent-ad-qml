from time import perf_counter

from .terminal_colors import tcols
import numpy as np
from . import qdata as qd
from . import plot
from . import util
from .feature_map_circuits import u2Reuploading

from qiskit_machine_learning.kernels import QuantumKernel


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["model_path"],
        train_events=args["ntrain"],
        valid_events=args["nvalid"],
        test_events=args["ntest"],
        kfolds=args["kfolds"],
        seed=args["seed"],
    )
    train_features = qdata.get_latent_space("train")
    test_folds, test_folds_labels = qdata.get_kfolded_data("test")

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

    scores = compute_model_scores(
        qsvm, kernel, train_features, test_folds, args["qsvm_model"]
    )
    plot.roc_plot(
        scores, qdata, test_folds_labels, args["qsvm_model"], args["display_name"]
    )


def compute_model_scores(
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
            #model.decision_function(fold)
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
