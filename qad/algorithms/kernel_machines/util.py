# Utility methods for the quantum kernel machine training and testing.

import os
import joblib
import re
import json
import numpy as np
from time import perf_counter
from typing import Tuple, Union
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from qad.algorithms.kernel_machines.qsvm import QSVM
from qad.algorithms.kernel_machines.one_class_svm import CustomOneClassSVM
from qad.algorithms.kernel_machines.one_class_qsvm import OneClassQSVM
from qad.algorithms.kernel_machines.terminal_enhancer import tcols


def print_accuracy_scores(test_acc: float, train_acc: float, is_unsup: bool):
    """Prints the train and test accuracies of the model.

    Parameters
    ----------
    test_acc : float
        The accuracy of the trained model on the test dataset.
    train_acc : float
        The accuracy of the trained model on the train dataset.
    is_unsup : bool
        Flag if the model is unsupervised. The printing is slighlty different if so.
    """
    if is_unsup:
        print(tcols.OKGREEN + f"Fraction of outliers in the traning set = {train_acc}")
    else:
        print(tcols.OKGREEN + f"Training accuracy = {train_acc}")
    print(f"Testing accuracy = {test_acc}" + tcols.ENDC)


def create_output_folder(
    args: dict, model: Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM]
) -> str:
    """Creates output folder for the given model name and arguments.

    Parameters
    ----------
    args : dict
        Arparse configuration arguments dictionary.
    model : Union[:class:`sklearn.svm.SVC`, `QSVM`, `CustomOneClassSVM`, `OneClassQSVM`]
        Kernel machine, classical or quantum.

    Returns
    -------
    str
        The output path.
    """

    if args["unsup"]:
        out_path = args["output_folder"] + f"_nu={model.nu}"
    else:
        out_path = args["output_folder"] + f"_c={model.C}"
    if args["quantum"]:
        out_path = out_path + f"_{args['run_type']}"
        if args["backend_name"] is not None and args["backend_name"] != "none":
            # For briefness remove the "ibmq" prefix for the output folder:
            backend_name = re.sub("ibmq?_", "", args["backend_name"])
            out_path += f"_{backend_name}"
    out_path = "trained_qsvms/" + out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def save_model(model: Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM], path: str):
    """Saves the qsvm model to a certain path.

    Parameters
    ----------
    model : Union[:class:`sklearn.svm.SVC`, `QSVM`, `CustomOneClassSVM`, `OneClassQSVM`]
        Kernel machine that we want to save.
    path : str
        Path to save the model in.
    """
    if isinstance(model, QSVM) or isinstance(model, OneClassQSVM):
        np.save(path + "/train_kernel_matrix.npy", model.kernel_matrix_train)
        qc_transpiled = model.get_transpiled_kernel_circuit(path)
        if model.backend is not None:
            model.save_circuit_physical_layout(qc_transpiled, path)
            model.save_backend_properties(path)
    joblib.dump(model, path + "/model")
    print("Trained model and plots saved in: " + tcols.OKCYAN + path + tcols.ENDC)


def load_model(path: str) -> Union[QSVM, SVC, CustomOneClassSVM, OneClassQSVM]:
    """Load model from pickle file, i.e., deserialisation.

    Parameters
    ----------
    path : str
        String of full path to load the model from.

    Returns
    -------
    Union[`QSVM`, :class:`sklearn.svm.SVC`, `CustomOneClassSVM`, `OneClassQSVM`]
        Joblib object of the trained model.
    """
    return joblib.load(path)


def print_model_info(model: Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM]):
    """Print information about the trained model, such as the C parameter value,
    number of support vectors, number of training and testing samples.

    Parameters
    ----------
    model : Union[:class:`sklearn.svm.SVC`, `QSVM`, `CustomOneClassSVM`, `OneClassQSVM`]
        The trained kernel machine.
    """
    print("\n-------------------------------------------")
    if isinstance(model, SVC):  # Check if it is a supervised SVM or a QSVM
        print(
            f"C = {model.C}\n"
            f"For classes: {model.classes_}, the number of support vectors for "
            f"each class are: {model.n_support_}"
        )
    else:
        print(f"nu = {model.nu}\n" f"Number of support vectors: {model.n_support_}")
    print("-------------------------------------------\n")


def init_kernel_machine(
    args: dict,
) -> Union[SVC, QSVM, CustomOneClassSVM, OneClassQSVM]:
    """Initialises the kernel machine. Depending on the flag, this will be
    a `SVM` or a `QSVM`.

    Parameters
    ----------
    args : dict
        The argument dictionary defined in the training script.

    Returns
    -------
    Union[:class:`sklearn.svm.SVC`, `QSVM`, `CustomOneClassSVM`, `OneClassQSVM`]
        The kernel machine model.
    """
    if args["quantum"]:
        if args["unsup"]:
            print(
                tcols.OKCYAN + "\nConfiguring the one-class Quantum Support Vector"
                " Machine." + tcols.ENDC
            )
            return OneClassQSVM(args)
        print(
            tcols.OKCYAN + "\nConfiguring the Quantum Support Vector"
            " Machine." + tcols.ENDC
        )
        return QSVM(args)

    if args["unsup"]:
        print(
            tcols.OKCYAN + "\nConfiguring the one-class Classical Support Vector"
            " Machine..." + tcols.ENDC
        )
        return CustomOneClassSVM(
            kernel=args["feature_map"], nu=args["nu_param"], gamma=args["gamma"]
        )
    print(
        tcols.OKCYAN + "\nConfiguring the Classical Support Vector"
        " Machine..." + tcols.ENDC
    )
    return SVC(kernel=args["feature_map"], C=args["c_param"], gamma=args["gamma"])


def eval_metrics(
    model: Union[QSVM, SVC, CustomOneClassSVM, OneClassQSVM],
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    out_path: str,
):
    """Computes different evaluation metrics of the kernel machine models.
    ROC curve, FPR @TPR (working point), and accuracy. Prints and saves
    corresponding plots. The execution of this function is also timed.

    Parameters
    ----------
    model : Union[`QSVM`, :class:`sklearn.svm.SVC`, `CustomOneClassSVM`, `OneClassQSVM`]
        Trained kernel machine model.
    train_data : :class:`numpy.ndarray`
        Training data array.
    train_labels : :class:`numpy.ndarray`
        Training data labels.
    test_data : :class:`numpy.ndarray`
        Testing data array.
    test_labels : :class:`numpy.ndarray`
        Testing data array.
    out_path : str
        Output path to saved the figures in.

    Raises
    ------
    TypeError
        Passed model not of the correct type.
    """
    print("Computing the test dataset accuracy of the models, ROC and PR plots...")
    test_time_init = perf_counter()
    train_acc = None
    if (
        isinstance(model, QSVM)
        or isinstance(model, OneClassQSVM)
        or isinstance(model, CustomOneClassSVM)
    ):
        train_acc = model.score(train_data, train_labels, train_data=True)
    elif isinstance(model, SVC):
        train_acc = model.score(train_data, train_labels)
    else:
        raise TypeError(
            tcols.FAIL
            + "The model should be either a SVC or a QSVM or a OneClassSVM or"
            " a OneClassQSVM object." + tcols.ENDC
        )
    y_score = model.decision_function(test_data)
    compute_roc_pr_curves(test_labels, y_score, out_path, model)
    plot_score_distributions(y_score, test_labels, out_path)

    y_score[y_score > 0.0] = 1
    y_score[y_score < 0.0] = 0
    test_acc = accuracy_score(test_labels, y_score)
    print_accuracy_scores(test_acc, train_acc, isinstance(model, OneClassSVM))

    test_time_fina = perf_counter()
    exec_time = test_time_fina - test_time_init
    print(
        f"Completed evaluation in: {exec_time:.2e} sec. or "
        f"{exec_time/60:.2e} min. " + tcols.ROCKET
    )


def plot_score_distributions(y_score: np.ndarray, y_label: np.ndarray, out_path: str):
    """Plots and saves the score distributions for signal and background as a histogram.

    Parameters
    ----------
    y_score : :class:`numpy.ndarray`
        Output score of a model.
    y_label : :class:`numpy.ndarray`
        True labels.
    out_path : str
        Output path to save the plot in.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.hist(
        y_score[y_label == 1],
        histtype="step",
        linewidth=2,
        bins=60,
        label="Signal",
        density=True,
        color="royalblue",
    )
    plt.hist(
        y_score[y_label == 0],
        histtype="step",
        linewidth=2,
        bins=60,
        label="Background",
        density=True,
        color="red",
    )
    plt.xlabel("score")
    plt.ylabel("A.U.")
    plt.yscale("log")
    plt.legend()
    plt.savefig(out_path + "/score_distribution.pdf")
    plt.clf()
    print("\n Saving score distributions for signal and background " + tcols.SPARKS)


def compute_roc_pr_curves(
    test_labels: np.ndarray,
    y_score: np.ndarray,
    out_path: str,
    model: Union[SVC, QSVM, OneClassQSVM, CustomOneClassSVM],
):
    """Computes the ROC and Precision-Recall (PR) curves and saves them in the model
    out_path. Also, prints the 1/FPR value around a TPR working point, default=0.8.

    Parameters
    ----------
    test_labels : :class:`numpy.ndarray`
        Testing data truth labels.
    y_score : :class:`numpy.ndarray`
        Output model scores on the testing dataset.
    out_path : str
        Path to save plot.
    model : Union[:class:`sklearn.svm.SVC`, `QSVM`, `OneClassQSVM`, `CustomOneClassSVM`]
        Kernel machine model.
    """

    def create_plot_model_label(
        model,
    ) -> str:
        """Creates the label for the legend, include hyperparameter and
        feature map or kernel name info.

        Parameters
        ----------
        model : Union[:class:`sklearn.svm.SVC`, `QSVM`, `OneClassQSVM`, `CustomOneClassSVM`]
            Kernel machine model.

        Returns
        -------
        str
            Label for the plot.
        """
        if isinstance(model, OneClassSVM):
            label = (
                f"\nFeature map: {model.feature_map_name}\n"
                + r"$\nu$ = "
                + f"{model.nu}"
            )
        elif isinstance(model, QSVM):
            label = f"\nFeature map: {model.feature_map_name}" + f"\nC = {model.C}"
        else:
            label = f"\nFeature map: {model.kernel}" + f"\nC = {model.C}"
        return label

    fpr, tpr, thresholds = roc_curve(y_true=test_labels, y_score=y_score)
    auc = roc_auc_score(test_labels, y_score)
    one_over_fpr = get_fpr_around_tpr_point(fpr, tpr)
    plt.plot(
        tpr,
        1.0 / fpr,
        label=f"AUC: {auc:.3f}\n" + r"FPR$^{-1}$:" + f" {one_over_fpr[0]:.3f}"
        f" ± {one_over_fpr[1]:.3f}" + create_plot_model_label(model),
    )
    plt.yscale("log")
    plt.xlabel("TPR")
    plt.xlim([0.0, 1.1])
    plt.ylabel(r"FPR$^{-1}$")
    plt.legend()
    plt.savefig(out_path + "/roc.pdf")
    plt.clf()

    p, r, thresholds = precision_recall_curve(test_labels, probas_pred=y_score)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(out_path + "/pr.pdf")
    print("\nComputed ROC and PR curves " + tcols.SPARKS)
    plt.clf()


def get_fpr_around_tpr_point(
    fpr: np.ndarray, tpr: np.ndarray, tpr_working_point: float = 0.8
) -> Tuple[float]:
    """Computes the mean 1/FPR value that corresponds to a small window aroun a given
    TPR working point (default: 0.8). If there are no values in the window, it widened
    sequentially until it includes some values around the working point.

    Parameters
    ----------
    fpr : :class:`numpy.ndarray`
        False positive rate of the model on the test dataset.
    tpr : :class:`numpy.ndarray`
        True positive rate of the model on the test dataset.
    tpr_working_point : float, optional
        Working point of TPR (signal efficiency), by default 0.8

    Returns
    -------
    Tuple
        The mean and standard deviation of 1/FPR @ TPR=`tpr_working_point`.
    """
    ind = np.array([])
    low_bound = tpr_working_point * 0.999
    up_bound = tpr_working_point * 1.001
    while len(ind) == 0:
        ind = np.where(np.logical_and(tpr >= low_bound, tpr <= up_bound))[0]
        low_bound *= 0.99  # open the window by 1%
        up_bound *= 1.01
    fpr_window_no_zeros = fpr[ind][fpr[ind] != 0]
    one_over_fpr_mean = np.mean(1.0 / fpr_window_no_zeros), np.std(
        1.0 / fpr_window_no_zeros
    )
    print(
        f"\nTPR values around {tpr_working_point} window with lower bound {low_bound}"
        f" and upper bound: {up_bound}"
    )
    print(
        f"Corresponding mean 1/FPR value in that window: {one_over_fpr_mean[0]:.3f} ± "
        f"{one_over_fpr_mean[1]:.3f}"
    )
    return one_over_fpr_mean


def export_hyperparameters(
    model: Union[QSVM, SVC, CustomOneClassSVM, OneClassQSVM], outdir: str
):
    """Saves the hyperparameters of the model to a json file. QSVM and SVM have
    different hyperparameters.

    Parameters
    ----------
    model : Union[`QSVM`, :class:`sklearn.svm.SVC`, `CustomOneClassSVM`, `OneClassQSVM`]
       Kernel machine model.
    outdir : str
        Output directory, where the json file is saved.
    """
    file_path = os.path.join(outdir, "hyperparameters.json")
    if isinstance(model, QSVM):
        hp = {
            "C": model.C,
            "nqubits": model.nqubits,
            "feature_map_name": model.feature_map_name,
            "backend_config": model.backend_config,
        }  # FIXME this doesn't include OneClassQSVM
    else:
        hp = {"C": model.C}
    params_file = open(file_path, "w")
    json.dump(hp, params_file)
    params_file.close()
