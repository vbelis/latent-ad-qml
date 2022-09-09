from time import perf_counter
import numpy as np

import util
import argparse
import data_processing
from terminal_enhancer import tcols


def main(args: dict):
    _, test_loader = data_processing.get_data(args)
    test_features, test_labels = test_loader[0], test_loader[1]
    sig_fold, bkg_fold = data_processing.get_kfold_data(
        test_features,
        test_labels,
        args["kfolds"]
    )
    output_path = args["model"]
    model = util.load_model(output_path + "model")

    print("Computing model scores... ", end="")
    scores_time_init = perf_counter()
    score_sig = np.array([model.decision_function(fold) for fold in sig_fold])
    score_bkg = np.array([model.decision_function(fold) for fold in bkg_fold])
    scores_time_fina = perf_counter()
    exec_time = scores_time_fina - scores_time_init
    print(
        tcols.OKGREEN
        + "Completed in: "
        + tcols.ENDC
        + f"{exec_time:2.2e} sec. or {exec_time/60:2.2e} min. "
        + tcols.ROCKET
    )
    print(
        f"Saving the signal and background k-fold scores in the folder: "
        + tcols.OKCYAN
        + f"{output_path}"
        + tcols.ENDC
    )
    np.save(output_path + "sig_scores.npy", score_sig)
    np.save(output_path + "bkg_scores.npy", score_bkg)

def get_arguments() -> dict:
    """
    Parses command line arguments and gives back a dictionary.
    Returns: Dictionary with the arguments
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--sig_path",
        type=str,
        required=True,
        help="Path to the signal/anomaly dataset (.h5 format).",
    )
    parser.add_argument(
        "--bkg_path",
        type=str,
        required=True,
        help="Path to the QCD background dataset (.h5 format).",
    )
    parser.add_argument(
        "--test_bkg_path",
        type=str,
        required=True,
        help="Path to the background testing dataset (.h5 format).",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The folder path of the QSVM model."
    )
    parser.add_argument(
        "--ntest", type=int, default=720, help="Number of test events for the QSVM."
    )
    parser.add_argument(
        "--kfolds", type=int, default=5, help="Number of k-validation/test folds used."
    )
    args = parser.parse_args()

    args = {
        "sig_path": args.sig_path,
        "bkg_path": args.bkg_path,
        "test_bkg_path": args.test_bkg_path,
        "model": args.model,
        "ntest": args.ntest,
        "kfolds": args.kfolds,
    }
    return args


if __name__ == "__main__":
    args = get_arguments()
    main(args)

