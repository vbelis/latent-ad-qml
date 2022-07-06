from time import perf_counter
import numpy as np

import util
import data_processing
from terminal_enhancer import tcols


def main(args):
    _, test_loader = data_processing.get_data(args)
    test_features, test_labels = test_loader[0], test_loader[1]
    sig_fold, bkg_fold = data_processing.get_kfold_data(
        test_features,
        test_labels,
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
