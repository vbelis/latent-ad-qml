# The quantum and classical SVM training script. Here, the model is instantiated
# with some parameters, the circuit is built, and then it is trained on a data set.
# The model is then checked for overtraining by computing the accuracy on the test
# and train data sets. The model along  are saved in a folder
# with the name of a user's choosing..

import argparse
import json
from time import perf_counter
from typing import Callable
from qiskit.utils import algorithm_globals

import util
import data_processing
from terminal_enhancer import tcols

seed = 12345
algorithm_globals.random_seed = seed


def main(args: dict):
    train_loader, test_loader = data_processing.get_data(args)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]

    model = util.init_kernel_machine(args)
    out_path = util.create_output_folder(args, model)

    time_and_train(model.fit, train_features, train_labels)
    util.print_model_info(model)

    util.eval_metrics(model, train_features, train_labels, test_features, test_labels, out_path)
    util.save_model(model, out_path)
    util.export_hyperparameters(model, out_path)


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
    exec_time = train_time_fina - train_time_init
    print(
        "Training completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
        f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS
    )

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
        "--quantum", action="store_true", help="Flag to choose between QSVM and SVM."
    )
    parser.add_argument(
        "--unsup", action="store_true", help="Flag to choose between unsupervised and supervised models"
    )
    parser.add_argument(
        "--nqubits", type=int, help="Number of qubits for quantum feature map circuit."
    )
    parser.add_argument("--feature_map", type=str, help="Feature map circuit for the QSVM.")
    parser.add_argument(
        "--backend_name",
        type=str,
        help="The IBM backend. Could be a simulator"
        ", noise model, or a real quantum computer",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        choices=["ideal", "noisy", "hardware"],
        help="Choose way to run the QSVM: Ideal computation,"
        "noisy simulation or on real quantum hardware.",
    )
    parser.add_argument(
        "--output_folder", required=True, help="The name of the model to be saved."
    )
    parser.add_argument(
        "--c_param", type=float, default=1.0, help="The C parameter of the SVM."
    )
    parser.add_argument(
        "--nu_param", type=float, default=1.0, help="The nu parameter of the one-class SVM."
    )
    parser.add_argument(
        "--gamma",
        nargs="+",
        default="scale",
        help="The gamma parameter of the SVM with rbf kernel.",
    )
    parser.add_argument(
        "--ntrain", type=int, default=600, help="Number of training events for the QSVM."
    )
    parser.add_argument(
        "--ntest", type=int, default=720, help="Number of test events for the QSVM."
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Initiate grid search on the C hyperparameter.",
    )
    args = parser.parse_args()

    # Load private configuration file for ibmq_api_token and provider details.
    with open("private_config_vasilis.json") as pconfig:
        private_configuration = json.load(pconfig)

    # Different configuration keyword arguments for the QuantumInstance depending
    # on the run_type. They can be tweaked as desired before running.
    initial_layout = [22, 25, 24, 23, 21, 18, 15, 12]  # for Cairo

    seed = 12345
    config_noisy = {
        "optimization_level": 3,
        "initial_layout": initial_layout,
        "seed_transpiler": seed,
        "seed_simulator": seed,
        "shots": 5000,
    }
    config_hardware = {
        "optimization_level": 3,
        "initial_layout": initial_layout,
        "seed_transpiler": seed,
        "shots": 5000,
    }
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
        "noisy": lambda: config_noisy,
        "hardware": lambda: config_hardware,
    }
    config = switcher.get(args.run_type, lambda: None)()

    args = {
        "sig_path": args.sig_path,
        "bkg_path": args.bkg_path,
        "test_bkg_path": args.test_bkg_path,
        "c_param": args.c_param,
        "nu_param": args.nu_param,
        "output_folder": args.output_folder,
        "gamma": args.gamma,
        "quantum": args.quantum,
        "unsup": args.unsup,
        "nqubits": args.nqubits,
        "feature_map": args.feature_map,
        "backend_name": args.backend_name,
        "ibmq_api_config": private_configuration["IBMQ"],
        "run_type": args.run_type,
        "config": config,
        "ntrain": args.ntrain,
        "ntest": args.ntest,
        "seed": seed,  # For the data shuffling.
        "grid_search": args.grid_search,
    }
    return args


if __name__ == "__main__":
    args = get_arguments()
    if not args["grid_search"]:
        main(args)
    else:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        print(
            tcols.BOLD
            + tcols.HEADER
            + f"\nInitialising grid search for C = {c_values}..."
            + tcols.ENDC
        )
        for c in c_values:
            print(tcols.UNDERLINE + tcols.BOLD + f"\nC = {c}" + tcols.ENDC)
            args["c_param"] = c
            main(args)
    
