"""
import argparse
import numpy as np
import sys
sys.path.append("..")
from qiskit_machine_learning.kernels import QuantumKernel

from . import util
from qsvm.feature_map_circuits import u2Reuploading
from qsvm.test import compute_model_scores

parser = argparse.ArgumentParser(formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)
parser.add_argument("--test_data_path", type=str, required=True,
                    help="Path to the test data file.")
parser.add_argument('--qsvm_model', type=str, required=True,
                    help="The folder path of the QSVM model.")
parser.add_argument('--run_type', type=str, required=True,
                    choices=['ideal', 'noisy', 'hardware'],
                    help='Choose way to run the QSVM: Ideal computation,'
                    'noisy simulation or on real quantum hardware.')

args = parser.parse_args()

def main(args):
    x_test = np.load(args.test_data_path)
    n_test = len(x_test)

    qsvm = util.load_qsvm(args["qsvm_model"] + "model")
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


    

def compute_scores(model, kernel, x_train, x_test):
    scores = model.decision_function(kernel.evaluate(x_vec=x_test, y_vec=x_train))

    return scores


if __name__ == "__main__":
    main(args)
"""
import numpy as np
import h5py
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)
parser.add_argument("--quantum_scores_path", type=str, required=True,
                    help="Path to the qsvm scores.")
parser.add_argument("--classical_scores_path", type=str, required=True,
                    help="Path to the svm benchmark scores.")
parser.add_argument("--output_path", type=str, required=True, 
				    help="Path to save the output h5 file.")
args = parser.parse_args()


quantum_scores = np.load(args.quantum_scores_path)
classical_scores = np.load(args.classical_scores_path)
quantum_scores = quantum_scores.flatten()
classical_scores = classical_scores.flatten()
n_test = len(quantum_scores)

quantum_sig_scores = quantum_scores[:int(n_test/2)]
quantum_bkg_scores = quantum_scores[int(n_test/2):]
classical_sig_scores = classical_scores[:int(n_test/2)]
classical_bkg_scores = classical_scores[int(n_test/2):]
print(quantum_bkg_scores.shape)
print(quantum_sig_scores.shape)
print(classical_bkg_scores.shape)
print(classical_sig_scores.shape)

# qcd = bkg here.
h5f = h5py.File(args.output_path, 'w')
h5f.create_dataset('classical_loss_qcd', data=classical_bkg_scores)
h5f.create_dataset('classical_loss_sig', data=classical_sig_scores)
h5f.create_dataset('quantum_loss_qcd', data=quantum_bkg_scores)
h5f.create_dataset('quantum_loss_sig', data=quantum_sig_scores)
h5f.close()
