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
h5f.create_dataset('classic_loss_qcd', data=classical_bkg_scores)
h5f.create_dataset('classic_loss_sig', data=classical_sig_scores)
h5f.create_dataset('quantum_loss_qcd', data=quantum_bkg_scores)
h5f.create_dataset('quantum_loss_sig', data=quantum_sig_scores)
h5f.close()
