# Gets the scores of a classical and a quantum model on the k-fold
# testing signal and background samples, and outputs an .h5 file
# used for plotting the ROC curves and AUCs.

import numpy as np
import argparse
import h5py

from terminal_enhancer import tcols

def save_scores_h5(classical_path: str, quantum_path: str, out_path: str):
    """
    Save the scores of a model to an .h5 file following the following convention.
    Data frame with keys 'classic_loss_qcd', 'classic_loss_sig' for the classical
    model background and signal test scores respectively. Correspondingly, 
    for the quantum 'quantum_loss_qcd', 'quantum_loss_sig'.
    
    Args:
        classical_path: path to the classical model to load sig and bkg scores
                        of shape: (kfolds, n_test/kfolds)
        bkg_scores: path to the quantum model, to load sig and bkg scores
                    of shape: (kfolds, n_test/kfolds)
        output_path: Path to the generated .h5 file.
    """
    print("Loading scores of the quantum model: " + tcols.OKBLUE + 
          "{quantum_path}" + tcols.ENDC)
    quantum_sig = np.load(args.quantum_folder + "sig_scores.npy")
    quantum_bkg = np.load(args.quantum_folder + "bkg_scores.npy")
    print("Loading scores of the classical model: " + tcols.OKBLUE + 
          "{classical_path}" + tcols.ENDC)
    classical_sig = np.load(args.classical_folder + "sig_scores.npy")
    classical_bkg = np.load(args.classical_folder + "bkg_scores.npy")

    h5f = h5py.File(args.out_path, 'w')
    h5f.create_dataset('quantum_loss_qcd', data=quantum_bkg)
    h5f.create_dataset('quantum_loss_sig', data=quantum_sig)
    h5f.create_dataset('classic_loss_qcd', data=classical_bkg)
    h5f.create_dataset('classic_loss_sig', data=classical_sig)
    h5f.close()
    print("Created a .h5 file containing the quantum and classical scores in: "
          + tcols.OKGREEN + f"{out_path}" + tcols.ENDC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)
    parser.add_argument("--classical_folder", type=str, required=True,
                        help="Folder of the trained classical model.")
    parser.add_argument("--quantum_folder", type=str, required=True,
                        help="Path to the QCD background dataset (.h5 format).")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to the output file containing the four " 
                             "arrays of scores, sig+bkg for both classical and"
                             " quantum models in .h5 format.")
    args = parser.parse_args()

    
    save_scores_h5(args.classical_folder, args.quantum_folder, args.out_path,)
