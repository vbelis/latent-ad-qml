
# TODO: 
def save_scores_h5(sig_scores: np.ndarray, bkg_scores: np.ndarray, 
                   is_quantum: bool, output_path: str):
    """
    Save the scores of a model to an .h5 file following the following convention.
    Data frame with keys 'classic_loss_qcd', 'classic_loss_sig' for the classical
    model background and signal test scores respectively. Correspondingly, for the 
    quantum 'quantum_loss_qcd', 'quantum_loss_sig'.
    
    Args:
        sig_scores: the signal scores of the model, shape: (kfolds, n_test/kfolds)
        bkg_scores: the background scores of the model, shape: (kfolds, n_test/kfolds)
        is_quantum: Specify whether the model scores are coming from a quantum model.
                    This is needed to specify the names of the stored arrays.
        output_path: Path to the generated .h5 file.
    """
    h5f = h5py.File(output_path, 'w')
    if is_quantum:
        h5f.create_dataset('quantum_loss_qcd', data=bkg_scores)
        h5f.create_dataset('quantum_loss_sig', data=sig_scores)
    else:
        h5f.create_dataset('classic_loss_qcd', data=bkg_scores)
        h5f.create_dataset('classic_loss_sig', data=sig_scores)
    h5f.close()
