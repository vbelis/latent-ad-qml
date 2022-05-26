import h5py
import numpy as np
from typing import Tuple

from terminal_colors import tcols


def get_data(args: dict) -> Tuple:
    """
    Loads a dataset specified by a path for the qsvm training. The raw data file
    is in .h5 format.
    Args:
        args: Dictionary with all the specification and parameters for the qsvm
              training.
    Returns:
        A tuple with the data "loaders" (inspired by PyTorch jargon). Each loader
        consists of the data vectors (features) and the corresponding labels. 
        The loaders contain the training and the test data.
    """
    print(tcols.BOLD + "\nPreparing training and testing datasets... " 
          + tcols.ENDC)
    print("Signal ", end="")
    x_sig = h5_to_ml_ready_numpy(args["sig_path"])
    print("Background ", end="")
    x_bkg = h5_to_ml_ready_numpy(args["bkg_path"])
    print("Testing background ", end="")
    x_bkg_test = h5_to_ml_ready_numpy(args["test_bkg_path"])
    #print("\n")
    train_loader = get_train_dataset(x_sig, x_bkg, args["ntrain"])
    test_loader = get_test_dataset(x_sig, x_bkg_test, args["ntest"])
    return train_loader, test_loader

def h5_to_ml_ready_numpy(file_path: str) -> np.ndarray:
    """
    Takes a .h5 file and returns an flattened numpy array. The .h5 file
    follows the convention: under the 'latent_space' key the autoencoder
    latent representation of the di-jet event is saved. 
    It is of shape (n, 2,latent_dim), hence we want to reshape it to 
    (n, 2*latent_dim,), where n is the number of events in the .h5 file.
    
    Args:
        file_path: Full path to .h5 file.
    Returns:
        The corresponding numpy array of shape ()
    """
    h5_file = h5py.File(file_path, 'r')
    latent_rep = np.asarray(h5_file.get('latent_space'))
    print("raw data: " + tcols.OKBLUE + f"{file_path}" + tcols.ENDC + 
          ", ", end="")
    latent_rep_flat = reshaper(latent_rep)
    return latent_rep_flat

def reshaper(array: np.ndarray) -> np.ndarray:
    """
    Takes the signal and background arrays, flattens, and stacks them.
    Returns the data feature array.
    Args:
        Array to reshape. The expected initial shape is (n, 2, latent_dim).
    Returns:
        Reshaped array of shape (n, 2*latent_dim). Where n is the total number
        of events in the file.
    """
    print(f"reshape {array.shape}", end="")
    array = np.reshape(array, (len(array), -1))
    print(f" -> {array.shape}")
    return array

def get_train_dataset(sig: np.ndarray, bkg: np.ndarray, ntrain: int) \
                      -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructing the training dataset based on the conventions used for this work.
    Namely, last ntrain/2 from the sig file and the first ntrain/2 from the bkg file.
    Args:
        sig: Array containing all the signal events needed for the training.
        bkg: Array containing all the background events needed for the training.
        ntrain: Number of requested training samples in total (sig+bkg).
    """
    sig = sig[-int(ntrain/2):]
    bkg = bkg[:int(ntrain/2)]
    x_data_train = np.concatenate((sig,bkg))
    print(f"Created training dataset of shape: {x_data_train.shape}")
    y_data_train = create_output_y(int(ntrain/2))
    return x_data_train, y_data_train

def get_test_dataset(sig: np.ndarray, bkg_test: np.ndarray, ntest: int) \
                     -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructing the testing dataset based on the conventions used for this work.
    Namely, first ntest/2 samples from the signal and first ntest/2 from the 
    dedicate background test file. The default ntest = 20k.
    Args:
        sig: Array containing all the signal events needed for the testing.
        bkg: Array containing all the background events needed for the testing.
        ntrain: Number of requested testing samples in total (sig+bkg).
    """
    sig = sig[:int(ntest/2)]
    bkg_test = bkg_test[:int(ntest/2)]
    x_data_test = np.concatenate((sig,bkg_test))
    y_data_test = create_output_y(int(ntest/2))
    print(f"Created testing dataset of shape: {x_data_test.shape}")
    return x_data_test, y_data_test

def create_output_y(n: int) -> np.ndarray:
    """
    Creates the out target/label files according the data file structure.
    n can refer to the number of training events or test events.
    """
    y_data = np.concatenate((np.ones(n), np.zeros(n)))
    return y_data

def get_kfold_data(test_data: np.ndarray, kfolds: int = 5) -> np.ndarray:
    """
    Split the testing dataset into k folds. With this resampling technique we 
    will estimate the variance and the mean of the expected ROC curve and AUC.
    Args:
        test_data: Array of the testing data.
        kfolds: Number of requested k-folds.
    Returns:
        Returns array of shape (kfolds, ntest/kfolds).
    """
    kfold_data = np.split(test_data, kfolds)
    # FIXME need to think which dataset to split, the final one or before some reshaping step?

