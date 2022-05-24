import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)
parser.add_argument("--sig_path", type=str, required=True,
                    help="Path to the signal data we want to preprocess.")
parser.add_argument("--bkg_path", type=str, required=True,
                    help="Path to the background data we want to preprocess.")
parser.add_argument("--test_bkg_path", type=str, required=True,
					help="File for bkg testing samples (as agreed with Kinga/Ema.")
parser.add_argument("--ntrain", type=int, required=True, 
				    help="Number of training data samples.")
parser.add_argument("--ntest", type=int, required=True, 
				    help="Number of testing data samples.")
parser.add_argument("--output_folder", type=str, required=True, 
				    help="Folder to save the output datasets.")

args = parser.parse_args()
args = {
	"sig_path": args.sig_path,
	"bkg_path": args.bkg_path,
	"test_bkg_path": args.test_bkg_path,
	"ntrain": args.ntrain,
	"ntest": args.ntest,
	"output_folder": args.output_folder,
}

def main(args):
	x_sig = h5_to_ml_ready_numpy(args["sig_path"])
	x_bkg = h5_to_ml_ready_numpy(args["bkg_path"])
	x_bkg_test = h5_to_ml_ready_numpy(args["test_bkg_path"])
	print(f"Array shape before flattening: {x_sig.shape}")
	x_data_train, y_data_train = get_train_dataset(x_sig, x_bkg, 
	 											   args["ntrain"])
	x_data_test, y_data_test = get_test_dataset(x_sig, x_bkg_test,
												args["ntest"])
	output_folder = args['output_folder']+f"ntrain{args['ntrain']}"
	if not os.path.exists(output_folder): os.makedirs(output_folder)
	np.save(f"{output_folder}/x_data_train.npy", x_data_train)
	np.save(f"{output_folder}/y_data_train.npy", y_data_train)
	np.save(f"{output_folder}/x_data_test.npy", x_data_test)
	np.save(f"{output_folder}/y_data_test.npy", y_data_test)

#def create_output_folder(sig_path):
#../../data/kinga_collab/extra_ema_tests/8/latentrep_RSGraviton_WW_NA_35.h5

def get_train_dataset(sig, bkg, ntrain):
	sig = sig[-int(ntrain/2):] #last ntrain/2 from the sig file
	bkg = bkg[:int(ntrain/2)] #first ntrain/2 from the bkg file
	x_data_train = np.concatenate((sig,bkg))
	print(f"Created training dataset of shape: {x_data_train.shape}")
	y_data_train = create_output_y(int(ntrain/2))
	return x_data_train, y_data_train

def get_test_dataset(sig, bkg_test, ntest):
	sig = sig[:int(ntest/2)]
	bkg_test = bkg_test[:int(ntest/2)]
	x_data_test = np.concatenate((sig,bkg_test))
	y_data_test = create_output_y(int(ntest/2))
	print(f"Created testing dataset of shape: {x_data_test.shape}")
	return x_data_test, y_data_test

def h5_to_ml_ready_numpy(file_path):
	h5_file = h5py.File(file_path, 'r')
	latent_rep = np.asarray(h5_file.get('latent_space'))
	latent_rep_flat = reshaper(latent_rep)
	return latent_rep_flat

def reshaper(array):
	"""
	Takes the signal and background arrays, flattens, and stacks them.
	Returns the data feature array.
	"""
	array = np.reshape(array, (len(array), -1))
	print(f"Flatten array shape: {array.shape}")
	return array

def create_output_y(n):
	"""
	Creates the out target/label files according the input training data file.
	"""
	# number of sig and bkg is the same.
	y_data = np.concatenate((np.ones(n), np.zeros(n)))
	return y_data


if __name__ == '__main__':
	main(args)
		

