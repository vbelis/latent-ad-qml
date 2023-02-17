import argparse
import numpy as np
import h5py

import qad.algorithms.kmedians.classical.c_kmedians as KMed


def train_kmedians(
    latent_dim, train_size, read_file, seed=None, k=2, tolerance=1.0e-3, save_dir=None
):
    """Performs training of k-medians.

    Parameters
    ----------
    latent_dim : int
        Latent dimension of input data.
    train_size : int
        Number of training samples.
    read_file : str
        Name of the file where training data is saved.
    seed : int
        Seed for data shuffling.
    k : int
        Number of classes in k-medians.
    tolerance : float
        Tolerance for algorithm convergence.
    save_dir : str
        Name of the file for saving results.
    """

    # read train data
    with h5py.File(read_file, "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]

        data_train = np.vstack([l1[:train_size], l2[:train_size]])
        if seed:
            np.random.seed(seed)  # matters for small data sizes
        np.random.shuffle(data_train)

    kmedians = KMed.Kmedians(k=k, tolerance=tolerance)
    kmedians.fit(data_train)

    loss = kmedians.loss
    centroids = kmedians.centroids

    if save_dir:
        np.save(
            f"{save_dir}/centroids_lat{latent_dim}_{str(train_size)}.npy", centroids
        )
        np.save(f"{save_dir}/LOSS_lat{latent_dim}_{str(train_size)}.npy", loss)
        print("Centroids and labels saved!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="read arguments for qkmedians training"
    )
    parser.add_argument(
        "-latent_dim", dest="latent_dim", type=int, help="latent dimension"
    )
    parser.add_argument(
        "-train_size", dest="train_size", type=int, help="training data size"
    )
    parser.add_argument(
        "-read_file", dest="read_file", type=str, help="training data file"
    )
    parser.add_argument(
        "-seed", dest="seed", type=int, help="seed for consistent results"
    )
    parser.add_argument("-k", dest="k", type=int, default=2, help="number of classes")
    parser.add_argument(
        "-tolerance", dest="tolerance", type=float, default=1.0e-3, help="tolerance"
    )
    parser.add_argument(
        "-save_dir", dest="save_dir", type=str, help="directory to save results"
    )

    args = parser.parse_args()

    train_kmedians(
        args.latent_dim,
        args.train_size,
        args.read_file,
        args.seed,
        args.k,
        args.tolerance,
        args.save_dir,
    )
