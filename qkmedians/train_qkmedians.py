import numpy as np
np.random.seed(42)
import math
import h5py

import qibo
qibo.set_backend("tensorflow")

import qkmedians as qkmed
import utils as u

qibo.set_device("/GPU:0")


def train_qkmedians(latent_dim, train_size, read_file, seed=1234, k=2, tolerance=1.e-3, save_file=None):

    # read train data
    with h5py.File(read_dir+file_name, 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]

        data_train = np.vstack([l1[:train_size], l2[:train_size]])
        np.random.seed(seed)
        np.random.shuffle(data_train)

    # train qkmedians
    centroids = qkmed.initialize_centroids(data_train, k)   # Intialize centroids

    i = 0
    loss=[]
    while True:
        cluster_label, _ = qkmed.find_nearest_neighbour_DI(data_train,centroids)       # find nearest centers
        print(f'Found cluster assignments for iteration: {i+1}')
        new_centroids = qkmed.find_centroids_GM(data_train, cluster_label, clusters=k)               # find centroids

        loss_epoch = np.linalg.norm(centroids - new_centroids)
        loss.append(loss_epoch)

        if loss_epoch < tolerance:
            centroids = new_centroids
            print(f"Converged after {i+1} iterations.")
            break
        i += 1     
        centroids = new_centroids

    np.save(f'{save_dir}/cluster_label_lat{latent_dim}_{str(train_size)}.npy', cluster_label)
    np.save(f'{save_dir}/centroids_lat{latent_dim]}_{str(train_size)}.npy', centroids)
    np.save(f'{save_dir}/LOSS_lat{latent_dim}_{str(train_size)}.npy', loss)
    print('Centroids and labels saved!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='read arguments for qkmedians training')
    parser.add_argument('-latent_dim', dest='latent_dim', type=str, help='latent dimension')
    parser.add_argument('-train_size', dest='train_size', type=int, help='training data size')
    parser.add_argument('-read_file', dest='read_file', type=str, help='training data file')
    parser.add_argument('-seed', dest='seed', type=int, help='seed for consistent results')
    parser.add_argument('-k', dest='k', type=int, default=2, help='number of classes')
    parser.add_argument('-tolerance', dest='tolerance', type=float, help='tolerance')
    parser.add_argument('-save_file', dest='save_file',type=str, help='file to save results')

    args = parser.parse_args()
    
    train_qkmedians(args.latent_dim, args.train_size, args.read_file, args.seed, args.k, args.tolerance, args.save_file)
