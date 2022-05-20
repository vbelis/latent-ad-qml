import numpy as np
np.random.seed(42)
import math
import h5py

import qibo
qibo.set_backend("tensorflow")

import qkmedians as qkmed
import utils as u

qibo.set_device("/GPU:0")

latent_dim='4' # args1
train_size=600 #args2

save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/diJet' #args3

# read QCD predicted data (test - SIDE)
read_dir =f'/eos/user/e/epuljak/private/epuljak/public/diJet/{latent_dims[j]}/' #args4
file_name = 'latentrep_QCD_sig_around35.h5' #args5
with h5py.File(read_dir+file_name, 'r') as file:
    data = file['latent_space']
    l1 = data[:,0,:]
    l2 = data[:,1,:]

    data_train = np.vstack([l1[:train_size], l2[:train_size]])
    np.random.shuffle(data_train)

# TRAIN Q-MEDIANS
k = 2 # number of clusters #args6
centroids = qkmed.initialize_centroids(data_train, k)   # Intialize centroids
tolerance=1.e-3 # args7

# run k-medians algorithm
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
        
    print(f'Iteration: {i+1}')
    if i == 200: 
        print('QKmedians stopped after 200 epochs!')
        centroids = new_centroids
        break
    i += 1     
    centroids = new_centroids

np.save(f'{save_dir}/cluster_labels/around35/cluster_label_lat{latent_dims[j]}_{str(n_train_samples)}.npy', cluster_label)
np.save(f'{save_dir}/centroids/around35/centroids_lat{latent_dims[j]}_{str(n_train_samples)}.npy', centroids)
np.save(f'{save_dir}/loss/around35/LOSS_lat{latent_dims[j]}_{str(n_train_samples)}.npy', loss)
print('Centroids and labels saved!')

# read args .... TODO