# Quantum anomaly detection in the latent space of particle physics events

Supervised and unsupervised anomaly detection in the latent space of high energy physics events with quantum machine learning

## Plotting
You can get the plots by running in a jupyter notebook the following. I am using the convention that I have for the `.h5` file paths, maybe you need to tweak that in the code.

### The 3 different signals for 8 latent dimensions and n_train=600, n_test=100k, k=5 folds
Firstly, load the score values from the saved files using our convention
```
import h5py
import sys
sys.path.append("..")

import numpy as np
import plotting as pl
%load_ext autoreload
%autoreload 2

read_dir='/eos/path/to/the/.h5/files/lat8/unsupervised'
n_folds = 5
latent_dim = '8'
n_samples_train=600
mass=['35', '15', '35']
br_na=['NA', 'BR', ''] # narrow (NA) or broad (BR)
signal_name=['RSGraviton_WW', 'RSGraviton_WW', 'AtoHZ_to_ZZZ']

q_loss_qcd=[]; q_loss_sig=[]; c_loss_qcd=[]; c_loss_sig=[]
for i in range(len(signal_name)):
    with h5py.File(f'{read_dir}/Latent_{latent_dim}_trainsize_{n_samples_train}_{signal_name[i]}{mass[i]}{br_na[i]}_n100k_kfold{n_folds}.h5', 'r') as file:
        q_loss_qcd.append(file['quantum_loss_qcd'][:])
        q_loss_sig.append(file['quantum_loss_sig'][:])
        c_loss_qcd.append(file['classic_loss_qcd'][:])
        c_loss_sig.append(file['classic_loss_sig'][:])
```

Then plot the results like so, making sure the `save_dir` exists.

```
legend_signal_names=['Narrow 'r'G $\to$ WW 3.5 TeV', 'Broad 'r'G $\to$ WW 1.5 TeV', r'A $\to$ HZ $\to$ ZZZ 3.5 TeV']
pl.plot_ROC_kfold_mean(q_loss_qcd, q_loss_sig, c_loss_qcd, c_loss_sig, legend_signal_names, n_folds,\
                title=r'testlat=8', save_dir='../jupyter_plots', pic_id='test')
```
Example for the unsupervised kernel machine:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/48251467/196224459-c9e18b6a-12e1-4a5f-8e66-e6ae979e27ef.png">

### The 4, 8, 16 latent dimension plot with n_train=600, n_test=40k, k=5 folds, for AtoHZ
Likewise, load the scores,
```
read_dir='/eos/user/v/vabelis/mpp_collab/journal_data/journal_plots/'
n_folds = 5
latent_dim = ['4', '8', '16']
n_samples_train=600
mass='35'
br_na=''
signal_name='AtoHZ_to_ZZZ'
n_test = ['40k', '40k', '40k']

q_loss_qcd=[]; q_loss_sig=[]; c_loss_qcd=[]; c_loss_sig=[]
for i in range(len(latent_dim)):
    with h5py.File(f'{read_dir}/lat{latent_dim[i]}/unsupervised/Latent_{latent_dim[i]}_trainsize_{n_samples_train}_{signal_name}{mass}{br_na}_n{n_test[i]}_kfold{n_folds}.h5', 'r') as file:
        q_loss_qcd.append(file['quantum_loss_qcd'][:])
        q_loss_sig.append(file['quantum_loss_sig'][:])
        c_loss_qcd.append(file['classic_loss_qcd'][:])
        c_loss_sig.append(file['classic_loss_sig'][:])
```
and plot:
```
legend_signal_names=['lat4', 'lat8', 'lat16']
pl.plot_ROC_kfold_mean(q_loss_qcd, q_loss_sig, c_loss_qcd, c_loss_sig, legend_signal_names, n_folds,\
                title=r'test A $\to$ HZ $\to$ ZZZ', save_dir='../jupyter_plots', pic_id='test')
```
Example for the unsupervised QSVM in lat=4, 8, 16:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/48251467/196227976-63ed8cae-9709-4697-879b-f64cf579a197.png">
