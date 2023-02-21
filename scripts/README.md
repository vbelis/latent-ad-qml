# Producing figures
After the unsuperised quantum and classical kernel machines have been trained and test scores have been saved, one can summarise their performance with a ROC curve plot. Firstly, following our convention the test scores are prepared for plotting using [`scripts/kernel_machines/scripts/prepare_plot_scores.py`](https://github.com/vbelis/latent-ad-qml/blob/docs-reformat/scripts/kernel_machines/prepare_plot_scores.py), and by running
```python
python prepare_plot_scores.py --classical_folder trained_qsvms/c_test_nu\=0.01/ --quantum_folder trained_qsvms/q_test_nu\=0.01_ideal/ --out_path test_plot --name_suffix n<n_test>_k<k_folds>
```

Then, we load the score values from the saved files using our convention, e.g. for the case of three different signals, with eight latent dimensions,  600 training datapoints, 100k testing datapoints, and k=5 folds

```python
read_dir='/path/to/data'
n_folds = 5
latent_dim = '8'
n_samples_train=600
mass=['35', '35', '15']
br_na=['NA', '', 'BR'] # narrow (NA) or broad (BR)
signal_name=['RSGraviton_WW', 'AtoHZ_to_ZZZ', 'RSGraviton_WW']
ntest = ['100', '100', '100']

q_loss_qcd=[]; q_loss_sig=[]; c_loss_qcd=[]; c_loss_sig=[]
for i in range(len(signal_name)):
    #if br_na[i]: 
    with h5py.File(f'{read_dir}/Latent_{latent_dim}_trainsize_{n_samples_train}_{signal_name[i]}'
                   '{mass[i]}{br_na[i]}_n{ntest[i]}k_kfold{n_folds}.h5', 'r') as file:
        q_loss_qcd.append(file['quantum_loss_qcd'][:])
        q_loss_sig.append(file['quantum_loss_sig'][:])
        c_loss_qcd.append(file['classic_loss_qcd'][:])
        c_loss_sig.append(file['classic_loss_sig'][:])
```

The final ROC plot, as it appears in the paper in Fig. 3, can be obtained 

```python
colors = ['forestgreen', '#EC4E20', 'darkorchid']
legend_signal_names=['Narrow 'r'G $\to$ WW 3.5 TeV', r'A $\to$ HZ $\to$ ZZZ 3.5 TeV', 'Broad 'r'G $\to$ WW 1.5 TeV']
pl.plot_ROC_kfold_mean(q_loss_qcd, q_loss_sig, c_loss_qcd, c_loss_sig, legend_signal_names, n_folds,\
                legend_title=r'Anomaly signature', save_dir='../jupyter_plots', pic_id='test',
                palette=colors, xlabel=r'$TPR$', ylabel=r'$FPR^{-1}$')
```
Example for the unsupervised kernel machine performance on different anomalies:
<p align="center">
<img width="550" alt="image" src="https://user-images.githubusercontent.com/48251467/220371963-0dbd3ef5-a1db-474d-a976-900a71fd8cc4.png">
</p>
