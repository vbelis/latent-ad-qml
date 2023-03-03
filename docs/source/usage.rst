The following are some examples on how to use the ``qad`` package to
train and test quantum models, and reproduce results from the paper. 

Unsupervised quantum kernel machine
===================================

The training and testing of the unsupervised kernel machine is accomplished using the
``train.py`` and ``test.py`` in ``scripts/kernel_machines/``,
respectively (see in the `repo <https://github.com/vbelis/latent-ad-qml/tree/main/scripts>`_). The configuration parameters of the model, e.g., quantum
or classical version, feature map, number of training samples, backend
used for the quantum computation, etc, are defined through the arguments
of the ``train.py`` and ``test.py`` scripts. For instance, to train the
model:

.. code:: python

   python train.py --sig_path /path/to/signal/data --bkg_path /path/to/background/data --test_bkg_path /path/to/test_background/data --unsup --nqubits 8 --feature_map u_dense_encoding --run_type ideal --output_folder quantum_test --nu_param 0.01 --ntrain 600 --quantum

To test the saved model:

.. code:: python

   python test.py --sig_path /path/to/signal/data --bkg_path /path/to/background/data --test_bkg_path /path/to/test_background/data --model trained_qsvms/quantum_test_nu\=0.01_ideal/

For details regarding different arguments of the ``train.py`` and ``test.py`` scripts
see below.

Train
^^^^^
.. autofunction:: scripts.kernel_machines.train.main

Test
^^^^
.. autofunction:: scripts.kernel_machines.test.main


Producing figures
=================

After the unsuperised quantum and classical kernel machines have been
trained and test scores have been saved, one can summarise their
performance with a ROC curve plot. Firstly, following our convention the
test scores are prepared for plotting using
``scripts/kernel_machines/scripts/prepare_plot_scores.py`` (`code <https://github.com/vbelis/latent-ad-qml/blob/docs-reformat/scripts/kernel_machines/prepare_plot_scores.py>`_),
and by running

.. code:: python

   python prepare_plot_scores.py --classical_folder trained_qsvms/c_test_nu\=0.01/ --quantum_folder trained_qsvms/q_test_nu\=0.01_ideal/ --out_path test_plot --name_suffix n<n_test>_k<k_folds>

Then, we load the score values from the saved files using our
convention, e.g. for the case of three different signals, with eight
latent dimensions, 600 training datapoints, 100k testing datapoints, and
k=5 folds

.. code:: python

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

The final ROC plot, as it appears in the paper in Fig. 3, can be
obtained

.. code:: python

   colors = ['forestgreen', '#EC4E20', 'darkorchid']
   legend_signal_names=['Narrow 'r'G $\to$ WW 3.5 TeV', r'A $\to$ HZ $\to$ ZZZ 3.5 TeV', 'Broad 'r'G $\to$ WW 1.5 TeV']
   pl.plot_ROC_kfold_mean(q_loss_qcd, q_loss_sig, c_loss_qcd, c_loss_sig, legend_signal_names, n_folds,\
                   legend_title=r'Anomaly signature', save_dir='../jupyter_plots', pic_id='test',
                   palette=colors, xlabel=r'$TPR$', ylabel=r'$FPR^{-1}$')

Example for the unsupervised kernel machine performance on different
anomalies:

.. raw:: html

    <p align="center">
        <img width="550" alt="image" src="https://user-images.githubusercontent.com/48251467/220371963-0dbd3ef5-a1db-474d-a976-900a71fd8cc4.png">
    </p>

.. raw:: html

   </p>

.. automodule:: qad.plot
   :members:
   :undoc-members:


Expressibility and entanglement capability analysis
===================================================

Given a data encoding quantum circuit we can compute its expressibility and entanglement
capability. These metrics along with the variance of the quantum kernel that is
constructed from the given quantum circuit can also be computed as function of the
number of qubits. 

These different computations of properties of the quantum feature map and the
corresponted quantu kernel can be computed using the script ``compute_expr_ent.py``.
The desired computation can be chosen using the ``argparse`` argument ``compute``.

.. code:: bash
   
   python compute_expr_ent.py --n_qubits 8 --n_shots 100000 --n_exp 20 --out_path test --compute expr_vs_qubits --data_path ../data/mpp_collab/journal_data/latent/lat8/latentrep_AtoHZ_to_ZZZ_35.h5 ../data/mpp_collab/journal_data/latent/lat16/latentrep_AtoHZ_to_ZZZ_35.h5 --data_dependent
   

.. automodule:: scripts.kernel_machines.compute_expr_ent
   :members:
   :undoc-members:



