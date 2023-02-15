Autoencoder
===========

Model
-----
.. autoclass:: qad.autoencoder.autoencoder.ParticleAutoencoder
   :members:
   :show-inheritance:

Layers
------
.. autoclass:: qad.autoencoder.layers.Conv1DTranspose
   :members:
   :show-inheritance:

.. autoclass:: qad.autoencoder.layers.StdNormalization
   :members:
   :show-inheritance:

.. autoclass:: qad.autoencoder.layers.StdUnnormalization
   :members:
   :show-inheritance:

Train
-----
.. autofunction:: scripts.autoencoder.main_train_ae.train

Predict
-------
.. autofunction:: scripts.autoencoder.main_predict_ae.map_to_latent_space
    