import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
#import datetime
from collections import namedtuple
#from matplotlib import pyplot as plt
import numpy as np
import optparse

import qad.autoencoder.autoencoder as auen



def train(
    data_sample,
    input_shape=(100, 3),
    latent_dim=6,
    epochs=10,
    read_n=int(1e4),
    act_latent=None,
):
    """Trains autoencoder

    Parameters
    ----------
    data_sample: :class:`numpy.ndarray`
        inputs
    input_shape: tuple, optional
        shape, default `(100, 3)`
    latent_dim: int, optional 
        latent dim, default `6`
    epochs: int, optional
        number of epochs, default `10`
    read_n: int, optional
        number of inputs, default `int(1e4)`
    act_latent: :class:`tensorflow.keras.Actication`, optional
        latent activation, default `None`
    """

    # get data
    train_ds, valid_ds = data_sample.get_datasets_for_training(
        read_n=read_n, test_dataset=False
    )
    model = auen.ParticleAutoencoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        x_mean_stdev=data_sample.get_mean_and_stdev(),
        activation_latent=act_latent,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), reco_loss=loss.threeD_loss)

    model.fit(
        train_ds,
        epochs=epochs,
        shuffle=True,
        validation_data=valid_ds,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=70, verbose=1
            ),
            tensorboard_callback,
        ],
    )

    return model


# ****************************************#
#           run training
# ****************************************#

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option("-data_sample", dest="data_sample", help="input data as numpy.array")
    parser.add_option("-input_shape", dest="input_shape", help='input shape')
    parser.add_option("-ld", dest="ld", help='latent_dim')
    parser.add_option("-ep", dest="ep", help='epochs')
    parser.add_option("-rn", dest="rn", help="read_n")
    parser.add_option("-al", dest="al", help='latent activation')
    (options,args) = parser.parse_args()

    # read in data sample - example
    data_sample = options.data_sample

    ae_model = train(data_sample, options.input_shape, options.ld, options.ep, options.rn, options.al)

    # model save
    print(">>> saving autoencoder to " + model_path)
    tf.saved_model.save(ae_model, model_path)
