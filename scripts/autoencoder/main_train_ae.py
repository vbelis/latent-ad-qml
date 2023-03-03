import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from collections import namedtuple
import numpy as np
import optparse
import h5py

import qad.autoencoder.autoencoder as auen
from qad.autoencoder.util import get_mean, get_std


def train(
    data_sample,
    input_shape=(100, 3),
    read_n=int(1e4),
    batch_size=256,
    latent_dim=6,
    epochs=10,
    act_latent=None,
):
    """Trains autoencoder

    Parameters
    ----------
    data_sample: :class:`numpy.ndarray`
        inputs
    input_shape: tuple, optional
        shape, default `(100, 3)`
    batch_size: int, optional
        batch size, default 256
    latent_dim: int, optional
        latent dim, default `6`
    epochs: int, optional
        number of epochs, default `10`
    act_latent: :class:`tensorflow.keras.Actication`, optional
        latent activation, default `None`
    """

    # data to tf.Dataset
    train_valid_split = int(len(data_sample) * 0.8)
    train_ds = tf.data.Dataset.from_tensor_slices(
        data_sample[:train_valid_split]
    ).batch(batch_size, drop_remainder=True)
    valid_ds = tf.data.Dataset.from_tensor_slices(
        data_sample[train_valid_split:]
    ).batch(batch_size, drop_remainder=True)

    model = auen.ParticleAutoencoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        x_mean_stdev=(get_mean(data_sample), get_std(data_sample)),
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
        ],
    )

    return model


# ****************************************#
#           run training
# ****************************************#

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option(
        "-read_data_path", dest="read_data_path", help="path to data file"
    )
    parser.add_option("-input_shape", dest="input_shape", help="input shape")
    parser.add_option("-batch_size", dest="batch_size", help="input shape")
    parser.add_option("-ld", dest="ld", help="latent_dim")
    parser.add_option("-ep", dest="ep", help="epochs")
    parser.add_option("-al", dest="al", help="latent activation")
    parser.add_option(
        "-model_path", dest="model_path", help="path for saving the model"
    )
    (options, args) = parser.parse_args()

    # read in data
    with h5py.File(options.read_data_path, "r") as file:
        data_sample = file["inputs"]
        data_sample = np.asarray(data_sample[:])

    ae_model = train(
        data_sample,
        options.input_shape,
        options.batch_size,
        options.ld,
        options.ep,
        options.al,
    )

    # model save
    print(">>> saving autoencoder to " + options.model_path)
    tf.saved_model.save(ae_model, options.model_path)
