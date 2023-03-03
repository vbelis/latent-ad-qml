import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from collections import namedtuple
import optparse
import h5py


def map_to_latent_space(data_sample, model) -> np.ndarray:  # [N x Z]
    """Autoencoder mapping input space to latent representation.

    Parameters
    ----------
    data_sample: :class:`tensorflow.data.Dataset`
        `tensorflow.data.Dataset.from_tensor_slices(input_sample (:class:`numpy.ndarray`)).batch(`batch_size`)`
    model: :class:`tensorflow.keras.Model`
        the autoencoder

    Returns
    ----------
    :class:`numpy.ndarray`
        latent representation
    """

    latent_coords = []

    for batch in data_sample:
        # run encoder
        coords = model.encoder(batch)
        latent_coords.append(coords)

    # return latent (per jet?)
    return np.concatenate(latent_coords, axis=0)


# ****************************************#
#           run prediction
# ****************************************#

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option(
        "-read_data_path", dest="read_data_path", help="path to data file"
    )
    parser.add_option("-batch_size", dest="batch_size", help="batch_size")
    parser.add_option("-model_path", dest="path", help="path to model file")
    (options, args) = parser.parse_args()

    # read in data sample
    with h5py.File(options.read_data_path, "r") as file:
        data_sample = file["test_data"]
        data_sample = np.asarray(data_sample[:])

    test_ds = train_ds = tf.data.Dataset.from_tensor_slices(data_sample).batch(
        options.batch_size
    )

    ae_model = tf.saved_model.load(options.model_path)

    latent_coords = map_to_latent_space(test_ds, ae_model)
