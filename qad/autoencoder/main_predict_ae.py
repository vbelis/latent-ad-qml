import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from collections import namedtuple

#import pofah.jet_sample as jesa
#import pofah.util.sample_factory as safa
#import pofah.path_constants.sample_dict_file_parts_input as sdi
#import util.persistence as pers
#import pofah.util.event_sample as evsa
#import inference.predict_autoencoder as pred



def map_to_latent_space(data_sample, model) -> np.ndarray: # [N x Z]
    """Autoencoder mapping input space to latent representation.

    Parameters
    ----------
    data_sample: :class:`tensorflow.data.Dataset`
        :class:`tensorflow.data.Dataset`.from_tensor_slices(input_sample (:class:`numpy.ndarray`)).batch(`batch_size`)
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
    parser.add_option("-path", dest="path", help='model_path')
    (options,args) = parser.parse_args()

    # read in data sample
    data_sample = ...

    ae_model = tf.saved_model.load(model_path)

    latent_coords = map_to_latent_space(data_sample, ae_model)
