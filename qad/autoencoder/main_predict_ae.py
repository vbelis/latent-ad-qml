import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from collections import namedtuple
import pathlib

import pofah.jet_sample as jesa
import pofah.util.sample_factory as safa
import pofah.path_constants.sample_dict_file_parts_input as sdi
import util.persistence as pers
import pofah.util.event_sample as evsa
import inference.predict_autoencoder as pred



def map_to_latent_space(data_sample, model) -> np.ndarray: # [N x Z]
     """prediction by autoencoder

    Parameters
    ----------
    data_sample: tf.data.Dataset.from_tensor_slices(data_sample).batch(2048)
        inputs
    model: tf.keras.Model
        the autoencoder

    Returns
    ----------
    np.ndarray
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
