import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


"""
    pass datasample through autoencoder to obtain latent space representation and write to disk
    for further usage in clusering and pca
"""


#****************************************#
#           Runtime Params
#****************************************#

#sample_ids = ['qcdSide', 'qcdSideExt', 'qcdSig', 'qcdSigExt', 'GtoWW35na']
# sample_ids = ['AtoHZ35']
sample_ids = ['GtoWW15br']


Parameters = namedtuple('Parameters', ' run_n read_n')
params = Parameters(run_n=50, read_n=int(1e6))

paths = safa.SamplePathDirFactory(sdi.path_dict)
output_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.run_n)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


#****************************************#
#           Load Autoencoder
#****************************************#

# load model
model_path_ae = pers.make_model_path(run_n=params.run_n, date='20211110', prefix='AE') # todo: remove date from model path string

print('[main_predict_ae] >>> loading autoencoder ' + model_path_ae)
ae_model = tf.saved_model.load(model_path_ae)


for sample_id in sample_ids:

    #****************************************#
    #           Read Data
    #****************************************#

    sample = evsa.EventSample.from_input_dir(name=sample_id, path=paths.sample_dir_path(sample_id), read_n=params.read_n)
    p1, p2 = sample.get_particles() 


    #****************************************#
    #           Apply Autoencoder
    #****************************************#

    latent_coords = pred.map_to_latent_space(data_sample=np.vstack([p1, p2]), model=ae_model, read_n=params.read_n)
    latent_j1, latent_j2 = np.split(latent_coords, 2)
    latent_coords = np.stack([latent_j1,latent_j2], axis=1)

    #****************************************#
    #           Write results to list
    #****************************************#

    print('[main_predict_ae] >>> writing results for ' + sample_id + 'to ' + output_dir)

    # latent results
    sample_out = jesa.JetSampleLatent.from_event_sample(sample)
    sample_out.add_latent_representation(latent_coords)
    sample_out.dump(os.path.join(output_dir, sample_out.name+'.h5'))

