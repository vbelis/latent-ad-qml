import setGPU
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import datetime
from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np

import data.data_sample as dasa
import util.persistence as pers
import models.autoencoder as auen
import anpofah.sample_analysis.sample_converter as saco
import pofah.util.utility_fun as utfu
import vande.vae.losses as loss


def compare_jet_images(test_ds, model):
    print(">>> plotting jet image comparison original vs predicted")
    batch = next(test_ds.as_numpy_iterator())
    for i in np.random.choice(len(batch), 3):
        particles = batch[i]
        img = saco.convert_jet_particles_to_jet_image(particles)
        plt.imshow(np.squeeze(img), cmap="viridis")
        plt.savefig("fig/img_orig_" + str(i) + ".png")
        plt.clf()
        particles_pred = model.predict(particles[np.newaxis, :, :])
        img_pred = saco.convert_jet_particles_to_jet_image(particles_pred)
        plt.imshow(np.squeeze(img_pred), cmap="viridis")
        plt.savefig("fig/img_pred_" + str(i) + ".png")


def train(
    data_sample,
    input_shape=(100, 3),
    latent_dim=6,
    epochs=10,
    read_n=int(1e4),
    act_latent=None,
):

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
    # print(model.summary())

    # tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

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
    # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.8, patience=7, verbose=1)])

    return model


# ****************************************#
#           Runtime Params
# ****************************************#

do_clustering = True

Parameters = namedtuple(
    "Parameters",
    "run_n epochs latent_dim read_n sample_id_train cluster_alg act_latent",
)
params = Parameters(
    run_n=50,
    epochs=200,
    latent_dim=8,
    read_n=int(1e6),
    sample_id_train="qcdSide",
    cluster_alg="kmeans",
    act_latent=tf.keras.activations.tanh,
)

model_path = pers.make_model_path(run_n=params.run_n, prefix="AE", mkdir=True)
data_sample = dasa.DataSample(params.sample_id_train)

# ****************************************#
#           Autoencoder
# ****************************************#

# train AE model
print(">>> training autoencoder run " + str(params.run_n))
ae_model = train(
    data_sample,
    epochs=params.epochs,
    latent_dim=params.latent_dim,
    read_n=params.read_n,
    act_latent=params.act_latent,
)

# model save
print(">>> saving autoencoder to " + model_path)
tf.saved_model.save(ae_model, model_path)
