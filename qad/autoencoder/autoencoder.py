import numpy as np
import tensorflow as tf

import qad.autoencoder.layers as layers


class ParticleAutoencoder(tf.keras.Model):
    """Autoencoder model for dimensionality reduction.

    Attributes
    ----------
    input_shape: `tuple`, optional
        shape of input, default (100,3)
    latent_dim: int, optional
        size of the latent dimension, default 6
    x_mean_stdev: `tuple`, optional
        mean and standard deviation of inputs, default (0,1)
    kernel_n: int, optional
        number of kernels, default 16
    kernel_sz: int, optional
        kernel size, default 3
    activation: string, optional
        activation function, default "elu"
    activation_latent: :class:`tensorflow.keras.activations`, optional
        activation before bottleneck, default :class:tensorflow.keras.activations.linear`
    """

    def __init__(
        self,
        input_shape=(100, 3),
        latent_dim=6,
        x_mean_stdev=(0, 1),
        kernel_n=16,
        kernel_sz=3,
        activation="elu",
        activation_latent=tf.keras.activations.linear,
        **kwargs
    ):
        super(ParticleAutoencoder, self).__init__(**kwargs)
        self._input_shape = input_shape
        self.latent_dim = latent_dim
        self.kernel_n = kernel_n
        self.kernel_sz = kernel_sz
        self.activation = activation
        self.activation_latent = activation_latent
        self.initializer = "he_uniform"
        self.kernel_1D_sz = 3
        self.x_mean_stdev = x_mean_stdev
        self.encoder = self.build_encoder(*self.x_mean_stdev)
        self.decoder = self.build_decoder(*self.x_mean_stdev)

    def build_encoder(self, mean, stdev):
        """Builds encoder model

        Parameters
        ----------
        mean : float
            Mean of data.
        stdev : float
            Standard deviation of data.

        Returns
        -------
        :class:`tensorflow.keras.Model`
            encoder
        """

        inputs = tf.keras.layers.Input(
            shape=self._input_shape, dtype=tf.float32, name="encoder_input"
        )
        # normalize
        normalized = layers.StdNormalization(mean_x=mean, std_x=stdev)(inputs)
        # add channel dim
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(
            normalized
        )  # [B x 100 x 3] => [B x 100 x 3 x 1]
        # 2D Conv
        x = tf.keras.layers.Conv2D(
            filters=self.kernel_n,
            kernel_size=self.kernel_sz,
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(x)
        # Squeeze
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(
            x
        )  # remove width axis for 1D Conv [ B x int(100-kernel_width/2) x 1 x kernel_n ] -> [ B x int(100-kernel_width/2) x kernel_n ]
        # 1D Conv * 2
        self.kernel_n += 4
        x = tf.keras.layers.Conv1D(
            filters=self.kernel_n,
            kernel_size=self.kernel_1D_sz,
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # [ B x 96 x 10 ]
        self.kernel_n += 4
        x = tf.keras.layers.Conv1D(
            filters=self.kernel_n,
            kernel_size=self.kernel_1D_sz,
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # [ B x 94 x 14 ]
        # Pool
        x = tf.keras.layers.AveragePooling1D()(x)  # [ B x 47 x 14 ]
        # shape info needed to build decoder model
        self.shape_convolved = x.get_shape().as_list()
        # Flatten
        x = tf.keras.layers.Flatten()(x)  # [B x 658]
        # Dense * 3
        x = tf.keras.layers.Dense(
            int(self.latent_dim * 17),
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # reduce convolution output
        x = tf.keras.layers.Dense(
            int(self.latent_dim * 4),
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # reduce again
        # x = Dense(8, activation=self.activation, kernel_initializer=self.initializer)(x)

        # *****************************
        #         latent space
        z = tf.keras.layers.Dense(
            self.latent_dim, activation=self.activation_latent, name="z"
        )(x)

        # instantiate encoder model
        encoder = tf.keras.Model(name="encoder", inputs=inputs, outputs=z)
        encoder.summary()
        # plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
        return encoder

    def build_decoder(self, mean, stdev):
        """Builds decoder model

        Parameters
        ----------
        mean : float
            Mean of data.
        stdev : float
            Standard deviation of data.

        Returns
        -------
        :class:`tensorflow.keras.Model`
            decoder
        """
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name="z")
        # Dense * 3
        x = tf.keras.layers.Dense(
            int(self.latent_dim * 4),
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            latent_inputs
        )  # inflate to input-shape/200
        x = tf.keras.layers.Dense(
            int(self.latent_dim * 17),
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # double size
        x = tf.keras.layers.Dense(
            np.prod(self.shape_convolved[1:]),
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(x)
        # Reshape
        x = tf.keras.layers.Reshape(tuple(self.shape_convolved[1:]))(x)
        # Upsample
        x = tf.keras.layers.UpSampling1D()(x)  # [ B x 94 x 16 ]
        # 1D Conv Transpose * 2
        self.kernel_n -= 4
        x = layers.Conv1DTranspose(
            filters=self.kernel_n,
            kernel_sz=self.kernel_1D_sz,
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # [ B x 94 x 16 ] -> [ B x 96 x 8 ]
        self.kernel_n -= 4
        x = layers.Conv1DTranspose(
            filters=self.kernel_n,
            kernel_sz=self.kernel_1D_sz,
            activation=self.activation,
            kernel_initializer=self.initializer,
        )(
            x
        )  # [ B x 96 x 8 ] -> [ B x 98 x 4 ]
        # Expand
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(
            x
        )  #  [ B x 98 x 1 x 4 ]
        # 2D Conv Transpose
        x = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=self.kernel_sz, name="conv_2d_transpose"
        )(x)
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(
            x
        )  # [B x 100 x 3 x 1] -> [B x 100 x 3]
        outputs_decoder = layers.StdUnnormalization(mean_x=mean, std_x=stdev)(x)

        # instantiate decoder model
        decoder = tf.keras.Model(latent_inputs, outputs_decoder, name="decoder")
        decoder.summary()
        # plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
        return decoder

    @classmethod
    def load(cls, path):
        """Loads autoencoder.

        Parameters
        ----------
        path : str
            Model path.

        Returns
        -------
        :class:`tensorflow.keras.Model`
            autoencoder
        """
        custom_objects = {
            "Conv1DTranspose": layers.Conv1DTranspose,
            "StdNormalization": layers.StdNormalization,
            "StdUnnormalization": layers.StdUnnormalization,
        }
        encoder = tf.keras.models.load_model(
            os.path.join(path, "encoder.h5"),
            custom_objects=custom_objects,
            compile=False,
        )
        decoder = tf.keras.models.load_model(
            os.path.join(path, "decoder.h5"),
            custom_objects=custom_objects,
            compile=False,
        )
        model = tf.keras.models.load_model(
            os.path.join(path, "vae.h5"), custom_objects=custom_objects, compile=False
        )
        return encoder, decoder, model

    def compile(self, optimizer, reco_loss):
        """Compiles the autoencoder - set model's configuration.

        Parameters
        ----------
        optimizer : :class:`tensorflow.keras.Optimizer`
            optimizer.
        reco_loss : `Callable`
            reconstruction loss function

        """
        super(ParticleAutoencoder, self).compile()
        self.optimizer = optimizer
        self.reco_loss = reco_loss

    def call(self, x):
        """Calls the model.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            inputs

        Returns
        -------
        :class:`numpy.ndarray`
            outputs
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, x):
        """Training step.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            inputs

        Returns
        -------
        dict
            loss dictionary
        """

        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)  # Forward pass
            loss = self.reco_loss(x, x_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}

    def test_step(self, x):
        """Inference step.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            inputs

        Returns
        -------
        dict
            loss dictionary
        """

        x_pred = self(x, training=False)
        loss = tf.math.reduce_mean(self.reco_loss(x, x_pred))

        return {"loss": loss}
