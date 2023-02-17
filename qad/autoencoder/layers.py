import tensorflow as tf


class Conv1DTranspose(tf.keras.layers.Layer):

    """Custom 1d transposed convolution that expands to 2d output for vae decoder

    Attributes
    ----------
    activation: :class:`tensorflow.keras.activation`
        activation
    ConvTranspose: :class:`tensorflow.keras.Layer`
        sublayer
    ExpandChannel: :class:`tensorflow.keras.Lambda`
        sublayer
    filters: int
        number kernels
    kernel_initializer: :class:`tensorflow.keras.Initializer`
        kernel init
    kernel_sz: int
        kernel size
    SqueezeChannel: :class:`tensorflow.keras.Layer`
        sublayer
    """

    def __init__(self, filters, kernel_sz, activation, kernel_initializer, **kwargs):
        """Construtor.

        Parameters
        ----------
        filters: int
            number kernels
        activation: :class:`tensorflow.keras.activation`
            activation
        kernel_initializer: :class:`tensorflow.keras.Initializer`
            kernel init
        kernel_sz: int
            kernel size
        """
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.kernel_sz = kernel_sz
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=(self.kernel_sz, 1),
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
        )
        self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

    def call(self, inputs):
        """Call.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            call

        Returns
        -------
        :class:`tensorflow.Tensor`
            outputs
        """
        # expand input and kernel to 2D
        x = self.ExpandChannel(inputs)  # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        # call Conv2DTranspose
        x = self.ConvTranspose(x)
        # squeeze back to 1D and return
        x = self.SqueezeChannel(x)
        return x

    def get_config(self):
        """Get configuration of layer.

        Returns
        -------
        dict
            dictionary containing configuration used to initialize this layer
        """
        config = super(Conv1DTranspose, self).get_config()
        config.update(
            {
                "kernel_sz": self.kernel_sz,
                "filters": self.filters,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config


class StdNormalization(tf.keras.layers.Layer):
    """Normalizing input feature to std Gauss (mean 0, var 1).

    Attributes
    ----------
    mean_x: float
        mean of inputs
    std_x: float
        standard deviation of inputs
    """

    def __init__(self, mean_x, std_x, name="Std_Normalize", **kwargs):
        """Constructor.

        Parameters
        ----------
        mean_x: float
            mean of inputs
        std_x: float
            standard deviation of inputs
        name: str, optional
            name
        """
        kwargs.update({"name": name, "trainable": False})
        super(StdNormalization, self).__init__(**kwargs)
        self.mean_x = mean_x
        self.std_x = std_x

    def get_config(self):
        """Get configuration for layer.

        Returns
        -------
        dict
            dictionary containing configuration used to initialize this layer
        """
        config = super(StdNormalization, self).get_config()
        config.update({"mean_x": self.mean_x, "std_x": self.std_x})
        return config

    def call(self, x):
        """Call.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            call

        Returns
        -------
        :class:`tensorflow.Tensor`
            outputs
        """
        return (x - self.mean_x) / self.std_x


class StdUnnormalization(StdNormalization):
    """Rescaling feature to original domain"""

    def __init__(self, mean_x, std_x, name="Un_Normalize", **kwargs):
        """Constructor.

        Parameters
        ----------
        mean_x: float
            mean of inputs
        std_x: float
            standard deviation of inputs
        name: str, optional
            name
        """
        super(StdUnnormalization, self).__init__(
            mean_x=mean_x, std_x=std_x, name=name, **kwargs
        )

    def call(self, x):
        """Call.

        Parameters
        ----------
        inputs: :class:`tensorflow.Tensor`
            call

        Returns
        -------
        :class:`tensorflow.Tensor`
            outputs
        """
        return (x * self.std_x) + self.mean_x
