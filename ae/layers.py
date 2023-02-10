import tensorflow as tf


# custom 1d transposed convolution that expands to 2d output for vae decoder
class Conv1DTranspose(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_sz, activation, kernel_initializer, **kwargs):
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.kernel_sz = kernel_sz
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=(self.kernel_sz,1), activation=self.activation, kernel_initializer=self.kernel_initializer)
        self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

    def call(self, inputs):
        # expand input and kernel to 2D
        x = self.ExpandChannel(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        # call Conv2DTranspose
        x = self.ConvTranspose(x)
        # squeeze back to 1D and return
        x = self.SqueezeChannel(x)
        return x

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config.update({'kernel_sz': self.kernel_sz, 'filters': self.filters, 'activation': self.activation, 'kernel_initializer': self.kernel_initializer})
        return config


class StdNormalization(tf.keras.layers.Layer):
    """normalizing input feature to std Gauss (mean 0, var 1)"""
    def __init__(self, mean_x, std_x, name='Std_Normalize', **kwargs):
        kwargs.update({'name': name, 'trainable': False})
        super(StdNormalization, self).__init__(**kwargs)
        self.mean_x = mean_x
        self.std_x = std_x

    def get_config(self):
        config = super(StdNormalization, self).get_config()
        config.update({'mean_x': self.mean_x, 'std_x': self.std_x})
        return config

    def call(self, x):
        return (x - self.mean_x) / self.std_x


class StdUnnormalization(StdNormalization):
    """ rescaling feature to original domain """

    def __init__(self, mean_x, std_x, name='Un_Normalize', **kwargs):
        super(StdUnnormalization, self).__init__(mean_x=mean_x, std_x=std_x, name=name, **kwargs)

    def call(self, x):
        return (x * self.std_x) + self.mean_x


class MinMaxNormalization(tf.keras.layers.Layer):
    """normalizing input feature to std Gauss (mean 0, var 1)"""
    def __init__(self, min_x, max_x, name='MinMax_Normalize', **kwargs):
        kwargs.update({'name': name, 'trainable': False})
        super(MinMaxNormalization, self).__init__(**kwargs)
        self.min_x = min_x
        self.max_x = max_x

    def get_config(self):
        config = super(MinMaxNormalization, self).get_config()
        config.update({'min_x': self.min_x, 'max_x': self.max_x})
        return config

    def call(self, x):
        return (x - self.min_x) / (self.max_x - self.min_x)


class MinMaxUnnormalization(MinMaxNormalization):
    """ rescaling feature to original domain """

    def __init__(self, min_x, max_x, name='MinMax_Un_Normalize', **kwargs):
        super(MinMaxUnnormalization, self).__init__(min_x=min_x, max_x=max_x, name=name, **kwargs)

    def call(self, x):
        return x * (self.max_x - self.min_x) + self.min_x



class LogTransform(tf.keras.layers.Layer):
    ''' logarithmic transformation (shifted to 0,0) '''

    def __init__(self, x_min, **kwargs):
        (super(LogTransform, self).__init__)(**kwargs)
        self.x_min = x_min

    def get_config(self):
        config = super(LogTransform, self).get_config()
        config.update({'x_min': self.x_min})
        return config

    def call(self, x):
        return tf.math.log(x - self.x_min + 1.0)

