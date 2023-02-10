import tensorflow as tf


class Conv1DTranspose(tf.keras.layers.Layer):

    """ custom 1d transposed convolution that expands to 2d output for vae decoder
    
    Attributes:
        activation (tf.keras.activation): activation
        ConvTranspose (tf.keras.Layer): sublayer
        ExpandChannel (tf.keras.Lambda): sublayer
        filters (int): number kernels
        kernel_initializer (tf.keras.Initializer): kernel init
        kernel_sz (int): kernel size
        SqueezeChannel (tf.keras.Layer): sublayer
    """
    
    def __init__(self, filters, kernel_sz, activation, kernel_initializer, **kwargs):
        """construtor
        
        Args:
            filters (int): number kernels
            activation (tf.keras.activation): activation
            kernel_initializer (tf.keras.Initializer): kernel init
            kernel_sz (int): kernel size
            **kwargs: additional params
        """
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.kernel_sz = kernel_sz
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.ExpandChannel = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.ConvTranspose = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=(self.kernel_sz,1), activation=self.activation, kernel_initializer=self.kernel_initializer)
        self.SqueezeChannel = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

    def call(self, inputs):
        """call
        
        Args:
            inputs (tf.Tensor): call
        
        Returns:
            tf.Tensor: outputs
        """
        # expand input and kernel to 2D
        x = self.ExpandChannel(inputs) # [ B x 98 x 4 ] -> [ B x 98 x 1 x 4 ]
        # call Conv2DTranspose
        x = self.ConvTranspose(x)
        # squeeze back to 1D and return
        x = self.SqueezeChannel(x)
        return x

    def get_config(self):
        """get config for layer
        
        Returns:
            TYPE: config of layer
        """
        config = super(Conv1DTranspose, self).get_config()
        config.update({'kernel_sz': self.kernel_sz, 'filters': self.filters, 'activation': self.activation, 'kernel_initializer': self.kernel_initializer})
        return config


class StdNormalization(tf.keras.layers.Layer):
    """normalizing input feature to std Gauss (mean 0, var 1)
    
    Attributes:
        mean_x (float): mean of inputs
        std_x (float): std-dev of inputs
    """
    def __init__(self, mean_x, std_x, name='Std_Normalize', **kwargs):
        """constructor
        
        Args:
            mean_x (float): mean of inputs
            std_x (float): std-dev of inputs
            name (str, optional): name
            **kwargs: addiational params
        """
        kwargs.update({'name': name, 'trainable': False})
        super(StdNormalization, self).__init__(**kwargs)
        self.mean_x = mean_x
        self.std_x = std_x

    def get_config(self):
        """get config for layer
        
        Returns:
            TYPE: config of layer
        """
        config = super(StdNormalization, self).get_config()
        config.update({'mean_x': self.mean_x, 'std_x': self.std_x})
        return config

    def call(self, x):
       """call
        
        Args:
            inputs (tf.Tensor): call
        
        Returns:
            td.Tensor: outputs
        """
        return (x - self.mean_x) / self.std_x


class StdUnnormalization(StdNormalization):
    """rescaling feature to original domain 
    """

    def __init__(self, mean_x, std_x, name='Un_Normalize', **kwargs):
        """constructor
        
        Args:
            mean_x (float): mean of inputs
            std_x (float): std-dev of inputs
            name (str, optional): name
            **kwargs: addiational params
        """
        super(StdUnnormalization, self).__init__(mean_x=mean_x, std_x=std_x, name=name, **kwargs)

    def call(self, x):
         """call
        
        Args:
            inputs (tf.Tensor): call
        
        Returns:
            td.Tensor: outputs
        """
        return (x * self.std_x) + self.mean_x

