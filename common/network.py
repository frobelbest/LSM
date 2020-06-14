import tensorflow as tf

_BATCH_NORM_DECAY = 0.95
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(name,inputs,is_training,data_format):
    batch_norm_training=(is_training and (not fixed_batchnorm))
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,center=True,
        scale=True,training=batch_norm_training,trainable=batch_norm_training,fused=True,name=name,reuse=tf.AUTO_REUSE)
    return inputs

def batch_norm_relu(name,inputs,is_training,data_format):
    inputs = batch_norm(name,inputs,is_training,data_format)
    inputs = tf.nn.relu(inputs)
    return inputs

def symmetric_padding(inputs,padding,data_format):
    with tf.name_scope('padding'):
        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [padding, padding], [padding, padding]],mode='SYMMETRIC')
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]],mode='SYMMETRIC')
    return padded_inputs

#should do reflect padding!zero padding is stupid!
def conv2d(name,inputs,filters,kernel_size,strides=1,padding=1,dilation=1,data_format='channels_first',is_training=False,use_bias=False,activation=None):

    if kernel_size > 1 and padding > 0:
        inputs = symmetric_padding(inputs,padding,data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='VALID', use_bias=use_bias,
        kernel_initializer=tf.initializers.orthogonal(),activation=activation,
        data_format=data_format,name=name,dilation_rate=(dilation,dilation),trainable=is_training)

def projection_shortcut(inputs,filters,stride,is_training,data_format,reuse_variables=tf.AUTO_REUSE,name='downsample'):
    with tf.variable_scope(name,reuse=reuse_variables):
        inputs = conv2d('0',inputs=inputs, filters=filters, kernel_size=1, strides=stride,data_format=data_format,is_training=is_training)
        inputs = batch_norm('1',inputs,is_training,data_format)
        return inputs

class residual_block:

    def __init__(self,name,filters,strides,downsample,dilation,residual,is_training,data_format,activation=None):
        pass

    def inference(self,inputs):
        return None

class building_block(residual_block):
    
    expansion = 1
    
    def __init__(self,name,filters,strides=1,downsample=None,dilation=(1,1),residual=True,is_training=False,data_format='channels_first',activation=tf.nn.relu):

        
        self.name=name

        #origin parameters
        self.filters=filters
        self.strides=strides
        self.downsample=downsample
        self.dilation=dilation
        self.residual=residual

        #tensorflow parameters
        self.is_training=is_training
        self.data_format=data_format
        self.activation=activation


    def inference(self,inputs):

        with tf.variable_scope(self.name):

            if self.residual:
                if self.downsample is None:
                    shortcut = inputs
                else:
                    shortcut = self.downsample(inputs,self.filters*self.expansion,self.strides,self.is_training,self.data_format)
    
            inputs = conv2d('conv1',inputs=inputs, filters=self.filters, kernel_size=3, strides=self.strides,
                            padding=self.dilation[0],dilation=self.dilation[0],data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm_relu('bn1',inputs,self.is_training,self.data_format)
            
            inputs = conv2d('conv2',inputs=inputs, filters=self.filters, kernel_size=3, strides=1,
                            padding=self.dilation[1],dilation=self.dilation[1],data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm('bn2',inputs,self.is_training,self.data_format)

            if self.residual:
                return self.activation(inputs + shortcut)
            else:
                return self.activation(inputs)


class bottleneck_block(residual_block):

    expansion = 4

    def __init__(self,name,filters,strides=1,downsample=None,dilation=(1,1),residual=True,is_training=False,data_format='channels_first',activation=tf.nn.relu):
        
        self.name=name

        #origin parameters
        self.filters=filters
        self.strides=strides
        self.downsample=downsample
        self.dilation=dilation
        self.residual=residual

        #tensorflow parameters
        self.is_training=is_training
        self.data_format=data_format
        self.activation=activation

    def inference(self,inputs):
        with tf.variable_scope(self.name):

            if self.downsample is None:
                shortcut = inputs
            else:
                shortcut = self.downsample(inputs,self.filters*self.expansion,self.strides,self.is_training,self.data_format)

            inputs = conv2d('conv1',inputs=inputs, filters=self.filters, kernel_size=1, strides=1,data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm_relu('bn1',inputs,self.is_training,self.data_format)

            inputs = conv2d('conv2',inputs=inputs, filters=self.filters, kernel_size=3, strides=self.strides,
                            padding=self.dilation[1],dilation=self.dilation[1],data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm_relu('bn2',inputs,self.is_training,self.data_format)

            inputs = conv2d('conv3',inputs=inputs, filters=self.expansion*self.filters, kernel_size=1, strides=1,data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm('bn3',inputs,self.is_training,self.data_format)

            if self.residual:
                return self.activation(inputs + shortcut)
            else:
                return self.activation(inputs)