import tensorflow as tf
from common.network import bottleneck_block,conv2d,projection_shortcut

class Subspace:
    def __init__(self,subspace_dim,is_training,filters=None,data_format='channels_first'):
        self.subspace_dim=subspace_dim
        self.is_training =is_training
        self.filters=filters
        self.data_format=data_format

    def __call__(self,inputs,reuse_variables):
        with tf.variable_scope('SUBSPACE',reuse=reuse_variables):
            for i in range(len(self.filters)):
                inputs=bottleneck_block('%d'%i,self.filters[i],1,projection_shortcut,residual=True,is_training=self.is_training,data_format=self.data_format).inference(inputs)
            return conv2d('subspace',inputs,self.subspace_dim+2,3,is_training=self.is_training,data_format=self.data_format,use_bias=True)
