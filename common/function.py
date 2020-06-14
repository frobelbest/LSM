import tensorflow as tf
import numpy as np

if tf.__version__<'1.14.0':
    
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import nn_ops
    from tensorflow.python.ops import array_ops

    @ops.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
    def _DepthwiseConv2DNativeBackpropInputGrad(op, grad):
      return [None,
              nn_ops.depthwise_conv2d_native_backprop_filter(
                  grad,
                  array_ops.shape(op.inputs[1]),
                  op.inputs[2],
                  op.get_attr("strides"),
                  op.get_attr("padding"),
                  data_format=op.get_attr("data_format")),
              nn_ops.depthwise_conv2d_native(
                  grad,
                  op.inputs[1],
                  op.get_attr("strides"),
                  op.get_attr("padding"),
                  data_format=op.get_attr("data_format"))]

def upsample2X(inputs):
        kernel=[[[[ 0.0625,  0.1875,  0.1875,  0.0625],
                  [ 0.1875,  0.5625,  0.5625,  0.1875],
                  [ 0.1875,  0.5625,  0.5625,  0.1875],
                  [ 0.0625,  0.1875,  0.1875,  0.0625]]]]
        filters      =int(inputs.get_shape()[1])
        filter_kernel=np.tile(kernel,(filters,1,1,1))
        filter_kernel=np.transpose(filter_kernel,(2,3,0,1)).astype(np.float32)
        shape        =inputs.get_shape()
        inputs       =tf.pad(inputs,[[0,0],[0,0],[1,1],[1,1]],'SYMMETRIC')
        output       =tf.nn.depthwise_conv2d_native_backprop_input([shape[0],shape[1],2*shape[2]+4,2*shape[3]+4],tf.constant(filter_kernel),inputs,strides=(1,1,2,2),padding='SAME',data_format='NCHW')
        output       =output[:,:,2:-2,2:-2]
        return output

def resampler(inputs,xy):
    inputs=tf.transpose(inputs,[0,2,3,1])
    inputs=tf.contrib.resampler.resampler(inputs,tf.transpose(xy,[0,2,3,1]))
    inputs=tf.transpose(inputs,[0,3,1,2])
    return inputs