
import tensorflow as tf
import sys
sys.path.append('../..')
from common.network import *

class DRN:#dilated resnet without dilation 

    def __init__(self,block=None,layers=None,
                 channels=(16, 32, 64, 128, 256, 512),is_trainings=5*[False],reuse_variables=False):

        self.channels=channels
        self.layers=layers
        self.block=block
        self.data_format='channels_first'
        self.is_trainings=is_trainings
        self.reuse_variables=reuse_variables

    def layer(self,name,inputs,block,filters,blocks,stride=1,dilation=1,new_level=True,residual=True,is_training=None):
        with tf.variable_scope(name):
            assert dilation == 1 or dilation % 2 == 0
            if stride != 1 or inputs.get_shape().as_list()[1] != filters * block.expansion:
                downsample=projection_shortcut
            else:
                downsample=None

            if stride==2:
                inputs =tf.nn.avg_pool(inputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')

            outputs=block('0',filters,1,downsample,dilation=(1,1) if dilation==1 else (dilation//2 if new_level else dilation, dilation),residual=residual,is_training=is_training,data_format=self.data_format).inference(inputs)
            for i in range(1,blocks):
                outputs=block(str(i),filters,residual=residual,dilation=(dilation,dilation),is_training=is_training,data_format=self.data_format).inference(outputs)
            return outputs

    def conv_layers(self,name,inputs,filters,convs,stride=1,dilation=1,is_training=None):
        with tf.variable_scope(name):
            
            outputs=inputs
            if stride==2:
                outputs=tf.nn.avg_pool(outputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')

            for i in range(convs):
                outputs=conv2d(str(2*i),inputs=outputs,filters=filters,padding=dilation,dilation=dilation,kernel_size=3,strides=1,data_format=self.data_format,is_training=is_training)
                outputs=batch_norm_relu(str(2*i+1),inputs=outputs,is_training=is_training,data_format=self.data_format)
            return outputs

    def drn22_no_dilation(self,inputs):

        self.block =building_block
        self.layers=[1,1,2,2,2,2]

        inputs=tf.nn.batch_normalization(inputs/255.0,tf.constant([0.485, 0.456, 0.406]),tf.constant([0.229*0.229,0.224*0.224,0.225*0.225]),offset=None,scale=None,variance_epsilon=0.0)
        if self.data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])

        with tf.variable_scope('DRN',reuse=self.reuse_variables):
            with tf.variable_scope('layer0'):
                self.conv1  = conv2d('0',inputs=inputs,filters=self.channels[0],kernel_size=7,strides=1,padding=3,is_training=self.is_trainings[-1],data_format=self.data_format)
                self.layer0 = batch_norm_relu('1',inputs=self.conv1,is_training=self.is_trainings[-1],data_format=self.data_format)
            self.layer1 = self.conv_layers('layer1',inputs=self.layer0,filters=self.channels[0],convs=self.layers[0],stride=1,is_training=self.is_trainings[-1])
            self.layer2 = self.conv_layers('layer2',inputs=self.layer1,filters=self.channels[1],convs=self.layers[1],stride=2,is_training=self.is_trainings[-1])
            self.layer3 = self.layer('layer3',inputs=self.layer2,block=self.block,filters=self.channels[2],blocks=self.layers[2],stride=2,is_training=self.is_trainings[-2])
            self.layer4 = self.layer('layer4',inputs=self.layer3,block=self.block,filters=self.channels[3],blocks=self.layers[3],stride=2,is_training=self.is_trainings[-3])
            self.layer5 = self.layer('layer5',inputs=self.layer4,block=self.block,filters=self.channels[4],blocks=self.layers[4],stride=2,is_training=self.is_trainings[-4])
            self.layer6 = self.layer('layer6',inputs=self.layer5,block=self.block,filters=self.channels[5],blocks=self.layers[5],stride=2,is_training=self.is_trainings[-5])
        return [self.layer6,self.layer5,self.layer4,self.layer3,self.layer2]
    