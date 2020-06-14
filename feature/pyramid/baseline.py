import tensorflow as tf
from common.function import upsample2X
from common.network import conv2d,batch_norm_relu,projection_shortcut

class Pyramid:

    def __init__(self,levels,is_trainings,reuse_variables):
        self.levels=levels
        self.is_trainings=is_trainings
        self.reuse_variables=reuse_variables
        self.layer_dict={}
        self.data_format='channels_first'

    def aggregation(self,input1,input2,filters,name,is_training):
        with tf.variable_scope(name):
            inputs=tf.concat([input1,input2],axis=1)
            output=conv2d('conv2d',inputs,filters,1,is_training=is_training)
            output=batch_norm_relu('batch_norm_relu',output,is_training,self.data_format)
            return output

    def level_feature(self,level,is_training,reuse_variables):

        with tf.variable_scope(('DLA_%d'%level),reuse=reuse_variables):

            layer_name_1='layer_{0}_{1}'.format(0,level)
            input1=self.layer_dict[layer_name_1]

            if level==0:
                layer_name='layer_{}_{}'.format(level,0)
                self.layer_dict[layer_name]=input1
            else:
                
                layer_name_2='layer_{0}_{1}'.format(level-1,0)
                input2=self.layer_dict[layer_name_2]

                input2=projection_shortcut(input2,int(input1.get_shape()[1]),stride=1,is_training=is_training,data_format=self.data_format,reuse_variables=reuse_variables)
                input2=tf.nn.relu(input2)
                
                upsample_name='upsample_{}_{}'.format(level,0)
                input2=upsample2X(input2)

                aggregate_name='aggregation_{}_{}'.format(level,0)
                aggregated=self.aggregation(input1,input2,int(input1.get_shape()[1]),aggregate_name,is_training)

                layer_name='layer_{}_{}'.format(level,0)
                self.layer_dict[layer_name]=aggregated

    def __call__(self,inputs):

        outputs=[]

        for l in range(self.levels):
            layer_name='layer_0_{}'.format(l)
            self.layer_dict[layer_name]=inputs[l]
        
        for level in range(self.levels):
            self.level_feature(level,is_training=self.is_trainings[level],reuse_variables=self.reuse_variables)
            features = self.layer_dict['layer_%d_%d'%(level,0)]
            _,C,_,_ = features.get_shape()
            features = projection_shortcut(features,C,stride=1,is_training=self.is_trainings[level],data_format=self.data_format,reuse_variables=self.reuse_variables,name='project%d'%level)
            outputs.append(features)
        
        return outputs  