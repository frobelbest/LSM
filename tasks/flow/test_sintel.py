import dataset_utils
import sys
sys.path.append('../..')
import feature.backbone.drn as DRN
import feature.pyramid.baseline as Pyramid
import model as Model

import tensorflow as tf
import numpy as np

batch_size=1
num_epochs=1
dataset_utils.do_crop=False

resize_width =1024
resize_height=416

with tf.device('/cpu:0'):
    im1s,im2s,ests,msks=dataset_utils.dataset_loader(batch_size=batch_size,
                                             num_epochs=num_epochs,
                                             do_shuffle=False,
                                             shuffle_size=64,
                                             dataset_list=dataset_utils.load_data_flow_sintel_png(),
                                             data_parser=dataset_utils.parser_flow_sintel_png)

    data_width =dataset_utils.crop_width
    data_height=dataset_utils.crop_height
    num_training_samples=dataset_utils.num_training_samples

    im1s.set_shape([batch_size,data_height,data_width,3])
    im2s.set_shape([batch_size,data_height,data_width,3])
    ests.set_shape([batch_size,data_height,data_width,2])
    msks.set_shape([batch_size,data_height,data_width])

    ests  =tf.image.resize_images(ests,size=[resize_height,resize_width])
    ests.set_shape([batch_size,resize_height,resize_width,2])

    print(ests.get_shape())

steps_per_epoch=int(num_training_samples/batch_size)
num_total_steps=steps_per_epoch*num_epochs

with tf.device('/gpu:0'):

    inputs = tf.concat([im1s,im2s],axis=0)
    inputs = tf.image.resize_images(inputs,size=[resize_height,resize_width])

    reuse_variables = False
    backbone_trainings = [False,False,False,False,False]
    drn = DRN.DRN(is_trainings=backbone_trainings,reuse_variables=reuse_variables)
    backbone_features = drn.drn22_no_dilation(inputs)

    is_trainings = [False,False,False,False]
    levels = 4 
    pyramid = Pyramid.Pyramid(levels=levels,is_trainings=backbone_trainings,reuse_variables=reuse_variables)
    pyramid_features = pyramid(backbone_features)


    model=Model.Model(levels=levels,subspace_dims=[2, 4, 8,16],group_dims=[32,16,8,4],scale_factors=[32,16,8,4],is_trainings=is_trainings)

    flow_x,flow_y=model.Test(pyramid_features)
    flow_y   =(data_height/resize_height)*flow_y
    flow_x   =(data_width/resize_width)*flow_x

    estimated=tf.concat([flow_x,flow_y],axis=1)
    acc      =tf.reduce_mean(tf.norm(tf.abs(estimated-tf.transpose(ests,[0,3,1,2])),axis=1),axis=[-2,-1])

config = tf.ConfigProto(allow_soft_placement=True,inter_op_parallelism_threads=20)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_saver=tf.train.Saver()
train_saver.restore(sess,'G:/iccv19_/v4/flow_2019_10_23_10_54_22/epoch/depth_32724.ckpt')

_acc=[]
print(num_training_samples)
for step in range(num_training_samples//batch_size):
    _acc.append(sess.run(acc))
    print('Step %d' %(step))
print('median:',np.median(_acc),'mean:',np.mean(_acc))