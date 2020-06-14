import numpy as np
import tensorflow as tf
from common.network import projection_shortcut

class Context:

    def __init__(self,inputs,group_dim,pooling_size,pooling_chan,is_training,reuse_variables):

        self.data_format='channels_first'
        self.pooling_size = pooling_size
        self.pooling_chan = [group_dim*x for x in pooling_chan]
        self.is_training=is_training

        image_ctx = projection_shortcut(inputs,group_dim,stride=1,is_training=is_training,data_format=self.data_format,reuse_variables=reuse_variables,name='downsample') 
        _,variance = tf.nn.moments(image_ctx,axes=[1,2,3],keep_dims=True)
        self.image_ctx = tf.rsqrt(variance+1e-3)*image_ctx

    def __call__(self,inputs,reuse_variables):

        inputs   =tf.concat([self.image_ctx]+inputs,axis=1)
        N,C,H,W  =inputs.get_shape()
        outputs  =[projection_shortcut(inputs,2*(C//2),stride=1,is_training=self.is_training,data_format='channels_first',reuse_variables=reuse_variables,name='spp_0')]
       
        x_int,y_int=np.meshgrid(np.arange(int(W)),np.arange(int(H)))
        x_int= x_int.flatten()
        y_int= y_int.flatten()


        index00s=[]
        index01s=[]
        index10s=[]
        index11s=[]
        areas  =[]

        for i in range(len(self.pooling_size)):

            half_size = self.pooling_size[i]//2
            offset_y0 = np.maximum(y_int-half_size,0)
            offset_x0 = np.maximum(x_int-half_size,0)
            offset_y1 = np.minimum(y_int+1+half_size,int(H))
            offset_x1 = np.minimum(x_int+1+half_size,int(W))

            index00=(offset_y0*(W+1)+offset_x0).astype(np.int32)
            index01=(offset_y0*(W+1)+offset_x1).astype(np.int32)
            index10=(offset_y1*(W+1)+offset_x0).astype(np.int32)
            index11=(offset_y1*(W+1)+offset_x1).astype(np.int32)
            area   =np.reshape((offset_x1-offset_x0)*(offset_y1-offset_y0),(H*W,1)).astype(np.float32)

            index00s.append(tf.constant(index00,dtype=tf.int32))
            index01s.append(tf.constant(index01,dtype=tf.int32))
            index10s.append(tf.constant(index10,dtype=tf.int32))
            index11s.append(tf.constant(index11,dtype=tf.int32))
            areas.append(tf.constant(area,dtype=tf.float32))

        inputs        = tf.transpose(inputs,[2,3,0,1])
        intergral     = tf.pad(tf.cumsum(tf.cumsum(inputs,axis=0,exclusive=False),axis=1,exclusive=False),[[1,0],[1,0],[0,0],[0,0]])
        intergral     = tf.reshape(intergral,[(H+1)*(W+1),N*C])

        for i in range(len(self.pooling_size)):
            output = (tf.gather(intergral,index00s[i])+tf.gather(intergral,index11s[i])
                     -tf.gather(intergral,index10s[i])-tf.gather(intergral,index01s[i]))/areas[i]
            output  = tf.reshape(tf.transpose(output,[1,0]),[N,C,H,W])
            output  = projection_shortcut(output,self.pooling_chan[i],stride=1,is_training=self.is_training,data_format='channels_first',reuse_variables=reuse_variables,name='spp_%d'%(self.pooling_size[i]+1))
            outputs.append(output)
        return tf.concat(outputs,axis=1)

