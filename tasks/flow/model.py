import numpy as np
import tensorflow as tf
from common.function import upsample2X,resampler
from lsm.context  import Context
from lsm.subspace import Subspace

class Model:
    
    data_format='channels_first'

    def __init__(self,levels=None,subspace_dims=None,group_dims=None,scale_factors=None,is_trainings=None,reuse_variables=None):

        self.levels=levels
        self.subspace_dims=subspace_dims
        self.group_dims=group_dims
        self.scale_factors=scale_factors
        self.is_trainings=is_trainings
        self.reuse_variables=reuse_variables

        self.block_size = [4,4,4,4]
        self.spp_size = [[3,5,7],[3,5,9],[3,5,9,13],[3,7,15,23]]
        self.spp_chan = [[2,2,1],[2,2,1],[2,1,1,1] ,[2,1,1,1]]
        self.iters    = [1,6,4,3]

    def derivative_2d(self,inputs):
        _,_,height,width= inputs.get_shape()
        gradx  = 0.5*(inputs[:,:,:,2:width]-inputs[:,:,:,0:width-2])
        gradx  = tf.concat([inputs[:,:,:,1:2]-inputs[:,:,:,0:1],gradx,inputs[:,:,:,width-1:width]-inputs[:,:,:,width-2:width-1]],axis=-1)
        
        grady  = 0.5*(inputs[:,:,2:height,:]-inputs[:,:,0:height-2,:])
        grady  = tf.concat([inputs[:,:,1:2,:]-inputs[:,:,0:1,:],grady,inputs[:,:,height-1:height,:]-inputs[:,:,height-2:height-1,:]],axis=-2)
        return gradx,grady

    def warp_2d(self,im1,im2,gradx=None,grady=None,x=None,y=None,height=None,width=None,warp_gradient=True):
        mask =tf.to_float(tf.logical_not(tf.reduce_any(tf.concat([x<0,x>float(width-1),y<0,y>float(height-1)],axis=1),axis=1,keepdims=True)))
        xy   =tf.concat([x,y],axis=1)
        im2  =resampler(im2,xy)
        diff =mask*(im1-im2)
        if warp_gradient:
            gradx=resampler(gradx,xy)
            grady=resampler(grady,xy)
            gradx=mask*gradx
            grady=mask*grady
            return diff,gradx,grady,mask
        else:
            return diff

    def singleIterationFlow(self,im1,im2,flow_0_x,flow_0_y,subspace_dim,group_dim,context_constructor,subspace_generator,reuse_variables):

        N,C,H,W = im2.get_shape()

        grad_x_2,grad_y_2 = self.derivative_2d(im2)
        grad_x_1,grad_y_1 = self.derivative_2d(im1)
        x0,y0             = tf.meshgrid(np.arange(int(W)),np.arange(int(H)))
        x0                = tf.to_float(x0)
        y0                = tf.to_float(y0)
        x0                = tf.tile(tf.expand_dims(tf.expand_dims(x0,axis=0),axis=1),[N,1,1,1])
        y0                = tf.tile(tf.expand_dims(tf.expand_dims(y0,axis=0),axis=1),[N,1,1,1])

        if flow_0_x is None or flow_0_y is None:
            diff = im1-im2
            grad_x=(grad_x_2+grad_x_1)/2
            grad_y=(grad_y_2+grad_y_1)/2
        else:
            x                         = x0+flow_0_x
            y                         = y0+flow_0_y
            diff,grad_x_2,grad_y_2,mask = self.warp_2d(im1,im2,grad_x_2,grad_y_2,x,y,int(H),int(W))
            grad_x=(mask*grad_x_1+grad_x_2)/2
            grad_y=(mask*grad_y_1+grad_y_2)/2

        with tf.name_scope('grouping_equations'):

            shape  =[N,C//group_dim,group_dim,H,W]
            grad_xy=tf.reduce_mean(tf.reshape(tf.multiply(grad_x,grad_y,name='grad_xy'),shape),axis=1)
            grad_x2=tf.reduce_mean(tf.reshape(tf.square(grad_x,name='grad_x2'),shape),axis=1)
            grad_y2=tf.reduce_mean(tf.reshape(tf.square(grad_y,name='grad_y2'),shape),axis=1)
            diff_x =tf.reduce_mean(tf.reshape(tf.multiply(grad_x,diff,name='diff_x'),shape),axis=1)
            diff_y =tf.reduce_mean(tf.reshape(tf.multiply(grad_y,diff,name='diff_y'),shape),axis=1)

        with tf.name_scope('transform_equations'):

            determinat  = tf.multiply(grad_x2,grad_y2)-tf.square(grad_xy)
            determinat_x= tf.multiply(diff_x,grad_y2) -tf.multiply(grad_xy,diff_y)
            determinat_y= tf.multiply(grad_x2,diff_y) -tf.multiply(grad_xy,diff_x)

            _,variance   = tf.nn.moments(determinat,axes=[1,2,3],keep_dims=True)
            std_inv_det  = tf.rsqrt(variance+1e-3)

            determinat   =std_inv_det*determinat
            determinat_x =std_inv_det*determinat_x
            determinat_y =std_inv_det*determinat_y

        with tf.name_scope('estimate_subspaces'):

            if flow_0_x is None and flow_0_y is None:

                subspace_feature_x=context_constructor([determinat,determinat_x],reuse_variables)
                subspace_feature_y=context_constructor([determinat,determinat_y],True)

            else:

                flow_0_mean_x,flow_0_var_x = tf.nn.moments(flow_0_x,axes=[1,2,3],keep_dims=True)
                inv_x                 = tf.rsqrt(flow_0_var_x+1e-3)
                flow_0_x_normalized   = inv_x*(flow_0_x-flow_0_mean_x)

                flow_0_mean_y,flow_0_var_y = tf.nn.moments(flow_0_y,axes=[1,2,3],keep_dims=True)
                inv_y                 = tf.rsqrt(flow_0_var_y+1e-3)
                flow_0_y_normalized   = inv_y*(flow_0_y-flow_0_mean_y)

                subspace_feature_x=context_constructor([determinat,inv_x*determinat_x,flow_0_x_normalized],reuse_variables)
                subspace_feature_y=context_constructor([determinat,inv_y*determinat_y,flow_0_y_normalized],True)

            subspace_x=subspace_generator(subspace_feature_x,reuse_variables)
            subspace_y=subspace_generator(subspace_feature_y,True)

            subspace_x,occ_x=tf.split(subspace_x,[subspace_dim,2],axis=1)
            subspace_y,occ_y=tf.split(subspace_y,[subspace_dim,2],axis=1)
            occ_x           =tf.nn.softmax(occ_x,axis=1)
            occ_y           =tf.nn.softmax(occ_y,axis=1)
            occ             =tf.reshape(occ_x[:,0,:,:]+occ_y[:,0,:,:],[N,H*W,1])/2
            occ            /=tf.reduce_max(occ,axis=1,keepdims=True)
            occ             =tf.maximum(occ,1e-1)

            with tf.name_scope('solve_equations'):

                subspace_x=tf.reshape(tf.transpose(subspace_x,[0,2,3,1]),[N,H*W,subspace_dim])
                subspace_y=tf.reshape(tf.transpose(subspace_y,[0,2,3,1]),[N,H*W,subspace_dim])

                _,variance_x=tf.nn.moments(subspace_x,axes=[1],keep_dims=True)
                subspace_x  =tf.rsqrt(variance_x+1e-3)*subspace_x

                _,variance_y=tf.nn.moments(subspace_y,axes=[1],keep_dims=True)
                subspace_y  =tf.rsqrt(variance_y+1e-3)*subspace_y

                shape  =[N,H*W,1]
                grad_xy=occ*tf.reshape(tf.reduce_mean(grad_xy,axis=1),shape)
                grad_x2=occ*tf.reshape(tf.reduce_mean(grad_x2,axis=1),shape)
                grad_y2=occ*tf.reshape(tf.reduce_mean(grad_y2,axis=1),shape)
                diff_x =occ*tf.reshape(tf.reduce_mean(diff_x ,axis=1),shape)
                diff_y =occ*tf.reshape(tf.reduce_mean(diff_y ,axis=1),shape)

                _,var   = tf.nn.moments((grad_x2+grad_y2)/2,axes=[1],keep_dims=True)
                std_dev = tf.rsqrt(var+1e-3)

                grad_xy=std_dev*grad_xy
                grad_x2=std_dev*grad_x2
                grad_y2=std_dev*grad_y2
                diff_x =std_dev*diff_x
                diff_y =std_dev*diff_y

                npixel=tf.to_float(H*W)
                x2    =tf.matmul(subspace_x,grad_x2*subspace_x,transpose_a=True)
                y2    =tf.matmul(subspace_y,grad_y2*subspace_y,transpose_a=True)
                xy    =tf.matmul(subspace_x,grad_xy*subspace_y,transpose_a=True)
                yx    =tf.matmul(subspace_y,grad_xy*subspace_x,transpose_a=True)

                diff_x=tf.matmul(diff_x,subspace_x,transpose_a=True)
                diff_y=tf.matmul(diff_y,subspace_y,transpose_a=True)
                hessian=tf.concat([tf.concat([x2,xy],axis=2),tf.concat([yx,y2],axis=2)],axis=1)

                difference=tf.reshape(tf.concat([diff_x,diff_y],axis=-1),[N,2*subspace_dim,1])

                if flow_0_x is not None and flow_0_y is not None:

                    projection_x=tf.matmul(tf.matrix_inverse(tf.matmul(subspace_x,subspace_x,transpose_a=True)),subspace_x,transpose_b=True)
                    projection_y=tf.matmul(tf.matrix_inverse(tf.matmul(subspace_y,subspace_y,transpose_a=True)),subspace_y,transpose_b=True)

                    flow_0_x_reshaped=tf.reshape(flow_0_x-flow_0_mean_x,[N,H*W,1])
                    flow_0_y_reshaped=tf.reshape(flow_0_y-flow_0_mean_y,[N,H*W,1])

                    projected_x =tf.matmul(projection_x,flow_0_x_reshaped)
                    projected_y =tf.matmul(projection_y,flow_0_y_reshaped)
                    projected_h =tf.matmul(hessian,tf.concat([projected_x,projected_y],axis=1))
                    
                    residual_x  =tf.matmul(subspace_x,grad_x2*flow_0_x_reshaped+grad_xy*flow_0_y_reshaped,transpose_a=True)
                    residual_y  =tf.matmul(subspace_y,grad_xy*flow_0_x_reshaped+grad_y2*flow_0_y_reshaped,transpose_a=True)
                    difference -=(projected_h-tf.concat([residual_x,residual_y],axis=1))

                with tf.device('/cpu:0'):
                    solution = tf.matrix_solve(hessian,difference)

                solution_x,solution_y=tf.split(solution,2,axis=1)
                flow_x    =tf.reshape(tf.matmul(subspace_x,solution_x),[N,1,H,W])
                flow_y    =tf.reshape(tf.matmul(subspace_y,solution_y),[N,1,H,W])


            if flow_0_x is not None and flow_0_y is not None:

                flow_x =tf.reshape(tf.matmul(subspace_x,projected_x),[N,1,H,W])+flow_x+flow_0_mean_x
                flow_y =tf.reshape(tf.matmul(subspace_y,projected_y),[N,1,H,W])+flow_y+flow_0_mean_y

            return flow_x,flow_y

    
    def Test(self,layers):

        flow_x    = None
        flow_y    = None

        for level in range(self.levels):

            im1,im2=tf.split(layers[level],2,axis=0)
            N,C,H,W=im1.get_shape()
            
            with tf.variable_scope('LSM_%d'%level,reuse=self.reuse_variables):
                
                context_constructor= Context(im1,self.group_dims[level],self.spp_size[level],self.spp_chan[level],self.is_trainings[level],self.reuse_variables)
                subspace_generator = Subspace(self.subspace_dims[level],self.is_trainings[level],[C//4]*self.block_size[level])

                flow_x,flow_y=self.singleIterationFlow(im1,im2,flow_0_x=flow_x,flow_0_y=flow_y,
                                                       subspace_dim=self.subspace_dims[level],group_dim=self.group_dims[level],
                                                       context_constructor=context_constructor,subspace_generator=subspace_generator,reuse_variables=self.reuse_variables)
                if level!=0 and self.iters[level]>1:

                    def cond(cur_iter,*argv):
                        return cur_iter<self.iters[level]

                    def body(cur_iter,flow_x,flow_y):
                            flow_x,flow_y=self.singleIterationFlow(im1,im2,flow_0_x=flow_x,flow_0_y=flow_y,
                                                                   subspace_dim=self.subspace_dims[level],group_dim=self.group_dims[level],
                                                                   context_constructor=context_constructor,subspace_generator=subspace_generator,reuse_variables=True)
                            cur_iter=cur_iter+1
                            return cur_iter,flow_x,flow_y

                    cur_iter=1
                    _,flow_x,flow_y=tf.while_loop(cond,body,[cur_iter,flow_x,flow_y],back_prop=False,parallel_iterations=1)

            flow_x=2*upsample2X(flow_x)
            flow_y=2*upsample2X(flow_y)
        
        flow_x=2*upsample2X(flow_x)
        flow_y=2*upsample2X(flow_y)

        return flow_x,flow_y









