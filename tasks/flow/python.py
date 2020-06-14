import sys
sys.path.append('../..')
import feature.backbone.drn as DRN
import feature.pyramid.baseline as Pyramid
import model as Model

import flow_vis
import tensorflow as tf

class Estimator:

    def load_models(self,path):
        saver = tf.train.Saver()
        saver.restore(self.session,path)

    def __init__(self,model_path,image_size=(416,1024),profile=False):
        self.do_profiling=profile
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

        self.placeholder_image1 = tf.placeholder(dtype=tf.float32,shape=(image_size[0],image_size[1],3))
        self.placeholder_image2 = tf.placeholder(dtype=tf.float32,shape=(image_size[0],image_size[1],3))
        inputs=tf.stack([self.placeholder_image1,self.placeholder_image2],axis=0)

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
        self.flow=tf.reshape(tf.stack([flow_x,flow_y],axis=-1),[image_size[0],image_size[1],2])
        self.load_models(model_path)

    def vis_flow(self,flow):
      return flow_vis.flow_to_color(flow,convert_to_bgr=True)

    def __call__(self,image1,image2):

        fetches  = {'flow':self.flow}
        feed_dict= {self.placeholder_image1:image1,
                    self.placeholder_image2:image2}

        if not self.do_profiling:

          results=self.session.run(fetches,feed_dict=feed_dict)

        else:

          options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          profiler = tf.profiler.Profiler(self.session.graph)
          run_meta = tf.RunMetadata()
          results=self.session.run(fetches,feed_dict=feed_dict,options=options,run_metadata=run_meta)
          profiler.add_step(self.step,run_meta)
          profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder.trainable_variables_parameter()))
          opts = option_builder.ProfileOptionBuilder.time_and_memory()
          profiler.profile_operations(options=opts)
          opts = (option_builder.ProfileOptionBuilder(
              option_builder.ProfileOptionBuilder.time_and_memory())
              .with_step(self.step)
              .with_timeline_output("./time/time_%d.json"%self.step).build())
          profiler.profile_graph(options=opts)
          tf.profiler.advise(self.session.graph,run_meta=run_meta)

        return results['flow']








