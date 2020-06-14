import numpy as np
import os
import tensorflow as tf
#import specialops as sops
import sys
do_crop      =True
#global variables
input_width  =None
input_height =None

crop_width   =None
crop_height  =None

valid_thres  =0.7

grid_x       =None
grid_y       =None#;tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
max_flow_x   =None#;tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
min_flow_x   =None#;tf.expand_dims(tf.to_float(-grid_x),axis=-1)
max_flow_y   =None#;tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
min_flow_y   =None#;tf.expand_dims(tf.to_float(-grid_y),axis=-1)

outlier=[1549,
        2195,
        2570,
        4013,
        4053,
        5473,
        5590,
        5853,
        6181,
        6361,
        7571,
        9511, 
        10820,
        10936,
        10937,
        10938,
        10939,
        11787,
        12223,
        12530,
        13051]

num_training_samples=None
def load_data_disp():

  global input_width,input_height,crop_width,crop_height
  input_width =960
  input_height=540

  crop_width  =768
  crop_height =512


  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)

  _list=list(set(range(21818))-set(outlier))
  _list=_list+_list
  _list=np.random.permutation(_list)

  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  for i in range(len(_list)/2):

      im1.append('/local-scratch4/flythings/image_clean/right/%07d.png'%(_list[2*i]))
      im2.append('/local-scratch4/flythings/image_clean/left/%07d.png'%(_list[2*i]))
      est.append('/local-scratch4/flythings/disparity/right/%07d.pfm'%(_list[2*i]))
      msk.append('/local-scratch4/flythings/disparity_occlusions/right/%07d.png'%(_list[2*i]))

      im1.append('/local-scratch4/flythings/image_clean/left/%07d.png'%(_list[2*i+1]))
      im2.append('/local-scratch4/flythings/image_clean/right/%07d.png'%(_list[2*i+1]))
      est.append('/local-scratch4/flythings/disparity/left/%07d.pfm'%(_list[2*i+1]))
      msk.append('/local-scratch4/flythings/disparity_occlusions/left/%07d.png'%(_list[2*i+1]))


  im1=np.asarray(im1)
  im2=np.asarray(im2)
  est=np.asarray(est)
  msk=np.asarray(msk)
  
  return  tf.convert_to_tensor(im1,dtype=tf.string),\
          tf.convert_to_tensor(im2,dtype=tf.string),\
          tf.convert_to_tensor(est,dtype=tf.string),\
          tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_disp_val():


  global input_width,input_height,crop_width,crop_height
  input_width =960
  input_height=540

  crop_width  =768
  crop_height =512


  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)

  
  _list=range(4247)+range(4247)
  #_list=np.random.permutation(_list)

  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  for i in range(4247):

      im1.append('/local-scratch2/flythings/val/image_clean/right/%07d.png'%(_list[2*i]))
      im2.append('/local-scratch2/flythings/val/image_clean/left/%07d.png'%(_list[2*i]))
      est.append('/local-scratch2/flythings/val/disparity/right/%07d.pfm'%(_list[2*i]))
      msk.append('/local-scratch2/flythings/val/disparity_occlusions/right/%07d.png'%(_list[2*i]))

      # im1.append('/local-scratch2/flythings/val/image_clean/left/%07d.png'%(_list[2*i+1]))
      # im2.append('/local-scratch2/flythings/val/image_clean/right/%07d.png'%(_list[2*i+1]))
      # est.append('/local-scratch2/flythings/val/disparity/left/%07d.pfm'%(_list[2*i+1]))
      # msk.append('/local-scratch2/flythings/val/disparity_occlusions/left/%07d.png'%(_list[2*i+1]))


  im1=np.asarray(im1)
  im2=np.asarray(im2)
  est=np.asarray(est)
  msk=np.asarray(msk)
  
  return  tf.convert_to_tensor(im1,dtype=tf.string),\
          tf.convert_to_tensor(im2,dtype=tf.string),\
          tf.convert_to_tensor(est,dtype=tf.string),\
          tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow():



  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _list=range(21818)+range(21818)+range(21818)+range(21818)
  _flag=[0]*21818+[1]*21818+[2]*21818+[3]*21818
  _list=np.random.permutation(zip(*(_list,_flag)))

  for x in _list:
    if x[1]==0:
      flo_name='/local-scratch4/flythings/flow/right/into_future/%07d.flo'%(x[0])
      if os.path.isfile(flo_name):
        im1_name='/local-scratch4/flythings/image_clean/right/%07d.png'%(x[0])
        im2_name='/local-scratch4/flythings/image_clean/right/%07d.png'%(x[0]+1)
        msk_name='/local-scratch4/flythings/flow_occlusions/right/into_future/%07d.png'%(x[0])
      else:
        continue

    if x[1]==1:
      flo_name='/local-scratch4/flythings/flow/right/into_past/%07d.flo'%(x[0])
      if os.path.isfile(flo_name):
        im1_name='/local-scratch4/flythings/image_clean/right/%07d.png'%(x[0])
        im2_name='/local-scratch4/flythings/image_clean/right/%07d.png'%(x[0]-1)
        msk_name='/local-scratch4/flythings/flow_occlusions/right/into_past/%07d.png'%(x[0])
      else:
        continue

    if x[1]==2:
      flo_name='/local-scratch4/flythings/flow/left/into_future/%07d.flo'%(x[0])
      if os.path.isfile(flo_name):
        im1_name='/local-scratch4/flythings/image_clean/left/%07d.png'%(x[0])
        im2_name='/local-scratch4/flythings/image_clean/left/%07d.png'%(x[0]+1)
        msk_name='/local-scratch4/flythings/flow_occlusions/left/into_future/%07d.png'%(x[0])
      else:
        continue

    if x[1]==3:
      flo_name='/local-scratch4/flythings/flow/left/into_past/%07d.flo'%(x[0])
      if os.path.isfile(flo_name):
        im1_name='/local-scratch4/flythings/image_clean/left/%07d.png'%(x[0])
        im2_name='/local-scratch4/flythings/image_clean/left/%07d.png'%(x[0]-1)
        msk_name='/local-scratch4/flythings/flow_occlusions/left/into_past/%07d.png'%(x[0])
      else:
        continue

    im1.append(im1_name)
    im2.append(im2_name)
    est.append(flo_name)
    msk.append(msk_name)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

isnan=[
'into_past left 53',
'into_past right 53',
'into_future left 119',
'into_past left 121',
'into_past right 160',
'into_past left 161',
'into_past right 161',
'into_past left 162',
'into_past left 163',
'into_future left 879',
'into_past left 931',
'into_past right 1549',
'into_past right 1648',
'into_future left 1717',
'into_future left 3147',
'into_past right 3147',
'into_future right 3148',
'into_past right 3148',
'into_future left 3149',
'into_future right 3666',
'into_past left 4045',
'into_future left 4117',
'into_future right 4117',
'into_future left 4118',
'into_future left 4154',
'into_future left 4304',
'into_future left 4573',
'into_past left 4705',
'into_past left 4876',
'into_past right 5034',
'into_past left 5054',
'into_past right 5054',
'into_past left 5055',
'into_past right 5055',
'into_future left 6336',
'into_future right 6336',
'into_future left 6337',
'into_past left 6878',
'into_past right 6878',
'into_future left 6922',
'into_future left 11530',
'into_future left 14658',
'into_future left 15148',
'into_future left 15748',
'into_future left 16948',
'into_future left 17578']

islarge=[
53,#0.38252315
122,#0.47328126
133,#0.2963889
161,#0.28109762
162,#0.56125385
503,#0.30331597
581,#0.40748456
582,#0.2901196
1241,#0.42666087
1242,#0.77047455
1243,#0.6298148
1532,#0.28546104
1820,#0.42461997
1821,#0.2570737
2192,#0.33331984
2208,#0.54843366
2209,#0.6551331
2537,#0.7437365
2538,#0.72033566
2568,#0.30147183
2569,#0.59368443
2570,#0.33401814
2769,#0.26384452
3049,#0.27149883
3149,#0.34396604
4047,#0.28521606
4117,#0.3841956
4236,#0.3442496
4237,#0.29596257
4815,#0.3401331
4816,#0.82109183
4817,#0.9002488
4818,#0.89008486
5035,#0.25196952
5054,#0.265245
5055,#0.26618248
5295,#0.61877316
5296,#0.92711616
5297,#0.6266242
5298,#0.2948611
5493,#0.29461807
5555,#0.2560494
5556,#0.361549
5558,#0.41671103
5644,#0.25313273
5645,#0.43596643
5975,#0.28637153
6336,#0.3506983
6555,#0.26929784
6772,#0.93500966
6773,#0.8865471
6774,#0.38354746
6874,#0.31327352
6875,#0.9027932
6876,#0.838505
6877,#0.8610089
7566,#0.30568093
7567,#0.2563773
11270,#0.35209492
11784,#0.34818673
11785,#0.28549576
11786,#0.3572203
12223,#0.61923224
12530,#0.4167689
12543,#0.46024692
12544,#0.462174
12873,#0.25681326
13052,#0.34279707
13663,#0.5118615
13664,#0.37516397
]

def load_data_flow_png():

  global input_width,input_height,crop_width,crop_height
  input_width =960
  input_height=540

  crop_width  =768
  crop_height =512

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)

  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _ids=list(range(21818))+list(range(21818))+list(range(21818))+list(range(21818))
  _left_right =[1]*21818+[0]*21818+[0]*21818+[1]*21818
  _future_past=[0]*21818+[1]*21818+[1]*21818+[0]*21818
  _list=np.random.permutation(list(zip(*(_ids,_left_right,_future_past))))

  left_right =['left'  ,'right']
  future_past=['future','past']

  for x in _list:
    y='into_%s'%future_past[x[2]]+' %s'%left_right[x[1]]+' %d'%x[0]
    if y in isnan:
      continue

    if x[2]==0 and x[0] in islarge:
      continue
    
    if x[2]==1 and (x[0]-1) in islarge:
      continue
    
    flo_name='G:/FlyingThings3D_subset/train/flow/%s/into_%s/%07d.png'%(left_right[x[1]],future_past[x[2]],x[0])
    if os.path.isfile(flo_name):
      im1_name='G:/FlyingThings3D_subset/train/image_clean/%s/%07d.png'%(left_right[x[1]],x[0])
      
      if x[2]==0:
        im2_name='G:/FlyingThings3D_subset/train/image_clean/%s/%07d.png'%(left_right[x[1]],x[0]+1)
      else:
        im2_name='G:/FlyingThings3D_subset/train/image_clean/%s/%07d.png'%(left_right[x[1]],x[0]-1)

      msk_name='G:/FlyingThings3D_subset/train/flow_occlusions/%s/into_%s/%07d.png'%(left_right[x[1]],future_past[x[2]],x[0])

      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)
  
  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_png2():

  global input_width,input_height,crop_width,crop_height
  input_width =960
  input_height=540

  crop_width  =768
  crop_height =512

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)

  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  file=open('G:/FlyingThings3D_subset/train/info.txt')
  _list=file.read().splitlines()
  data =[]
  for x in _list:
    y=x.split(' ')
    y=(int(y[0]),int(y[1]),int(y[2]),float(y[3]),float(y[4]))
    if y[3]<0.2 and y[4]<0.2:
      data.append(y[0:3])
  
  data=np.random.permutation(data)
  left_right =['left'  ,'right']
  future_past=['future','past']

  for x in data:

    y='into_%s'%future_past[x[2]]+' %s'%left_right[x[1]]+' %d'%x[0]

    if y in isnan:
      continue

    flo_name='G:/FlyingThings3D_subset/train/flow/%s/into_%s/%07d.png'%(left_right[x[1]],future_past[x[2]],x[0])
    if os.path.isfile(flo_name):

      im1_name='G:/FlyingThings3D_subset/train/image_clean/%s/%07d.png'%(left_right[x[1]],x[0])

      if x[2]==0:
        im2_name='G:/FlyingThings3D_subset/train/image_clean/%s/%07d.png'%(left_right[x[1]],x[0]+1)
      else:
        im2_name='G:/FlyingThings3D_subset/train/image_clean/%s/%07d.png'%(left_right[x[1]],x[0]-1)

      msk_name='G:/FlyingThings3D_subset/train/flow_occlusions/%s/into_%s/%07d.png'%(left_right[x[1]],future_past[x[2]],x[0])
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_chairs():

  global input_width,input_height,crop_width,crop_height
  input_width =512
  input_height=384

  crop_width  =480
  crop_height =352


  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _ids=list(range(22232))+list(range(22232))
  _future_past=[0]*22232+[1]*22232
  _list=np.random.permutation(list(zip(*(_ids,_future_past))))
  future_past=['01','10']

  for x in _list:
    flo_name='G:/FlyingChairs2/train/%07d-flow_%s.png'%(x[0],future_past[x[1]])
    im1_name='G:/FlyingChairs2/train/%07d-img_%s.png'%(x[0],future_past[x[1]][0])
    im2_name='G:/FlyingChairs2/train/%07d-img_%s.png'%(x[0],future_past[x[1]][1])
    occ_name='G:/FlyingChairs2/train/%07d-occ_%s.png'%(x[0],future_past[x[1]])

    im1.append(im1_name)
    im2.append(im2_name)
    est.append(flo_name)
    msk.append(occ_name)
  
  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_chairs_val():

  global input_width,input_height,crop_width,crop_height
  input_width =512
  input_height=384

  crop_width  =512
  crop_height =384


  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _ids=list(range(640))+list(range(640))
  _future_past=[0]*640+[1]*640
  _list=list(zip(*(_ids,_future_past)))
  future_past=['01','10']

  for x in _list:
    flo_name='G:/FlyingChairs2/val/%07d-flow_%s.png'%(x[0],future_past[x[1]])
    im1_name='G:/FlyingChairs2/val/%07d-img_%s.png'%(x[0],future_past[x[1]][0])
    im2_name='G:/FlyingChairs2/val/%07d-img_%s.png'%(x[0],future_past[x[1]][1])
    occ_name='G:/FlyingChairs2/val/%07d-occ_%s.png'%(x[0],future_past[x[1]])

    im1.append(im1_name)
    im2.append(im2_name)
    est.append(flo_name)
    msk.append(occ_name)
  
  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_png2_val():

  global input_width,input_height,crop_width,crop_height
  input_width =960
  input_height=540

  crop_width  =960
  crop_height =540

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)

  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  file=open('G:/FlyingThings3D_subset/val/isnan.txt')
  _list=file.read().splitlines()
  file.close()
  _isnan=[]
  for x in _list:
    y=x.split(' ')
    y=(int(y[0]),int(y[1]),int(y[2]))
    _isnan.append(y)


  file=open('G:/FlyingThings3D_subset/val/info.txt')
  _list=file.read().splitlines()
  file.close()



  data =[]
  for x in _list:
    y=x.split(' ')
    y=(int(y[0]),int(y[1]),int(y[2]),float(y[3]),float(y[4]))
    if y[0:3] in _isnan:
        print(y)
        continue

    if y[3]<0.2 and y[4]<0.2:
      data.append(y[0:3])
  
  #data=np.random.permutation(data)
  left_right =['left'  ,'right']
  future_past=['future','past']

  

  for x in data:

    # y='into_%s'%future_past[x[2]]+' %s'%left_right[x[1]]+' %d'%x[0]
    # #print("%d %d %d"%(x[0],x[1],x[2]))
    # if "%d %d %d"%(int(x[0]),int(x[1]),int(x[2])) in _isnan:
    #   print(x)
    #   continue

    flo_name='G:/FlyingThings3D_subset/val/flow/%s/into_%s/%07d.png'%(left_right[x[1]],future_past[x[2]],x[0])
    if os.path.isfile(flo_name):

      im1_name='G:/FlyingThings3D_subset/val/image_clean/%s/%07d.png'%(left_right[x[1]],x[0])

      if x[2]==0:
        im2_name='G:/FlyingThings3D_subset/val/image_clean/%s/%07d.png'%(left_right[x[1]],x[0]+1)
      else:
        im2_name='G:/FlyingThings3D_subset/val/image_clean/%s/%07d.png'%(left_right[x[1]],x[0]-1)

      msk_name='G:/FlyingThings3D_subset/val/flow_occlusions/%s/into_%s/%07d.png'%(left_right[x[1]],future_past[x[2]],x[0])

      #print(im1_name,im2_name,flo_name,msk_name)
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)
  global num_training_samples
  num_training_samples=len(im1)
  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_sintel(do_shuffle=False):

  global input_width,input_height,crop_width,crop_height
  input_width =1024
  input_height=436

  crop_width  =1024
  crop_height =436

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _list=range(1063)
  if do_shuffle:
    _list=np.random.permutation(_list)
  for x in _list:
    flo_name='G:/sintel_flow/flow/%05d.png'%(x)
    if os.path.isfile(flo_name):
      im1_name='G:/sintel_flow/clean/%05d.png'%(x)
      im2_name='G:/sintel_flow/clean/%05d.png'%(x+1)
      msk_name='G:/sintel_flow/occlusion/%05d.png'%(x)
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_sintel_real_final_png():

  global input_width,input_height,crop_width,crop_height
  input_width =1024
  input_height=436

  crop_width  =768
  crop_height =320

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _list=range(1063)
  _list1=np.random.permutation(_list)
  _list2=np.random.permutation(_list)

  for i in range(1063):
    x=_list1[i]
    flo_name='G:/sintel_flow/flow/%05d.png'%(x)
    if os.path.isfile(flo_name):
      im1_name='G:/sintel_flow/clean/%05d.png'%(x)
      im2_name='G:/sintel_flow/clean/%05d.png'%(x+1)
      msk_name='G:/sintel_flow/occlusion/%05d.png'%(x)
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

    x=_list2[i]
    flo_name='G:/sintel_flow/flow/%05d.png'%(x)
    if os.path.isfile(flo_name):
      im1_name='G:/sintel_flow/final/%05d.png'%(x)
      im2_name='G:/sintel_flow/final/%05d.png'%(x+1)
      msk_name='G:/sintel_flow/occlusion/%05d.png'%(x)
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_sintel_real_final_png_val():

  global input_width,input_height,crop_width,crop_height
  input_width =1024
  input_height=436

  crop_width  =768
  crop_height =320

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _list=range(1063)
  for i in range(1063):
    x=_list[i]
    flo_name='G:/sintel_flow/flow/%05d.png'%(x)
    if os.path.isfile(flo_name):
      im1_name='G:/sintel_flow/clean/%05d.png'%(x)
      im2_name='G:/sintel_flow/clean/%05d.png'%(x+1)
      msk_name='G:/sintel_flow/occlusion/%05d.png'%(x)
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

    x=_list[i]
    flo_name='G:/sintel_flow/flow/%05d.png'%(x)
    if os.path.isfile(flo_name):
      im1_name='G:/sintel_flow/final/%05d.png'%(x)
      im2_name='G:/sintel_flow/final/%05d.png'%(x+1)
      msk_name='G:/sintel_flow/occlusion/%05d.png'%(x)
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_sintel_test():

  global input_width,input_height,crop_width,crop_height
  input_width =1024
  input_height=436

  crop_width  =768
  crop_height =436

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  file=open('G:/sintel_flow/test/list.txt','r')
  lines=file.read().splitlines()
  lines=np.random.permutation(lines)
  for x in lines:
    y=x
    index=int(x[-8:-4])+1
    y=y.replace(y[-8:-4],'%04d'%index)

    if os.path.isfile(y):
      im1_name=x
      im2_name=y
      msk_name=x
      flo_name=y
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_flow_sintel_png(do_shuffle=False):

  global input_width,input_height,crop_width,crop_height
  input_width =1024
  input_height=436

  crop_width  =1024
  crop_height =436

  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _list=range(1063)
  if do_shuffle:
    _list=np.random.permutation(_list)
  for x in _list:
    flo_name='G:/sintel_flow/flow/%05d.png'%(x)
    if os.path.isfile(flo_name):
      im1_name='G:/sintel_flow/clean/%05d.png'%(x)
      im2_name='G:/sintel_flow/clean/%05d.png'%(x+1)
      msk_name='G:/sintel_flow/occlusion/%05d.png'%(x)
      
      im1.append(im1_name)
      im2.append(im2_name)
      est.append(flo_name)
      msk.append(msk_name)

  global num_training_samples
  num_training_samples=len(im1)

  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)

def load_data_disp_sintel():

  global input_width,input_height,crop_width,crop_height
  input_width =1024
  input_height=436

  crop_width  =1024
  crop_height =436


  global grid_x,grid_y,max_flow_x,min_flow_x,max_flow_y,min_flow_y
  grid_x,grid_y=tf.meshgrid(np.arange(int(crop_width)),np.arange(int(crop_height)))
  max_flow_x   =tf.expand_dims(tf.to_float(crop_width-grid_x-1),axis=-1)
  min_flow_x   =tf.expand_dims(tf.to_float(-grid_x),axis=-1)
  max_flow_y   =tf.expand_dims(tf.to_float(crop_height-grid_y-1),axis=-1)
  min_flow_y   =tf.expand_dims(tf.to_float(-grid_y),axis=-1)


  im1 =[]
  im2 =[]
  est =[]
  msk =[]

  _list=range(1063)
  _list=np.random.permutation(_list)

  for x in _list:

    disp_name='/local-scratch4/dataset/sintel_disp/disp/%05d.png'%(x)
    im1_name ='/local-scratch4/dataset/sintel_disp/clean/left/%05d.png'%(x)
    im2_name ='/local-scratch4/dataset/sintel_disp/clean/right/%05d.png'%(x)
    msk_name ='/local-scratch4/dataset/sintel_disp/occlusion/%05d.png'%(x)
      
    im1.append(im1_name)
    im2.append(im2_name)
    est.append(disp_name)
    msk.append(msk_name)


  return tf.convert_to_tensor(im1,dtype=tf.string),\
         tf.convert_to_tensor(im2,dtype=tf.string),\
         tf.convert_to_tensor(est,dtype=tf.string),\
         tf.convert_to_tensor(msk,dtype=tf.string)



def flat_map_impl(im1_path,im2_path,est_path,msk_path):
  return tf.data.Dataset.from_tensors((tf.read_file(im1_path),tf.read_file(im2_path),tf.read_file(est_path),tf.read_file(msk_path)))

def parser_disp(im1_str,im2_str,est_str,msk_str):
  
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  sops.decode_pfm(est_str)
  msk  =  tf.image.decode_image(msk_str)

  offset_x= tf.random_uniform([],maxval=(input_width-crop_width),dtype=tf.int32)
  offset_y= tf.random_uniform([],maxval=(input_height-crop_height),dtype=tf.int32)

  im1 = tf.image.crop_to_bounding_box(im1,offset_y,offset_x,crop_height,crop_width)
  im2 = tf.image.crop_to_bounding_box(im2,offset_y,offset_x,crop_height,crop_width)
  est = tf.image.crop_to_bounding_box(est,offset_y,offset_x,crop_height,crop_width)
  msk = tf.equal(tf.image.crop_to_bounding_box(msk,offset_y,offset_x,crop_height,crop_width),255)

  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,est<min_flow_x,est>max_flow_x],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  # mean= tf.reduce_mean(msk)
  # msk = tf.to_float(mean>valid_thres)*msk
  # msk =tf.cond(mean>valid_thres,lambda:tf.ones(msk.get_shape()),lambda:tf.zeros(msk.get_shape()))
  # msk = tf.ones([crop_height,crop_width])
  return tf.to_float(im1),tf.to_float(im2),est,msk

def parser_disp_sintel(im1_str,im2_str,est_str,msk_str):
  
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  tf.image.decode_png(est_str,dtype=tf.uint8)
  msk  =  tf.image.decode_image(msk_str)
  est  = -(tf.to_float(est[:,:,0:1])*4+tf.to_float(est[:,:,1:2])/(2**6)) 
  
  offset_x= tf.random_uniform([],maxval=(input_width-crop_width),dtype=tf.int32)
  offset_y= tf.random_uniform([],maxval=(input_height-crop_height),dtype=tf.int32)

  im1 = tf.image.crop_to_bounding_box(im1,offset_y,offset_x,crop_height,crop_width)
  im2 = tf.image.crop_to_bounding_box(im2,offset_y,offset_x,crop_height,crop_width)
  est = tf.image.crop_to_bounding_box(est,offset_y,offset_x,crop_height,crop_width)
  msk = tf.equal(tf.image.crop_to_bounding_box(msk,offset_y,offset_x,crop_height,crop_width),255)

  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,est<min_flow_x,est>max_flow_x],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  return tf.to_float(im1),tf.to_float(im2),est,

def parser_disp_sintel_val(im1_str,im2_str,est_str,msk_str):
  
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  tf.image.decode_png(est_str,dtype=tf.uint8)
  msk  =  tf.image.decode_image(msk_str)
  est  = -(tf.to_float(est[:,:,0:1])*4+tf.to_float(est[:,:,1:2])/(2**6)) 
  msk = tf.equal(msk,255)

  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,est<min_flow_x,est>max_flow_x],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  return tf.to_float(im1),tf.to_float(im2),est,msk

def parser_disp_val(im1_str,im2_str,est_str,msk_str):
  
  # im1  =  sops.resample(tf.to_float(tf.expand_dims(tf.transpose(tf.image.decode_image(im1_str),[2,0,1]),axis=0)),size=[crop_height,crop_width])
  # im2  =  sops.resample(tf.to_float(tf.expand_dims(tf.transpose(tf.image.decode_image(im2_str),[2,0,1]),axis=0)),size=[crop_height,crop_width])

  # est  =  sops.resample(tf.reshape(sops.decode_pfm(est_str),[1,1,input_height,input_width]),size=[crop_height,crop_width])
  # msk  =  sops.resample(tf.to_float(tf.equal(tf.reshape(tf.image.decode_image(msk_str),[1,1,input_height,input_width]),255)),size=[crop_height,crop_width])
  
  # im1  =  tf.reshape(im1,[3,crop_height,crop_width])
  # im2  =  tf.reshape(im2,[3,crop_height,crop_width])
  # est  =  tf.reshape(est,[1,crop_height,crop_width])
  # msk  =  tf.reshape(msk,[crop_height,crop_width])
  # return im1,im2,(float(crop_width)/float(input_width))*est,msk
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  sops.decode_pfm(est_str)
  msk  =  tf.image.decode_image(msk_str)

  im1 = tf.image.crop_to_bounding_box(im1,(input_height-crop_height)/2,(input_width-crop_width)/2,crop_height,crop_width)
  im2 = tf.image.crop_to_bounding_box(im2,(input_height-crop_height)/2,(input_width-crop_width)/2,crop_height,crop_width)
  est = tf.image.crop_to_bounding_box(est,(input_height-crop_height)/2,(input_width-crop_width)/2,crop_height,crop_width)
  msk = tf.equal(tf.image.crop_to_bounding_box(msk,(input_height-crop_height)/2,(input_width-crop_width)/2,crop_height,crop_width),255)

  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,est<min_flow_x,est>max_flow_x],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  mean= tf.reduce_mean(msk)
  msk = tf.to_float(mean>valid_thres)*msk
  msk =tf.cond(mean>valid_thres,lambda:tf.ones(msk.get_shape()),lambda:tf.zeros(msk.get_shape()))
  
  return tf.to_float(im1),tf.to_float(im2),est,msk

def parser_flow(im1_str,im2_str,est_str,msk_str):
  
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  sops.decode_flo(est_str)
  msk  =  tf.image.decode_image(msk_str)

  offset_x= tf.random_uniform([],maxval=(input_width-crop_width),dtype=tf.int32)
  offset_y= tf.random_uniform([],maxval=(input_height-crop_height),dtype=tf.int32)

  print(input_width,crop_width,input_height,crop_height)

  im1 = tf.image.crop_to_bounding_box(im1,offset_y,offset_x,crop_height,crop_width)
  im2 = tf.image.crop_to_bounding_box(im2,offset_y,offset_x,crop_height,crop_width)
  est = tf.image.crop_to_bounding_box(est,offset_y,offset_x,crop_height,crop_width)
  msk = tf.equal(tf.image.crop_to_bounding_box(msk,offset_y,offset_x,crop_height,crop_width),255)
  
  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,est[:,:,0:1]<min_flow_x,est[:,:,0:1]>max_flow_x,est[:,:,1:2]<min_flow_y,est[:,:,1:2]>max_flow_y],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  mean= tf.reduce_mean(msk)
  msk = tf.cond(mean>valid_thres,lambda:tf.ones(msk.get_shape()),lambda:tf.zeros(msk.get_shape()))
  return tf.to_float(im1),tf.to_float(im2),est,msk

def parser_flow_sintel_png(im1_str,im2_str,est_str,msk_str):
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  tf.image.decode_png(est_str,channels=4,dtype=tf.uint8)
  msk  =  tf.image.decode_image(msk_str)

  if do_crop:

    offset_x= tf.random_uniform([],maxval=(input_width-crop_width),dtype=tf.int32)
    offset_y= tf.random_uniform([],maxval=(input_height-crop_height),dtype=tf.int32)

    im1 = tf.image.crop_to_bounding_box(im1,offset_y,offset_x,crop_height,crop_width)
    im2 = tf.image.crop_to_bounding_box(im2,offset_y,offset_x,crop_height,crop_width)
    est = tf.image.crop_to_bounding_box(est,offset_y,offset_x,crop_height,crop_width)
    msk = tf.image.crop_to_bounding_box(msk,offset_y,offset_x,crop_height,crop_width)

  msk = tf.equal(msk,255)
  ests= tf.split(est,4,axis=-1)

  recover_x=(tf.to_float(input_width)-1) *(tf.to_float(ests[0])/(100.0*255.0)+(2.0*tf.to_float(ests[2])/255.0)-1)
  recover_y=(tf.to_float(input_height)-1)*(tf.to_float(ests[3])/(100.0*255.0)+(2.0*tf.to_float(ests[1])/255.0)-1)

  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,recover_x<min_flow_x,recover_x>max_flow_x,recover_y<min_flow_y,recover_y>max_flow_y],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  est = tf.concat([recover_x,recover_y],axis=-1)
  return tf.to_float(im1),tf.to_float(im2),est,msk

def parser_flow_sintel_png_center(im1_str,im2_str,est_str,msk_str):
  
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  tf.image.decode_png(est_str,channels=4,dtype=tf.uint8)
  msk  =  tf.image.decode_image(msk_str)

  offset_x = (input_width-crop_width)//2
  offset_y = (input_height-crop_height)//2

  im1 = tf.image.crop_to_bounding_box(im1,offset_y,offset_x,crop_height,crop_width)
  im2 = tf.image.crop_to_bounding_box(im2,offset_y,offset_x,crop_height,crop_width)
  est = tf.image.crop_to_bounding_box(est,offset_y,offset_x,crop_height,crop_width)
  msk = tf.image.crop_to_bounding_box(msk,offset_y,offset_x,crop_height,crop_width)

  msk = tf.equal(msk,255)
  ests= tf.split(est,4,axis=-1)

  recover_x=(tf.to_float(input_width)-1) *(tf.to_float(ests[0])/(100.0*255.0)+(2.0*tf.to_float(ests[2])/255.0)-1)
  recover_y=(tf.to_float(input_height)-1)*(tf.to_float(ests[3])/(100.0*255.0)+(2.0*tf.to_float(ests[1])/255.0)-1)

  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,recover_x<min_flow_x,recover_x>max_flow_x,recover_y<min_flow_y,recover_y>max_flow_y],axis=-1),axis=-1))
  msk = tf.to_float(msk)
  est = tf.concat([recover_x,recover_y],axis=-1)
  return tf.to_float(im1),tf.to_float(im2),est,msk


def parser_flow_chairs_png(im1_str,im2_str,est_str,msk_str):
  
  im1  =  tf.image.decode_image(im1_str)
  im2  =  tf.image.decode_image(im2_str)
  est  =  tf.image.decode_png(est_str,channels=4,dtype=tf.uint8)
  msk  =  tf.image.decode_image(msk_str)

  ests= tf.split(est,4,axis=-1)
  recover_x=(tf.to_float(input_width)-1) *(tf.to_float(ests[0])/(100.0*255.0)+(2.0*tf.to_float(ests[2])/255.0)-1)
  recover_y=(tf.to_float(input_height)-1)*(tf.to_float(ests[3])/(100.0*255.0)+(2.0*tf.to_float(ests[1])/255.0)-1)
  est = tf.concat([recover_x,recover_y],axis=-1)

  msk = tf.equal(msk,255)
  msk = tf.logical_not(tf.reduce_any(tf.concat([msk,recover_x<min_flow_x,recover_x>max_flow_x,recover_y<min_flow_y,recover_y>max_flow_y],axis=-1),axis=-1))
  msk = tf.to_float(msk)

  return tf.to_float(im1),tf.to_float(im2),est,msk

def dataset_loader(batch_size,num_epochs,shuffle_size,do_shuffle=False,dataset_list=None,data_parser=None,num_calls=16):

  dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
  dataset = dataset.apply(tf.contrib.data.parallel_interleave(flat_map_impl,cycle_length=batch_size,block_length=16))
  if do_shuffle:
    dataset = dataset.shuffle(shuffle_size)
  dataset = dataset.prefetch(shuffle_size)
  dataset = dataset.repeat()

  dataset = dataset.apply(tf.contrib.data.map_and_batch(data_parser,num_parallel_calls=num_calls,batch_size=batch_size))
  dataset = dataset.prefetch(1)
  return dataset.make_one_shot_iterator().get_next()


def median_downsampling(inputs,channels=2,height=None,width=None,level=5):
  inputs=tf.transpose(inputs,[0,3,1,2])
  for i in range(level):
    inputs=tf.nn.avg_pool(inputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')
  
  scale=(2**level)
  if channels==1:
    valid =tf.is_finite(inputs)
    inputs=tf.where(valid,inputs,tf.zeros(inputs.get_shape()))
    valid =tf.to_float(valid)
    return tf.reshape(inputs,[-1,height//scale,width//scale]),tf.reshape(valid,[-1,height//scale,width//scale])
  elif channels==2:
    valid =tf.is_finite(inputs)
    inputs=tf.where(valid,inputs,tf.zeros(inputs.get_shape()))
    valid =tf.to_float(tf.reduce_all(valid,axis=1))
    return tf.reshape(inputs,[-1,channels,height//scale,width//scale]),tf.reshape(valid,[-1,height//scale,width//scale])
  else:
    assert 0


def average_downsampling_pyramid(inputs,height=None,width=None,level=5):
  assert len(inputs.get_shape())==3
  outputs=[]
  inputs =tf.expand_dims(inputs,axis=1)
  for i in range(level):
    inputs=tf.nn.avg_pool(inputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')
    outputs.append(tf.squeeze(inputs,axis=1))
  outputs.reverse()
  return outputs

def median_downsampling_pyramid(inputs,channels=2,height=None,width=None,level=5):
  
  inputs     =tf.transpose(inputs,[0,3,1,2])
  downsamples=[inputs]
  for i in range(level):
    inputs=tf.nn.avg_pool(inputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')
    downsamples.append(inputs)
  
  valids = []
  outputs= []
  
  if  channels==1:
    for i in range(len(downsamples)):
      
      valid      =tf.is_finite(downsamples[i])
      downsampled=tf.where(valid,downsamples[i],tf.zeros(valid.get_shape()))
      valid      =tf.to_float(valid)

      scale      =(2**i)
      outputs.append(tf.reshape(downsampled,[-1,height//scale,width//scale]))
      valids.append(tf.reshape(valid,[-1,height//scale,width//scale]))

  elif channels==2:

    for i in range(len(downsamples)):

      valid      =tf.is_finite(downsamples[i])
      downsampled=tf.where(valid,downsamples[i],tf.zeros(valid.get_shape()))
      valid      =tf.to_float(tf.reduce_all(valid,axis=1))

      scale =(2**i)
      outputs.append(tf.reshape(downsampled,[-1,channels,height//scale,width//scale]))
      valids.append(tf.reshape(valid,[-1,height//scale,width//scale]))
  else:
    assert 0

  valids.reverse()
  outputs.reverse()
  return outputs,valids






  


