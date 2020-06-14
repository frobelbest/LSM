import python
import cv2
import numpy as np
import flow_vis

image1=cv2.imread('./00000.png')
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
image1=cv2.resize(image1,(1024,416))

image2=cv2.imread('./00002.png')
image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
image2=cv2.resize(image2,(1024,416))

flow_estimator=python.Estimator('./depth_32724.ckpt')
flow=flow_estimator(image1,image2)

rgb   =flow_estimator.vis_flow(flow)
cv2.imwrite('./flow_color.png',rgb)

