import numpy as np
import cv2
import pic_optimizer
import controller
imput = cv2.imread('newmossge.png')
sample=cv2.imread('msamp.png')
im=np.zeros((64,64,3),dtype=np.uint8)
im=imput[100:164,:64]


#bed of nails flow field
nail_width=2
over_divisor=2
lamb_e=.5
lamb_m=2
flow=np.zeros((im.shape[0],im.shape[1],2),dtype=np.int32)
#flow[::nail_width,::nail_width]=1
r=np.arange(im.shape[0]*im.shape[1])
#flowy=(((r/im.shape[1]).reshape(im.shape[0],im.shape[1]))/im.shape[0])*(2*np.pi)
#flowy=(np.sin(flowy)*2)%im.shape[0]
#flowx=((np.float32(r%im.shape[1]).reshape(im.shape[0],im.shape[1]))/im.shape[1])*(2*np.pi)
#flowx=(np.cos(flowx)*2)%im.shape[1]
#flow=np.dstack((np.int32(flowy),np.int32(flowx)))
flow[:int(flow.shape[1]/2),:,0]=-1;
flow[int(flow.shape[1]/2):,:,:]=1;
flow[:,:int(flow.shape[1]/2),:]=-1;
flow[:,int(flow.shape[1]/2):,:]=1;
flow[:,:,0]=flow[:,:,0]%flow.shape[0]
flow[:,:,1]=flow[:,:,1]%flow.shape[1]
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
outname='./mrfEM/sm-sliding-moss-step-divisor-'+str(over_divisor)+'-le-'+str(lamb_e)+'-lm-'+str(lamb_m)+'.mp4'
out = cv2.VideoWriter(outname ,fourcc, 30, (im.shape[1],im.shape[0]),True)

po=pic_optimizer.picturer(im,flow,[16 ,8],sample,lamb_e,lamb_m,over_divisor,20)
out.write(im)
for i in range(60):
    po.optimize(20)
    im[:,:,:]=po.im
    out.write(im)
    po.update_controls()
    print "i "+str(i)
