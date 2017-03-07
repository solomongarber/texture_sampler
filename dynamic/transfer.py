import numpy as np
import cv2
import pic_optimizer


#sample=cv2.imread('../msamp.png')
#synth=cv2.imread('../img/newmossge1.png')

#sample=cv2.imread('../msamp.png')
sample=cv2.imread('../img/bigmosseed.png')
synth=cv2.imread('../img/faceb.jpg')
synth=synth[77:577,250:750,:]
x=synth.shape[1]
y=synth.shape[0]

im=synth
sz=im.shape
numpix=y*x
name='face'
flow=np.zeros((sz[0],sz[1],2),dtype=np.int32)
#flow=np.ones((sz[0],sz[1],2),dtype=np.int32)
#flow=np.int32((flow+np.random.randn(sz[0],sz[1],2))%x)
#flow[:,:,0]=np.arange(numpix).reshape(y,x)/x-x/2
#flow[:,:,1]=np.arange(numpix).reshape(y,x)%x-y/2
#mag=np.sqrt(np.sum(np.square(flow),2))
#mag[mag==0]=1
#magz=np.dstack((mag,mag))
#flow=np.int32(((flow/magz)+np.random.randn(sz[0],sz[1],2))%x)
#flow=np.int32((flow/magz)%x)
lambm=2
lambe=1
supports=[32,16,8]
opt=pic_optimizer.picturer(im,flow,supports,sample,lambe,lambm,2,20)



opt.optimize(20)
cv2.imwrite('facebtrans-gitsq-bigseed.png',np.uint8(opt.im))
