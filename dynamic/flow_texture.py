import numpy as np
import cv2
import pic_optimizer


x=64
y=64
#sample=cv2.imread('../msamp.png')
#synth=cv2.imread('../img/newmossge1.png')

sample=cv2.imread('../msamp.png')
synth=cv2.imread('../img/newmossge1.png')


im=synth[200:200+y,80:80+x]
sz=im.shape
numpix=y*x
name='bub'
flow=np.zeros((sz[0],sz[1],2),dtype=np.int32)
#flow=np.ones((sz[0],sz[1],2),dtype=np.int32)
#flow=np.int32((flow+np.random.randn(sz[0],sz[1],2))%x)
flow[:,:,0]=np.arange(numpix).reshape(y,x)/x-x/2
flow[:,:,1]=np.arange(numpix).reshape(y,x)%x-y/2
mag=np.sqrt(np.sum(np.square(flow),2))
mag[mag==0]=1
magz=np.dstack((mag,mag))
#flow=np.int32(((flow/magz)+np.random.randn(sz[0],sz[1],2))%x)
flow=np.int32((flow/magz)%x)
lambm=2
lambe=1
supports=[16,8]
opt=pic_optimizer.picturer(im,flow,supports,sample,lambe,lambm,2,20)

results_dir='../vids/'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_name=results_dir+str(name)+'-sink-supports-'+str(supports)+'-lambs-'+str(lambe)+','+str(lambm)+'circfloww.mp4'
out = cv2.VideoWriter(out_name ,fourcc, 20, (y,x),True)

for i in range(x*2):
    opt.optimize(20)
    out.write(np.uint8(opt.im))
    opt.update_controls()
    
out.close()
