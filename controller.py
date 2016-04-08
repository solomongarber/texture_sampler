import numpy as np

def control(pic, flow):
    #lr=flow[:,:,0].reshape(-1)
    #ud=flow[:,:,1].reshape(-1)
    #ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    #xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]
    ans=np.zeros(pic.shape)
    ans[ys,xs,:]=pic[ud,lr,:]
    #for i in range(3):
    #     ans[:,:,i]*=np.int32(np.sum(flow,2)>0)
    return ans
