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

def interp(pic,flow):
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]

    u=np.int32(np.floor(ud))
    d=np.int32(np.ceil(ud))%pic.shape[0]
    udiffs=ud-u
    udiffs=np.dstack((udiffs,udiffs,udiffs))
    l=np.int32(np.floor(lr))
    r=np.int32(np.ceil(lr))%pic.shape[1]
    ldiffs=lr-l
    ldiffs=np.dstack((ldiffs,ldiffs,ldiffs))

    ul=pic[u,l,:]
    ur=pic[u,r,:]
    dl=pic[d,l,:]
    dr=pic[d,r,:]



    udl=ul*(1-udiffs)+dl*udiffs
    udr=ur*(1-udiffs)+dr*udiffs
    ans=np.zeros(pic.shape)
    ans[ys,xs,:]=udl*(1-ldiffs)+udr*ldiffs
    return ans

    
