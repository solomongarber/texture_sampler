import numpy as np
import cv2
import patch_finder
import controller

class picturer:
    def __init__(self,im,flow,supports,sample,lamb,over_divisor,converge):
        self.weights=[]
        self.finders=[]
        self.lamb=lamb
        self.converge=converge
        self.od=over_divisor
        self.steps=[]
        self.ind_shapes=[]
        self.bitmaps=[]
        self.wmats=[]
        self.inds=[]
        self.flow=np.zeros(flow.shape,dtype=np.int32)
        self.flow[:,:]=flow
        self.ctrl=controller.control(im,flow)
        self.mask=np.sum(self.ctrl,2)>0
        for support in supports:
            step=support/self.od
            self.steps.append(step)
            y_tot=im.shape[0]-support
            x_tot=im.shape[1]-support
            ind_shape=(np.ceil(np.float32(y_tot)/step),np.ceil(np.float32(x_tot)/step))
            max_ind=(sample.shape[0]-support)*(sample.shape[1]-support)
            self.inds.append(np.uint32(np.random.rand(ind_shape[0],ind_shape[1])*max_ind))
            self.ind_shapes.append(ind_shape)
            self.bitmaps.append(np.ones(ind_shape,dtype=np.uint8))
            self.finders.append(patch_finder.hooder(sample,support,lamb,self.ctrl))
            goose=cv2.getGaussianKernel(support,support/3)
            kern=np.dot(goose,goose.T)
            weigh=np.zeros((support,support,3),dtype=np.float)
            for color in range(3):
                weigh[:,:,color]=kern
            self.weights.append(weigh)
            weightmat=np.zeros(im.shape)
            for meta_ind in range(np.product(ind_shape)):
                x_off=step*(meta_ind%ind_shape[1])
                y_off=step*(meta_ind/ind_shape[1])
                weightmat[x_off:x_off+support,y_off:y_off+support,:]+=weigh
            #weightmat[weightmat==0]=1
            self.wmats.append(weightmat)
        self.supports=supports
        self.im=np.zeros(im.shape,dtype=im.dtype)
        self.im[:,:,:]=im
        self.lev=0
        self.sample=np.zeros(sample.shape)
        self.sample[:,:,:]=sample

    def get_block(self,ind):
        xoff=ind%(self.sample.shape[1]-self.supports[self.lev])
        yoff=ind/(self.sample.shape[1]-self.supports[self.lev])
        return self.sample[yoff:yoff+self.supports[self.lev],xoff:xoff+self.supports[self.lev],:]
    
    def expectation(self):
        ans=np.zeros(self.im.shape)
        support=self.supports[self.lev]
        weightiplier=np.zeros(self.im.shape)
        weightiplier[:,:,:]=self.wmats[self.lev]
        lamda=self.lamb*np.max(self.weights[self.lev])
        for meta_ind in range(np.product(self.ind_shapes[self.lev])):
            meta_x=meta_ind%self.ind_shapes[self.lev][1]
            meta_y=meta_ind/self.ind_shapes[self.lev][1]
            block=self.weights[self.lev]*self.get_block(self.inds[self.lev][meta_y,meta_x])
            x_off=self.steps[self.lev]*(meta_x)
            y_off=self.steps[self.lev]*(meta_y)
            #ans[x_off:x_off+support,y_off:y_off+support,:]+=block
            block2=self.ctrl[x_off:x_off+support,y_off:y_off+support,:]*lamda
            ans[x_off:x_off+support,y_off:y_off+support,:]+=block+block2
            weightiplier[x_off:x_off+support,y_off:y_off+support,:]+=lamda*np.int32(block2>0)
        weightiplier[weightiplier==0]=1
        ans=ans/weightiplier
        cv2.destroyAllWindows()
        cv2.imshow('ans',np.uint8(ans))
        self.im[ans>0]=ans[ans>0]


    def maximization(self,print_num):
        acc=0
        support=self.supports[self.lev]
        new_map=np.zeros(self.ind_shapes[self.lev],dtype=np.uint8)
        print np.product(self.ind_shapes[self.lev])
        for meta_ind in range(np.product(self.ind_shapes[self.lev])):
            meta_x=meta_ind%self.ind_shapes[self.lev][1]
            meta_y=meta_ind/self.ind_shapes[self.lev][1]
            if self.bitmaps[self.lev][meta_y,meta_x]:
                x_off=(self.steps[self.lev])*meta_x
                y_off=(self.steps[self.lev])*meta_y
                patch=self.im[x_off:x_off+support,y_off:y_off+support,:]
                patch2=self.ctrl[x_off:x_off+support,y_off:y_off+support,:]>0
                patch2=patch2[:,:,0]
                ind=self.finders[self.lev].query(patch,patch2)
                if self.inds[self.lev][meta_y,meta_x]!=ind:
                    new_map[meta_y,meta_x]=1
                    self.inds[self.lev][meta_y,meta_x]=ind
                    acc+=1
                    if acc%print_num==1:
                        print meta_ind
        self.bitmaps[self.lev][:,:]=cv2.dilate(new_map,np.ones((2*self.od-1,2*self.od-1)))
        return acc

    def optimize(self,printnum):
        for level in range(len(self.supports)):
            money=self.converge+1
            self.lev=level
            while money>self.converge:
                money=self.maximization(printnum)
                self.expectation()
            self.bitmaps[self.lev][:,:]=1
