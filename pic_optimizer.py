import numpy as np
import cv2
import patch_finder
import controller


#give supports in descending order eg [16,8]
class picturer:
    def __init__(self,im,flow,supports,sample,lamb_e,lamb_m,over_divisor,converge):
        #list of drop-off filters of size supports x supports
        self.weights=[]
        
        #list of patch finder object 
        #with identical control pixel frames  and samples
        #for patches of each support size
        self.finders=[]
        
        #multipliers for the control term in the E and M optimization stages
        self.lamb_e=lamb_e
        self.lamb_m=lamb_m
        
        #minimum changed nearest neighbors to constitute convergence
        self.converge=converge
        
        #number of neighborhoods overlapping each pixel in output image
        self.od=over_divisor

        #distance between neighborhoods in output image (step sizes list)
        self.steps=[]

        #shapes of the matrices that just contain the NN indicies
        #i.e. one entry per neighborhood under consideration
        self.ind_shapes=[]

        #to prevent extra work in M step
        self.bitmaps=[]

        #divisors to account for sum of weights being not equal to 1
        self.wmats=[]

        #matrices containing NN assignments
        self.inds=[]

        #for wraparound
        self.indices=[]
        
        #control flow
        self.flow=np.zeros(flow.shape,dtype=np.int32)
        self.flow[:,:]=flow

        #conrolled pixels to match
        self.ctrl=controller.control(im,flow)

        #logical indices of control pixels
        self.mask=np.sum(self.ctrl,2)>0

        i = 0
        for support in supports:
            step=support/self.od
            self.steps.append(step)
            y_tot=im.shape[0]-support
            x_tot=im.shape[1]-support

            self.lev=i
            i+=1
            ind_shape=(np.ceil(np.float32(y_tot)/step)+1,np.ceil(np.float32(x_tot)/step)+1)
            
            max_ind=(sample.shape[0]-support)*(sample.shape[1]-support)
            self.inds.append(np.uint32(np.random.rand(ind_shape[0],ind_shape[1])*max_ind))
            self.ind_shapes.append(ind_shape)
            self.bitmaps.append(np.ones(ind_shape,dtype=np.uint8))
            self.indices.append(np.arange(support*support))
            
            self.finders.append(patch_finder.hooder(sample,support,lamb_m,self.ctrl))
            goose=cv2.getGaussianKernel(support,support/3)
            kern=np.dot(goose,goose.T)
            weigh=np.zeros((support,support,3),dtype=np.float)
            for color in range(3):
                weigh[:,:,color]=kern
            self.weights.append(weigh)
            weightmat=np.zeros(im.shape)
            for meta_ind in range(int(np.product(ind_shape))):
                x_off=step*(meta_ind%ind_shape[1])
                y_off=step*(meta_ind/ind_shape[1])
                weigthmat=self.modify_mat(weightmat,x_off,y_off,support,weigh)
                #weightmat[x_off:x_off+support,y_off:y_off+support,:]+=weigh
            #weightmat[weightmat==0]=1
            self.wmats.append(weightmat)
        
        self.supports=supports

        #output image
        self.im=np.zeros(im.shape,dtype=im.dtype)
        self.im[:,:,:]=im

        #index of current neighborhood size in self.supports
        self.lev=0

        #exemplar texture image
        self.sample=np.zeros(sample.shape)
        self.sample[:,:,:]=sample

    def modify_mat(self,mat,off0,off1,sz,new_patch):
        y=off0+sz
        x=off1+sz
        inds=self.indices[self.lev]
        y_max=mat.shape[0]
        x_max=mat.shape[1]
        if(y<mat.shape[0]):
            if(x<mat.shape[1]):
                mat[off0:y,off1:x,:]+=new_patch
                return mat
        nps0=new_patch.shape[0]
        mat[np.int32(inds%nps0+off0)%y_max,np.int32(inds/nps0+off1)%x_max,:]+=new_patch[np.int32(inds%nps0)%y_max,np.int32(inds/nps0)%x_max,:]
        return mat
                
        
    def get_block(self,ind):
        xoff=ind%(self.sample.shape[1]-self.supports[self.lev])
        yoff=ind/(self.sample.shape[1]-self.supports[self.lev])

        #dont fix me with wraparound
        #block=self.get_block_off(yoff,xoff)
        #return block
        return self.sample[yoff:yoff+self.supports[self.lev],xoff:xoff+self.supports[self.lev],:]

    def get_block_off(self,yoff,xoff,mat):
        supp=self.supports[self.lev]
        y=yoff+supp
        x=xoff+supp
        inds=self.indices[self.lev]
        y_max=mat.shape[0]
        x_max=mat.shape[1]
        if(y<y_max):
            if(x<x_max):
                return mat[yoff:y,xoff:x,:]
        new_patch=np.zeros((supp,supp,3),dtype=mat.dtype)
        new_patch[np.int32(inds/supp)%y_max,np.int32(inds%supp)%x_max,:]=mat[np.int32(inds/supp+yoff)%y_max,np.int32(inds%supp+xoff)%x_max,:]
        return new_patch
    
    def expectation(self):
        ans=np.zeros(self.im.shape)
        support=self.supports[self.lev]
        wtplier=np.zeros(self.im.shape)
        wtplier[:,:,:]=self.wmats[self.lev]
        lamda=self.lamb_e*np.max(self.weights[self.lev])
        for meta_ind in range(int(np.product(self.ind_shapes[self.lev]))):
            meta_x=meta_ind%self.ind_shapes[self.lev][1]
            meta_y=meta_ind/self.ind_shapes[self.lev][1]
            block=self.weights[self.lev]*self.get_block(self.inds[self.lev][meta_y,meta_x])
            x_off=self.steps[self.lev]*(meta_x)
            y_off=self.steps[self.lev]*(meta_y)
            #ans[x_off:x_off+support,y_off:y_off+support,:]+=block
            #block2=self.ctrl[x_off:x_off+support,y_off:y_off+support,:]*lamda
            block2=self.get_block_off(x_off,y_off,self.ctrl)*lamda

            ans=self.modify_mat(ans,x_off,y_off,support,block+block2)
            wtplier=self.modify_mat(wtplier,x_off,y_off,support,lamda*np.int32(block2>0))
            #ans[x_off:x_off+support,y_off:y_off+support,:]+=block+block2
            #wtplier[x_off:x_off+support,y_off:y_off+support,:]+=lamda*np.int32(block2>0)
        wtplier[wtplier==0]=1
        ans=ans/wtplier
        cv2.destroyAllWindows()
        cv2.imshow('ans',np.uint8(ans))
        self.im[ans>0]=ans[ans>0]


    def maximization(self,print_num):
        acc=0
        support=self.supports[self.lev]
        new_map=np.zeros(self.ind_shapes[self.lev],dtype=np.uint8)

        #print the number of nodes to check in round
        print np.product(self.ind_shapes[self.lev])
        for meta_ind in range(int(np.product(self.ind_shapes[self.lev]))):
            meta_x=meta_ind%self.ind_shapes[self.lev][1]
            meta_y=meta_ind/self.ind_shapes[self.lev][1]
            if self.bitmaps[self.lev][meta_y,meta_x]:
                x_off=(self.steps[self.lev])*meta_x
                y_off=(self.steps[self.lev])*meta_y

                #fix me with wraparound
                #patch=self.im[x_off:x_off+support,y_off:y_off+support,:]
                #mask_patch=self.ctrl[x_off:x_off+support,y_off:y_off+support,:]>0
                patch=self.get_block_off(x_off,y_off,self.im)
                mask_patch=self.get_block_off(x_off,y_off,self.ctrl)
                
                mask_patch=mask_patch[:,:,0]
                ind=self.finders[self.lev].query(patch,mask_patch)
                if self.inds[self.lev][meta_y,meta_x]!=ind:
                    new_map[meta_y,meta_x]=1
                    self.inds[self.lev][meta_y,meta_x]=ind
                    acc+=1
                    if acc%print_num==1:
                        print meta_ind
        #this gets reset again after convergence in optimize
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

    def update_controls(self):
        print np.max(self.ctrl)
        cv2.imshow('ctrl',np.uint8(self.ctrl))
        self.ctrl=controller.control(self.im,self.flow)
        self.finders=[]
        for support in self.supports:
            h=patch_finder.hooder(self.sample,support,self.lamb_m,self.ctrl)
            self.finders.append(h)
