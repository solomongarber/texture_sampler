from sklearn import neighbors
import numpy as np
import cv2
class Tree:
    def __init__(self,sample,support,out_shape,over_divisor):
        self.sample=np.zeros(sample.shape)
        self.sample[:,:,:]=sample
        self.x=sample.shape[1]
        self.y=sample.shape[0]

        goose=cv2.getGaussianKernel(support,support/3)
        kern=np.dot(goose,goose.T)
        self.weights=np.zeros((support,support,3),dtype=np.float)
        for color in range(3):
            self.weights[:,:,color]=kern
        
        self.od=over_divisor
        self.step=support/self.od
        
        self.guess=np.zeros(out_shape,dtype=np.float)
        for pix in range(out_shape[0]*out_shape[1]):
            xo=int(self.x*np.random.rand())
            yo=int(self.y*np.random.rand())
            self.guess[pix/out_shape[1],pix%out_shape[1],:]=sample[yo,xo,:]
        max_node=(out_shape[0]/support)*(out_shape[1]/support)
        max_ind=(self.x-support)*(self.y-support)
        for node in range(max_node):
            x_off=(node%(out_shape[1]/support))*support
            y_off=(node/(out_shape[1]/support))*support
            ind=np.random.rand()*max_ind
            xoff=ind%(self.x-support)
            yoff=ind/(self.x-support)
            print sample[yoff:yoff+support,xoff:xoff+support,:].shape
            self.guess[y_off:y_off+support,x_off:x_off+support]=sample[yoff:yoff+support,xoff:xoff+support,:]
        self.support=support
        
        
        self.x_tot=out_shape[1]-support
        self.y_tot=out_shape[0]-support
        self.ind_shape=(self.y_tot/self.step,self.x_tot/self.step)
        self.bitmap=np.ones(self.ind_shape,dtype=np.uint8)

        self.data=np.zeros(((self.x-support)*(self.y-support),support*support*3))
        #max_ind=(self.x-self.support)*(self.y-self.support)
        self.inds=np.uint32(np.random.rand(self.ind_shape[0],self.ind_shape[1])*max_ind)
        self.wmat=np.zeros(out_shape)
        for meta_ind in range(np.product(self.ind_shape)):
            x_off=self.step*(meta_ind%self.ind_shape[1])
            y_off=self.step*(meta_ind/self.ind_shape[1])
            self.wmat[x_off:x_off+support,y_off:y_off+support,:]+=self.weights
        self.wmat[self.wmat==0]=1
        for ind in range((self.x-support)*(self.y-support)):
            xoff=ind%(self.x-support)
            yoff=ind/(self.x-support)
            self.data[ind,:]=sample[yoff:yoff+support,xoff:xoff+support,:].reshape(1,-1)
        ls=np.max((1,self.data.shape[0]/100))
        ls=np.min((ls,10000))
        self.tree=neighbors.KDTree(self.data,leaf_size=ls)
        self.out_shape=out_shape

    def change_support(self,support):
        goose=cv2.getGaussianKernel(support,support/3)
        kern=np.dot(goose,goose.T)
        self.weights=np.zeros((support,support,3),dtype=np.float)
        for color in range(3):
            self.weights[:,:,color]=kern
            
        self.step=support/self.od
        self.support=support
        
        self.data=np.zeros(((self.x-support)*(self.y-support),support*support*3))
        self.x_tot=self.out_shape[1]-support
        self.y_tot=self.out_shape[0]-support
        self.ind_shape=(self.y_tot/self.step,self.x_tot/self.step)
        self.bitmap=np.ones(self.ind_shape,dtype=np.uint8)
        self.inds=np.zeros(self.ind_shape,dtype=np.uint32)
        max_ind=(self.x-support)*(self.y-support)
        self.wmat=np.zeros(self.out_shape)
        for meta_ind in range(np.product(self.ind_shape)):
            x_off=self.step*(meta_ind%self.ind_shape[1])
            y_off=self.step*(meta_ind/self.ind_shape[1])
            self.wmat[x_off:x_off+support,y_off:y_off+support,:]+=self.weights
        self.wmat[self.wmat==0]=1
        for ind in range((self.x-support)*(self.y-support)):
            xoff=ind%(self.x-support)
            yoff=ind/(self.x-support)
            self.data[ind,:]=self.sample[yoff:yoff+support,xoff:xoff+support,:].reshape(1,-1)
        ls=np.max((1,self.data.shape[0]/100))
        ls=np.min((ls,1000))
        self.tree=neighbors.KDTree(self.data,leaf_size=ls)

    def upsample(self,sample):
        self.out_shape=(self.out_shape[0]*2,self.out_shape[1]*2,3)
        self.sample=sample
        self.x=sample.shape[1]
        self.y=sample.shape[0]
        self.data=np.zeros(((self.x-self.support)*(self.y-self.support),self.support*self.support*3))
        self.x_tot=self.out_shape[1]-self.support
        self.y_tot=self.out_shape[0]-self.support
        self.ind_shape=(self.y_tot/self.step,self.x_tot/self.step)
        self.bitmap=np.ones(self.ind_shape,dtype=np.uint8)
        self.inds=np.zeros(self.ind_shape,dtype=np.uint32)
        #seg=self.guess
        #jbg=np.uint8(seg)
        #jeg=cv2.pyrUp(jbg)
        #rse=np.float32(jeg)
        self.guess=np.float32(cv2.pyrUp(np.uint8(self.guess)))
        self.wmat=np.zeros(self.out_shape)
        for meta_ind in range(np.product(self.ind_shape)):
            x_off=self.step*(meta_ind%self.ind_shape[1])
            y_off=self.step*(meta_ind/self.ind_shape[1])
            self.wmat[x_off:x_off+self.support,y_off:y_off+self.support,:]+=self.weights
        self.wmat[self.wmat==0]=1
        for ind in range((self.x-self.support)*(self.y-self.support)):
            xoff=ind%(self.x-self.support)
            yoff=ind/(self.x-self.support)
            self.data[ind,:]=sample[yoff:yoff+self.support,xoff:xoff+self.support,:].reshape(1,-1)
        ls=np.max((1,self.data.shape[0]/100))
        ls=np.min((ls,10000))
        self.tree=neighbors.KDTree(self.data,leaf_size=ls)
        
    def getNeighbor(self,patch):
        ind=self.tree.query(patch.resehape(1,-1))[1][0][0]
        xoff=ind%self.x_tot
        yoff=ind/self.x_tot
        return self.sample[yoff:yoff+support,xoff:xoff+support,:]

    def expectation(self):
        #ans=np.zeros(self.out_shape)
        ans=self.rand_prop()
        ans[:(self.ind_shape[0]-1)*self.step+self.support,:(self.ind_shape[1]-1)*self.step+self.support,:]=0
        for meta_ind in range(np.product(self.ind_shape)):
            meta_x=meta_ind%self.ind_shape[1]
            meta_y=meta_ind/self.ind_shape[1]
            block=self.get_block(self.inds[meta_y,meta_x])
            x_off=self.step*(meta_x)
            y_off=self.step*(meta_y)
            #print (self.x-self.support)*(self.y-self.support)
            #print (meta_x,meta_y,self.y-self.support)
            #print self.inds[meta_y,meta_x]
            ans[x_off:x_off+self.support,y_off:y_off+self.support,:]+=self.weights*block
        ans=ans/self.wmat
        cv2.destroyAllWindows()
        cv2.imshow('ans',np.uint8(ans))
        self.guess[:,:,:]=ans

    def get_block(self,ind):
        xoff=ind%(self.x-self.support)
        yoff=ind/(self.x-self.support)
        return self.sample[yoff:yoff+self.support,xoff:xoff+self.support,:]

    def maximization(self,print_num):
        acc=0
        new_map=np.zeros(self.ind_shape,dtype=np.uint8)
        print np.product(self.ind_shape)
        for meta_ind in range(np.product(self.ind_shape)):
            meta_x=meta_ind%self.ind_shape[1]
            meta_y=meta_ind/self.ind_shape[1]
            if self.bitmap[meta_y,meta_x]:
                x_off=self.step*(meta_x)
                y_off=self.step*(meta_y)
                patch=self.guess[x_off:x_off+self.support,y_off:y_off+self.support,:]
                ind=self.tree.query(patch.reshape(1,-1))[1][0][0]
                if self.inds[meta_y,meta_x]!=ind:
                    new_map[meta_y,meta_x]=1
                    self.inds[meta_y,meta_x]=ind
                    acc+=1
                    if acc%print_num==1:
                        print meta_ind
        self.bitmap[:,:]=cv2.dilate(new_map,np.ones((2*self.od-1,2*self.od-1)))
        return acc

    def rand_prop(self):
        ans=np.zeros(self.out_shape)
        for pix in range(self.out_shape[0]*self.out_shape[1]/16):
            xo=int((self.x-4)*np.random.rand())
            yo=int((self.y-4)*np.random.rand())
            up=pix/(self.out_shape[1]/4)*4
            left=pix%(self.out_shape[1]/4)*4
            block=ans[up:up+4,left:left+4,:]
            ans[up:up+4,left:left+4,:]=self.sample[yo:yo+4,xo:xo+4,:][:block.shape[0],:block.shape[1],:]
        return ans
            
def EM(self,thresh,pnum):
    acc=thresh+10
    while acc>thresh:
        acc=self.maximization(pnum)
        self.expectation()
        


