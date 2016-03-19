import numpy as np
import cv2
class hooder:
    def __init__(self,sample,support,lamb,ctrl):
        self.sample=sample
        self.winx=sample.shape[1]-support
        self.winy=sample.shape[0]-support
        self.win_shape=(self.winy,self.winx,3)
        self.support=support
        self.lamb=lamb
        self.ctrl=ctrl
        

    def set_ctrl(self,ctrl):
        self.ctrl=ctrl
    
    def query(self, patch, mask):
        lost_energy=(np.product(mask.shape)*self.lamb)/np.sum(mask)
        window=np.zeros(self.win_shape)
        energies=np.zeros((self.winy,self.winx,3))
        for ind in range(self.support*self.support):
            x_off=ind%self.support
            x_end=x_off+self.winx
            y_off=ind/self.support
            y_end=y_off+self.winy
            window[:,:,:]=self.sample[y_off:y_end,x_off:x_end,:]
            energies+=np.square(window[:,:]-patch[y_off,x_off,:])
            if mask[y_off,x_off]:
                energies+=lost_energy*(np.square(window[:,:]-self.ctrl[y_off,x_off,:]))
        return np.argmin(np.sum(energies,2))
        
