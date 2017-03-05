import cv2
import numpy as np


def rep_cols(im,cols,split):
    ans=np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
    pos=im*np.int32(im>split)-split
    neg=split-im*np.int32(im<split)
    pos_cap=np.max(pos)
    neg_cap=np.max(neg)
    bias=255/(pos_cap+neg_cap)
    pos=bias*(np.dstack((pos,pos,pos)))
    pos[:,:]=pos[:,:]*cols[0]
    neg=bias*(np.dstack((neg,neg,neg)))
    neg[:,:]=neg[:,:]*cols[1]
    ans[:,:,:]=np.uint8(pos+neg)
    cv2.imshow('ans',ans)
    return ans

def rep_bw(im):
    ans=np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
    temp=np.zeros(im.shape)
    temp[:,:]=im+np.min(im)
    temp[:,:]=temp*np.max(temp)
    ans[:,:]=np.uint8(temp*255)
    cv2.imshow('ans',ans)
    return ans

def rep_bicols(im,bicols,split):
    ans=np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
    pmask=np.int32(im>split)
    nmask=np.int32(im<split)
    #pos=im*np.int32(im>split)-split
    pos=im-split
    pos+=np.min(pos)
    pos/=np.max(pos)
    #pos*=255
    #neg=split-im*np.int32(im<split)
    neg=split-im
    neg+=np.min(neg)
    neg/=np.max(neg)
    #neg*=255
    #pos_cap=np.max(pos)
    #neg_cap=np.max(neg)
    #bias=255/(pos_cap+neg_cap)
    #bias=255/np.max((neg_cap,pos_cap))
    #pos=bias*(np.dstack((pos,pos,pos)))
    pos=np.dstack((pos,pos,pos))
    pos[:,:]=pos[:,:]*bicols[0,:]+(1-pos[:,:])*bicols[1,:]
    #neg=bias*(np.dstack((neg,neg,neg)))
    neg=np.dstack((neg,neg,neg))
    neg[:,:]=neg[:,:]*bicols[3,:]+(1-neg[:,:])*bicols[2,:]
    #middle=neg_cap*bias
    #ans[:,:,:]=np.uint8(pos*np.dstack((pmask,pmask,pmask))+neg*np.dstack((nmask,nmask,nmask)))
    ans[:,:,:]=np.uint8(128*pos+128*neg)
    cv2.imshow('ans',ans)
    return ans
