import cv2
import numpy as np
import controller
sz=800
iterate=13
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps=30
vid=0
shp=[sz-32,sz-32]
#out=cv2.VideoWriter('./vids/four0.mp4',fourcc,fps,shp,True)
def get_instructions(levs):
    if levs==1:
        return np.ones(1,dtype=np.uint8)
    tmp=get_instructions(levs-1)
    tmp2=-1*tmp[tmp.shape[0]::-1]
    return np.concatenate((tmp,get_instructions(1),tmp2))

ans=np.zeros((shp[0],shp[1],3),dtype=np.uint8)
#pointer=np.zeros((2,),dtype=np.int32)
pointers=np.zeros((4,2),dtype=np.int32)
pointers[:,:]=[[540,600],[543,597],[540,594],[537,597]]
states=np.arange(4)
#state=0
count=0
colors=np.array([[255,255,255],[0,255,5],[0,0,255],[255,5,0]])
movers=np.zeros((4,2),dtype=np.int8)
movers[[0,1],[1,0]]=1
movers[[2,3],[1,0]]=-1
rules=get_instructions(iterate)
print rules.shape
for rule in rules:
    c=(count*255)/rules.shape[0]
    for i in range(4):
        ans[pointers[i,0],pointers[i,1],:]=(colors[i])
        superstate=(states[i]+rule)%4
        for seg in range(3):
            pointers[i,:]=(pointers[i,:]+movers[states[i],:])%shp
            pointers[i,:]=(pointers[i,:]+movers[superstate,:]*int(seg>0))%shp
            ans[pointers[i,0],pointers[i,1],:]=(colors[i])
        states[i]=superstate
        pointers[i,:]=(pointers[i,:]+movers[states[i],:])%shp

    
    #out.write(ans)
    count+=1
    if count%400==0:
        vid+=1
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #out.release()
        #out=cv2.VideoWriter('./vids/four'+str(vid)+'.mp4',fourcc,fps,shp,True)
    if count%100==0:
        print count

#out.release()
#flow=np.zeros((sz,sz,2),dtype=np.int32)
#a=np.sum(ans,2)
#flow[:,:,0]=np.argmin(np.sum(a,1))
#flow[:,:,1]=np.argmin(np.sum(a,0))
#ans=controller.control(ans,flow)
cv2.imwrite('dragon'+str(iterate)+'.png',ans)
    
    
