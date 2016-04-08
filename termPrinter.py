import numpy as np
def p_term_bw(im):
    fmt= '\033[48;5;%dm  \033[0m'
    im2=np.float32(im-np.min(im))
    im2/=np.max(im2)
    im2=np.int32(im2*23.999)+232
    s=""
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            s+=fmt%im2[y,x]
        s+='\n'
    print s
