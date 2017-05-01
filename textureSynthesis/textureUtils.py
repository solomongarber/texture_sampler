import numpy as np
import scipy

def subtract_mean(pyr, pind, nband):
    m = np.mean(pyr[nband])
    for j in range(pind[nband][0]):
        for i in range(pind[nband][1]):
            pyr[nband][j, i] -= m
    return pyr, m


def expand(t, f):
    my,mx = t.shape[0:2]
    my = f*my
    mx = f*mx
    Te = np.zeros([my,mx],dtype=complex)
    T = f*f* np.fft.fftshift(np.fft.fft2(t))
    y1 = int(my/2+1 - my/(2*f))
    y2 = int(my/2 + my/(2*f))
    x1 = int(mx/2+1 - mx/(2*f))
    x2 = int(mx/2 + mx/(2*f))
    Te[y1:y2,x1:x2] = T[1:int(my/f),1:int(mx/f)]
    Te[y1-1,x1:x2] = T[0,1:int(mx/f)] / 2
    Te[y2+1,x1:x2] = np.conj(T[0,range(int(mx/f-1),0,-1)]/2)
    Te[y1:y2,x1-1] = T[1:int(my/f),0] / 2
    Te[y1:y2,x2+1] = np.conj(T[range(int(my/f-1),0,-1),0]/2)
    esq = T[0,0] / 4
    Te[y1-1,x1-1] = esq
    Te[y1-1,x2+1] = esq
    Te[y2+1,x1-1] = esq
    Te[y2+1,x2+1] = esq
    Te = np.fft.fftshift(Te)
    te = np.fft.ifft2(Te)
    if np.count_nonzero(np.imag(t)) == 0:
       	te = np.real(te)
    return te


def snr(s,n):

    es = np.sum(np.power(np.abs(s),2))
    en = np.sum(np.power(np.abs(n),2))
    X = 10*np.log10(es/en)
    return X


def modacor22(X, Cy, *args):
    if len(args) == 0:
        p = 1
    else:
        p = args[0]

    # Compute the autocorrelation function of the original image
    Ny, Nx = X.shape[0:2]
    Nc = Cy.shape[0] 	# Normally Nc<<Nx, only the low indices of the autocorrelation
    if 2*Nc-1 > Nx:
        print('Warning: Autocorrelation neighborhood too large for image: reducing')
        Nc = 2*np.floor(Nx/4) - 1
        first = (Cy.shape[0]-Nc) / 2
        Cy = Cy[first:first+Nc, first:first+Nc]

    Xf = np.fft.fft2(X)
    Xf2 = np.abs(Xf)**2
    Cx = np.fft.fftshift(np.real(np.fft.ifft2(Xf2))) / (2-np.isreal(X))
    Cy = Cy* np.prod(X.shape)	# Unnormalize the previously normalized correlation

    cy = int(Ny/2)
    cx = int(Nx/2)
    Lc = int((Nc-1)/2)
    Cy0 = Cy
    Cy = p*Cy + (1-p)*Cx[cy-Lc:cy+Lc+1, cx-Lc:cx+Lc+1]

    # Compare the actual correlation with the desired one
    # imStats(Cx(cy-Lc:cy+Lc,cx-Lc:cx+Lc),Cy)
    snrV = 10*np.log10(np.sum(Cy0**2) / np.sum((Cy0-Cx[cy-Lc:cy+Lc+1, cx-Lc:cx+Lc+1])**2))

    # Take just the part that has influence on the samples of Cy (Cy=conv(Cx,Ch))
    Cx = Cx[cy-2*Lc:cy+2*Lc+1, cx-2*Lc:cx+2*Lc+1]

    # Build the matrix that performs the convolution Cy1=Tcx*Ch1
    Ncx = 4*Lc
    M = int((Nc*Nc+1)/2)
    Tcx = np.zeros([M, M])

    for i in range(Lc, 2*Lc):
        for j in range(Lc, 3*Lc+1):
            nm = (i-Lc)*(2*Lc+1) + j-Lc
            ccx = Cx[i-Lc:i+Lc+1, j-Lc:j+Lc+1]
            ccxi = np.flip(np.flip(ccx[0:2*Lc+1, 0:2*Lc+1], 0), 1)
            ccx = ccx + ccxi
            ccx[Lc, Lc] = ccx[Lc, Lc]/2
            ccx = np.reshape(np.transpose(np.conj(ccx)), [np.prod(ccx.shape), 1], order='F')
            Tcx[nm, :] = np.transpose(ccx[0:M])

    i = 2*Lc
    for j in range(Lc, 2*Lc+1):
        nm = (i-Lc)*(2*Lc+1) + j-Lc
        ccx = Cx[i-Lc:i+Lc+1, j-Lc:j+Lc+1]
        ccxi = np.flip(np.flip(ccx[0:2*Lc+1, 0:2*Lc+1], 0), 1)
        ccx = ccx + ccxi
        ccx[Lc, Lc] = ccx[Lc, Lc] / 2
        ccx = np.reshape(np.transpose(np.conj(ccx)), [np.prod(ccx.shape), 1], order='F')
        Tcx[nm, :] = np.transpose(ccx[0:M])

    # Rearrange Cy indices and solve the equation
    Cy1 = np.reshape(np.transpose(np.conj(Cy)), [np.prod(Cy.shape), 1], order='F')
    Cy1 = Cy1[0:M]

    Ch1 = np.matmul(np.linalg.inv(Tcx), Cy1)

    # Rearrange Ch1
    Ch1 = np.insert(Ch1, Ch1.shape[0], Ch1[range(len(Cy1)-2, -1, -1)], 0) ########
    Ch = np.transpose(np.conj(np.reshape(Ch1, [Nc, Nc], order='F')))

    # Compute H from Ch (H is zero-phase) through the DFT
    aux = np.zeros([Ny, Nx])
    aux[cy-Lc:cy+Lc+1, cx-Lc:cx+Lc+1] = Ch
    Ch = np.fft.fftshift(aux)
    Chf = np.real(np.fft.fft2(Ch))
    Yf = np.multiply(Xf, np.sqrt(np.abs(Chf)))
    Y = np.fft.ifft2(Yf)

    return Y, snrV, Chf


def modkurt(ch,k,*args):
    if len(args) == 0:
        p = 1
    else:
        p = args[0]

    me = np.mean(ch)
    ch = ch - me

    # Compute the moments
    m = np.zeros(12)
    for n in range(1,12):
        m[n] = np.mean(np.power(ch,n+1))

    # The original kurtosis
    k0 = m[3] / m[1] / m[1]
    snrk = snr(k, k-k0)
    if snrk > 60:
        chm = ch + me
        return chm, snrk
    k = k0*(1-p) + k*p

    # Some auxiliar variables
    a = m[3] / m[1]

    # Coeficients of the numerator (A*lam^4+B*lam^3+C*lam^2+D*lam+E)
    A = (m[11]-4*a*m[9]-4*m[2]*m[8]+6*a*a*m[7]+12*a*m[2]*m[6]+6*m[2]*m[2]*m[5]-
	    4*a*a*a*m[5]-12*a*a*m[2]*m[4]+a*a*a*a*m[3]-12*a*m[2]*m[2]*m[3]+
	    4*a*a*a*m[2]*m[2]+6*a*a*m[2]*m[2]*m[1]-3*m[2]*m[2]*m[2]*m[2])
    B = 4*(m[9]-3*a*m[7]-3*m[2]*m[6]+3*a*a*m[5]+6*a*m[2]*m[4]+3*m[2]*m[2]*m[3]-
	    a*a*a*m[3]-3*a*a*m[2]*m[2]-3*m[3]*m[2]*m[2])
    C = 6*(m[7]-2*a*m[5]-2*m[2]*m[4]+a*a*m[3]+2*a*m[2]*m[2]+m[2]*m[2]*m[1])
    D = 4*(m[5]-a*a*m[1]-m[2]*m[2])
    E = m[3]

    # Define the coefficients of the denominator (F*lam^2+G)^2
    F = D/4
    G = m[1]

    # Now I compute its derivative with respect to lambda
    # (only the roots of derivative = 0 )
    d = np.zeros(5)
    d[0] = B*F
    d[1] = 2*C*F - 4*A*G
    d[2] = 4*F*D -3*B*G - D*F
    d[3] = 4*F*E - 2*C*G
    d[4] = -D*G

    mMlambda = np.roots(d)

    tg = np.divide(np.imag(mMlambda), np.real(mMlambda))
    mMlambda = mMlambda[np.where(np.abs(tg)<1e-6)]
    lNeg = mMlambda[np.where(mMlambda<0)]
    if len(lNeg) == 0:
        lNeg = -1/np.finfo(float).eps
    lPos = mMlambda[np.where(mMlambda>=0)]
    if len(lPos) == 0:
        lPos = 1/np.finfo(float).eps
    lmi = lNeg.max()
    lma = lPos.min()

    lam = [lmi,lma]
    mMnewKt = np.divide(np.polyval([A,B,C,D,E],lam), np.power(np.polyval([F,0,G],lam),2))
    kmin = mMnewKt.min()
    kmax = mMnewKt.max()

    # Given a desired kurtosis, solves for lambda

    if k <= kmin:
        lam = lmi
        print('warning: Saturating (down) kurtosis!')
        print(kmin)
    elif k>=kmax:
        lam = lma
        print('warning: Saturating (up) kurtosis!')
        print(kmax)
    else:
        # Coeficients of the algebraic equation
        c0 = E - k*G*G
        c1 = D
        c2 = C - 2*k*F*G
        c3 = B
        c4 = A - k*F*F

        # Solves the equation
        r = np.roots([c4,c3,c2,c1,c0])

        # Chose the real solution with minimum absolute value with the rigth sign
        tg = np.divide(np.imag(r), np.real(r))
        # lambda = real(r(find(abs(tg)<1e-6)));
        lamda = np.real(r[np.where(np.abs(tg)==0)])
        if len(lamda)>0:
            lam = lamda[np.where(np.abs(lamda)==np.abs(lamda).min())]
            lam = lam[0]
        else:
            lam = 0

    # Modify the channel
    chm = ch+lam*(ch*ch*ch-a*ch-m[2])	# adjust the kurtosis
    chm = chm* np.sqrt(m[1]/np.mean(np.power(chm,2)))	# adjust the variance
    chm = chm + me				# adjust the mean

    return chm, snrk


def modskew(ch,sk,*args):
    if len(args) == 0:
        p = 1
    else:
        p = args[0]

    N = np.prod(ch.shape)	# number of samples
    me = np.mean(ch)
    ch = ch - me

    m = np.zeros(6)
    for n in range(1,6):
        m[n] = np.mean(np.power(ch,(n+1)))

    sd = np.sqrt(m[1])	# standard deviation
    s = m[2]/(sd*sd*sd)	# original skewness
    snrk = snr(sk, sk-s)
    sk = s*(1-p) + sk*p

    # Define the coefficients of the numerator (A*lam^3+B*lam^2+C*lam+D)
    A = m[5]-3*sd*s*m[4]+3*sd*sd*(s*s-1)*m[3]+np.power(sd,6)*(2+3*s*s-np.power(s,4))
    B = 3*(m[4]-2*sd*s*m[3]+np.power(sd,5)*s*s*s)
    C = 3*(m[3]-np.power(sd,4)*(1+s*s))
    D = s*sd*sd*sd

    a = np.zeros(7)
    a[6] = A*A
    a[5] = 2*A*B
    a[4] = B*B+2*A*C
    a[3] = 2*(A*D+B*C)
    a[2] = C*C+2*B*D
    a[1] = 2*C*D
    a[0] = D*D

    # Define the coefficients of the denominator (A2+B2*lam^2)
    A2 = sd*sd
    B2 = m[3]-(1+s*s)*np.power(sd,4)

    b = np.zeros(7)
    b[6] = B2*B2*B2
    b[4] = 3*A2*B2*B2
    b[2] = 3*A2*A2*B2
    b[0] = A2*A2*A2

    # Now I compute its derivative with respect to lambda

    d = np.zeros(8)
    d[7] = B*b[6];
    d[6] = 2*C*b[6] - A*b[4]
    d[5] = 3*D*b[6]
    d[4] = C*b[4] - 2*A*b[2]
    d[3] = 2*D*b[4] - B*b[2]
    d[2] = -3*A*b[0]
    d[1] = D*b[2] - 2*B*b[0]
    d[0] = -C*b[0]

    d = d[range(7,-1,-1)]
    mMlambda = np.roots(d)

    tg = np.divide(np.imag(mMlambda),np.real(mMlambda))
    mMlambda = np.real(mMlambda[np.where(np.abs(tg)<1e-6)])
    lNeg = mMlambda[np.where(mMlambda<0)]
    if len(lNeg)==0:
        lNeg = -1/np.finfo(float).eps
    lPos = mMlambda[np.where(mMlambda>=0)]
    if len(lPos)==0:
        lPos = 1/np.finfo(float).eps
    lmi = lNeg.max()
    lma = lPos.min()

    lam = [lmi,lma]
    mMnewSt = np.divide(np.polyval([A, B, C, D], lam),
                        np.power(np.polyval(b[range(6, -1, -1)], lam), 0.5))
    skmin = mMnewSt.min()
    skmax = mMnewSt.max()

    # Given a desired skewness, solves for lambda
    if sk<=skmin:
            lam = lmi
            print('warning: Saturating (down) skewness!')
            print(skmin)
    elif sk>=skmax:
            lam = lma
            print('warning: Saturating (up) skewness!')
            print(skmax)
    else:
        # The equation is sum(c.*lam.^(0:6))=0
        c = a-b*sk*sk
        c = c[range(6,-1,-1)]
        r = np.roots(c)

        # Chose the real solution with minimum absolute value with the rigth sign
        lam = []
        co = 0
        for n in range(6):
            tg = np.imag(r[n])/np.real(r[n])
            if (np.abs(tg)<1e-6) & (np.sign(np.real(r[n])) == np.sign(sk-s)):
                co = co + 1
                lam = np.insert(lam, len(lam), np.real(r[n]))

        if len(lam) == 0:
            print('Warning: Skew adjustment skipped!')
            lam = [0]

        p = [A,B,C,D]

        if len(lam)>1:
            foo = np.sign(np.polyval(p,lam))
            if np.count_nonzero(foo==0) > 0:
                lam = lam[np.where(foo==0)]
            else:
                lam = lam[np.where(foo==np.sign(sk))]		# rejects the symmetric solution
            if len(lam)>0:
                lam = lam[np.where(np.abs(lam)==np.abs(lam).min())]	# the smallest that fix the skew
                lam = lam[0]
            else:
                lam = 0

    # Modify the channel
    chm = ch+lam*(np.power(ch,2)-sd*sd-sd*s*ch)	# adjust the skewness
    chm = chm*np.sqrt(m[1]/np.mean(np.power(chm,2)))	# adjust the variance
    chm = chm + me				# adjust the mean (These don't affect the skewness)

    return chm, snrk


def innerProd(X):
    res = np.matmul(np.transpose(np.conj(X)), X)
    return res


def adjustCorr1s(X,Co,*args):

    if len(args) < 1:
        mode = 2
    else:
        mode = args[0]

    if len(args) < 2:
        p = 1
    else:
        p = args[1]

    C = innerProd(X) / X.shape[0]
    D, E = np.linalg.eig(C)
    junk = np.sort(D)
    Ind = np.argsort(D)
    D = np.diag(np.sqrt(D[Ind[range(len(Ind)-1,-1,-1)]]))
    E = E[:,Ind[range(len(Ind)-1,-1,-1)]]

    Co0 = Co
    Co = (1-p)*C + p*Co

    Do, Eo = np.linalg.eig(Co)
    junk = np.sort(Do)
    Ind = np.argsort(Do)
    Do = np.diag(np.sqrt(Do[Ind[range(len(Ind)-1,-1,-1)]]))
    Eo = Eo[:,Ind[range(len(Ind)-1,-1,-1)]]

    if (mode == 0):
        Orth = scipy.linalg.orth(np.random.rand(C.shape))
    elif (mode == 1): # eye
        Orth = np.eye(C.shape)
    elif (mode == 2): # simple
        Orth = np.matmul(np.transpose(np.conj(E)), Eo)
    else:     # SVD
        tmp1 = np.matmul(np.matmul(np.transpose(np.conj(E)), Eo), np.linalg.inv(Do))
        U,S,V = numpy.linalg.svd(tmp1)
        Orth = np.matmul(U, np.transpose(np.conj(V)))

    M =  np.matmul(np.matmul(E, np.linalg.inv(D)), Orth)
    M = np.matmul(np.matmul(M, Do), np.transpose(np.conj(Eo)))

    newX = np.matmul(X, M)

    snr1 = 10*np.log10(np.sum(np.power(Co0,2))/np.sum(np.power(Co0-C,2)))

    return newX, snr1, M


def adjustCorr2s(X, Cx, Y, Cxy, *args):

    if len(args) < 1:
        mode = 2
    else:
        mode = args[0]

    if len(args) < 2:
        p = 1
    else:
        p = args[1]

    Bx = innerProd(X) / X.shape[0]
    Bxy = np.matmul(np.transpose(np.conj(X)), Y) / X.shape[0]
    By = innerProd(Y) / X.shape[0]
    iBy = np.linalg.inv(By)

    Current = Bx - np.matmul(np.matmul(Bxy, iBy), np.transpose(np.conj(Bxy)))
    Cx0 = Cx
    Cx = (1-p)*Bx + p*Cx
    Cxy0 = Cxy
    Cxy = (1-p)*Bxy + p*Cxy
    Desired = Cx - np.matmul(np.matmul(Cxy, iBy), np.transpose(np.conj(Cxy)))

    D, E = np.linalg.eig(Current)
    if np.count_nonzero(D < 0) > 0:
        ind = np.where(D<0)
        # print('Warning: negative current eigenvalues: %d\n' % D[ind])
    junk = np.sort(D)
    Ind = np.argsort(D)
    D = np.diag(np.sqrt(D[Ind[range(len(Ind)-1,-1,-1)]]))
    E = E[:,Ind[range(len(Ind)-1,-1,-1)]]

    Do, Eo = np.linalg.eig(Desired)
    if np.count_nonzero(Do < 0) > 0:
        ind = np.where(Do<0)
        # print ('Warning: negative desired eigenvalues: %d\n' % Do[ind])
    junk = np.sort(Do)
    Ind = np.argsort(Do)
    Do = np.diag(np.sqrt(Do[Ind[range(len(Ind)-1,-1,-1)]]))
    Eo = Eo[:,Ind[range(len(Ind)-1,-1,-1)]]

    if (mode == 0):
        Orth = scipy.linalg.orth(np.random.rand(D.shape))
    elif (mode == 1): # eye
        Orth = np.eye(D.shape)
    elif (mode == 2): # simple
        A = [np.eye(Cx.shape[0],Cx.shape[1]), np.matmul(-iBy, np.transpose(np.conj(Bxy)))]
        Ao =  [np.eye(Cx.shape[0],Cx.shape[1]), np.matmul(-iBy, np.transpose(np.conj(Cxy)))]
        tmp = np.matmul(np.transpose(np.conj(E)), np.linalg.pinv(A))
        tmp = np.matmul(np.matmul(tmp, Ao), Eo)
        U,S,V = np.linalg.svd(tmp)
        Orth = np.matmul(U, np.transpose(np.conj(V)))
    elif (mode == 3):
        Orth = np.matmul(np.transpose(np.conj(E)), Eo)
    else:     # SVD
        A = [np.eye(Cx.shape[0],Cx.shape[1]), np.matmul(-iBy, np.transpose(np.conj(Bxy)))]
        Ao =  [np.eye(Cx.shape[0],Cx.shape[1]), np.matmul(-iBy, np.transpose(np.conj(Cxy)))]
        tmp = np.matmul(np.transpose(np.conj(E)), np.linalg.pinv(A))
        tmp = np.matmul(np.matmul(tmp, Ao), Eo)
        tmp = np.matmul(tmp, np.linalg.inv(Do))
        U,S,V = np.linalg.svd(tmp)
        Orth = np.matmul(U, np.transpose(np.conj(V)))

    Mx = np.matmul(np.matmul(E, np.linalg.inv(D)), Orth)
    Mx = np.matmul(np.matmul(Mx, Do), np.transpose(np.conj(Eo)))
    My = np.matmul(iBy, np.transpose(np.conj(Cxy)) - np.matmul(np.transpose(np.conj(Bxy)), Mx))
    newX = np.matmul(X, Mx) + np.matmul(Y, My)

    if (Cx0 != Bx).any():
        snr1 = 10*np.log10(np.sum(np.power(Cx0,2))/np.sum(np.power(Cx0-Bx,2)))
    else:
        snr1 = np.inf
    if (Cxy0 != Bxy).any():
        snr2 = 10*np.log10(np.sum(np.power(Cxy0,2))/np.sum(np.power(Cxy0-Bxy,2)))
    else:
        snr2 = np.inf

    return newX, snr1, snr2, Mx, My