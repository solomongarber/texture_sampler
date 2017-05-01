
import sys
sys.path.append('pyPyrTools')
import pyPyrTools as ppt
import textureParams as tparams
import numpy as np
import skew2
import kurt2
import var2
from shift import *
from textureUtils import *


def textureAnalysis(im0, Nsc, Nor, Na):

    # 1D interpolation filter, for scale cross-correlations:
    interp = [-1/16, 0, 9/16, 1, 9/16, 0, -1/16] / np.sqrt(2)

    if np.mod(Na,2) == 0:
        print ('Na is not an odd integer.')
        return

    Ny, Nx = im0.shape[0:2]
    nth = np.log2( min(Ny,Nx)/Na )
    if nth < Nsc:
        print ('Warning: Na will be cut off for levels above #%d !' % np.floor(nth+1))

    la = int( (Na-1)/2 )

    # Pixel statistics
    mn0 = np.amin(im0)
    mx0 = np.amax(im0)
    mean0 = np.mean(im0)
    var0 = var2.var2(im0, mean0)
    skew0 = skew2.skew2(im0, mean0, var0)
    kurt0 = kurt2.kurt2(im0, mean0, var0)
    statg0 = [mean0, var0, skew0, kurt0, mn0, mx0]

    im0 = im0 + (mx0-mn0)/1000 * np.random.randn(Ny,Nx)

    # Build the steerable pyramid
    pyrm0 = ppt.SCFpyr(im0,Nsc,Nor-1)
    pyr0 = pyrm0.pyr
    pind0 = np.array(pyrm0.pyrSize)
    if np.count_nonzero(np.mod(pind0,2)) > 0:
        print ('Algorithm will fail: Some bands have odd dimensions!')
        return

    # Subtract mean of lowBand
    nband = pind0.shape[0] - 1
    pyr0, m = subtract_mean(pyr0, pind0, nband)

    rpyr0 = list(map(lambda i: np.real(pyr0[i]), range(len(pyr0))))
    apyr0 = list(map(lambda i: np.abs(pyr0[i]), range(len(pyr0))))

    # Subtract mean of magnitude
    magMeans0 = np.zeros([pind0.shape[0],1])
    for nband in range(len(pind0)):
        apyr0, magMeans0[nband] = subtract_mean(apyr0, pind0, nband)

    # Compute central autoCorr of lowband
    acr = np.nan * np.ones([Na,Na,Nsc+1])
    nband = pind0.shape[0] - 1
    ch = pyr0[nband]
    mpyr = ppt.SFpyr(np.real(ch),0,0)
    im = mpyr.pyr[1]
    Nly, Nlx = im.shape[0:2]
    Sch = min(Nly,Nlx)
    le = int(min(Sch/2-1, la))
    cy = int(Nly/2)
    cx = int(Nlx/2)

    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.power(np.abs(np.fft.fft2(im)),2)))) / np.prod(ch.shape)
    ac = ac[cy-le:cy+le+1,cx-le:cx+le+1]
    acr[la-le:la+le+1,la-le:la+le+1,Nsc] = ac

    skew0p = np.zeros([Nsc+1,1])
    kurt0p = np.zeros([Nsc+1,1])
    vari = ac[le,le]
    if vari/var0 > 1e-6:
        skew0p[Nsc] = np.mean(np.power(im,3)) / np.power(vari,1.5)
        kurt0p[Nsc] = np.mean(np.power(im,4)) / np.power(vari,2)
    else:
        skew0p[Nsc] = 0
        kurt0p[Nsc] = 3

    # Compute  central autoCorr of each Mag band, and the autoCorr of the combined (non-oriented) band.
    ace = np.nan * np.ones([Na,Na,Nsc,Nor])

    for nsc in range(Nsc-1,-1,-1):
        for nor in range(Nor):
            nband = nsc*Nor + nor + 1
            ch = apyr0[nband]
            Nly, Nlx = ch.shape[0:2]
            Sch = min(Nlx, Nly)
            le = int(min(Sch/2-1,la))
            cx = int(Nlx/2)
            cy = int(Nly/2)
            ac = np.fft.fftshift(np.real(np.fft.ifft2(np.power(np.abs(np.fft.fft2(ch)),2)))) / np.prod(ch.shape[0:2])
            ac = ac[cy-le:cy+le+1,cx-le:cx+le+1]
            ace[la-le:la+le+1,la-le:la+le+1,nsc,nor] = ac

        # Combine ori bands
        bandNums = np.array(range(Nor)) + nsc*Nor + 1

        # Make fake pyramid, containing dummy hi, ori, lo
        fakePind = []
        fakePind.append(pind0[bandNums[0]])
        for i in range(bandNums[0],bandNums[Nor-1]+2):
            fakePind.append(pind0[i])

        fakePyr = []
        fakePyr.append(np.zeros([fakePind[0][0],fakePind[0][1]]))
        for i in range(bandNums[0],bandNums[Nor-1]+1):
            fakePyr.append(rpyr0[i])
        fakePyr.append(np.zeros([fakePind[len(fakePind)-1][0],fakePind[len(fakePind)-1][1]]))

        mpyr1 = mpyr;
        mpyr1.pyr = fakePyr
        mpyr1.pyrSize = fakePind
        ch = mpyr1.reconSFpyr([1]) # recon ori bands only #####################
        im = np.real(expand(im,2)) / 4
        im = im + ch
        ac = np.fft.fftshift(np.real(np.fft.ifft2(np.power(np.abs(np.fft.fft2(im)),2)))) / np.prod(ch.shape)
        ac = ac[cy-le:cy+le+1,cx-le:cx+le+1]
        acr[la-le:la+le+1,la-le:la+le+1,nsc] = ac
        vari = ac[le,le]
        if vari/var0 > 1e-6:
            skew0p[nsc] = np.mean(im**3) / vari**1.5
            kurt0p[nsc] = np.mean(im**4) / vari**2
        else:
            skew0p[nsc] = 0
            kurt0p[nsc] = 3

    # Compute the cross-correlation matrices of the coefficient magnitudes
    # pyramid at the different levels and orientations
    C0 = np.zeros([Nor,Nor,Nsc+1])
    Cx0 = np.zeros([Nor,Nor,Nsc])

    Cr0 = np.zeros([2*Nor,2*Nor,Nsc+1]) + 0.j
    Crx0 = np.zeros([2*Nor,2*Nor,Nsc]) + 0.j

    for nsc in range(0,Nsc):
        firstBnum = nsc*Nor + 1
        cousinSz = np.prod(pind0[firstBnum])

        if nsc < Nsc-1:
            parents = np.zeros([cousinSz,Nor])
            rparents = np.zeros([cousinSz,Nor*2])
            for nor in range(0,Nor):
                nband = (nsc+1)*Nor + nor + 1

                tmp = expand(pyr0[nband],2) / 4
                rtmp = np.real(tmp)
                itmp = np.imag(tmp)
                # Double phase:
                tmp = np.multiply(np.sqrt(np.power(rtmp,2) + np.power(itmp,2)),
                                    np.exp(2 * 1.j * np.arctan2(rtmp,itmp)))
                tmp1 = np.reshape(np.real(tmp), [np.prod(tmp.shape[0:2]),1], order='F')
                for i in range(rparents.shape[0]): rparents[i,nor] = tmp1[i]
                tmp1 = np.reshape(np.imag(tmp), [np.prod(tmp.shape[0:2]),1], order='F')
                for i in range(rparents.shape[0]): rparents[i,Nor+nor] = tmp1[i]

                tmp = np.abs(tmp)
                tmp1 = np.reshape(tmp - np.mean(tmp), [np.prod(tmp.shape[0:2]),1], order='F')
                for i in range(parents.shape[0]):
                    parents[i,nor] = tmp1[i]
        else:
            tmp = np.real(expand(rpyr0[pind0.shape[0]-1],2)) / 4
            rparents = np.zeros([np.prod(tmp.shape[0:2]),5])
            rparents[:,0:1] = np.reshape(tmp, [np.prod(tmp.shape[0:2]),1], order='F')
            rparents[:,1:2] = np.reshape(shift(tmp,[0,1]), [np.prod(tmp.shape[0:2]),1], order='F')
            rparents[:,2:3] = np.reshape(shift(tmp,[0,-1]), [np.prod(tmp.shape[0:2]),1], order='F')
            rparents[:,3:4] = np.reshape(shift(tmp,[1,0]), [np.prod(tmp.shape[0:2]),1], order='F')
            rparents[:,4:5] = np.reshape(shift(tmp,[-1,0]), [np.prod(tmp.shape[0:2]),1], order='F')

            parents = np.array([])

        cousins = np.zeros([cousinSz*Nor,1])
        n = 0
        for i in range(Nor):
            tmp1 = np.reshape(apyr0[firstBnum+i], [np.prod(pind0[firstBnum+i]),1], order='F')
            for j in range(np.prod(pind0[firstBnum+i])):
                cousins[n] = tmp1[j]
                n += 1
                if n == np.prod(cousins.shape):
                    break
            if n == np.prod(cousins.shape):
                break
        cousins = np.reshape(cousins, [cousinSz, Nor], order='F')
        nc = cousins.shape[1]
        if len(parents) > 0:
            nq = parents.shape[1]
        else:
            nq = 0
        C0[0:nc,0:nc,nsc] = np.matmul(np.transpose(np.conj(cousins)), cousins) / cousinSz
        if nq > 0:
            Cx0[0:nc,0:nq,nsc] = np.matmul(np.transpose(cousins),parents) / cousinSz
            if nsc == Nsc:
                C0[0:np,0:np,Nsc] = np.matmul(np.transpose(np.conj(parents)), parents) / (cousinSz/4)

        cousins = np.zeros([cousinSz*Nor,1]) + 0.j
        n = 0
        for i in range(Nor):
            tmp1 = np.reshape(pyr0[firstBnum+i], [np.prod(pind0[firstBnum+i]),1], order='F')
            for j in range(np.prod(pind0[firstBnum+i])):
                cousins[n] = tmp1[j]
                n += 1
                if n == np.prod(cousins.shape):
                    break
            if n == np.prod(cousins.shape):
                break
        cousins = np.reshape(cousins, [cousinSz, Nor], order='F')
        nrc = cousins.shape[1]
        nrp = rparents.shape[1]
        Cr0[0:nrc,0:nrc,nsc] = np.matmul(np.transpose(np.conj(cousins)), cousins) /cousinSz
        if nrp > 0:
            Crx0[:nrc,0:nrp,nsc] = np.matmul(np.transpose(cousins),rparents) / cousinSz
            if nsc == Nsc:
                Cr0[0:nrp,0:nrp,Nsc] = np.matmul(np.transpose(np.conj(rparents)), rparents) / (cousinSz/4)

    # Calculate the mean, range and variance of the LF and HF residuals' energy.

    nband = pind0.shape[0]

    channel = pyr0[0]
    vHPR0 = np.mean(np.power(channel,2))

    statsLPim = [skew0p,kurt0p]

    params = tparams.textureParams(statg0, statsLPim, acr, ace, magMeans0, C0, Cx0, Cr0, Crx0, vHPR0)

    return params