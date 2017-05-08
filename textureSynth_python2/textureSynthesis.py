

import sys
sys.path.append('pyPyrTools')
import pyPyrTools as ppt
import textureParams

import numpy as np
import skew2
import kurt2
import var2
import shift
import mkAngle
import range2
from textureUtils import *


def textureSynthesis(params, im0, *args): # Niter, cmask, imask

    if len(args) > 0:
        Niter = args[0]
    else:
        Niter = 50

    if len(args) > 1:
        cmask = args[1] > 0.5
    else:
        cmask = np.ones([4, 1])

    imaskexist = False
    if len(args) > 2:
        imask = args[2]
        imaskexist = True

    # Extract parameters
    statg0 = params.pixelStats
    mean0 = statg0[0]
    var0 = statg0[1]
    skew0 = statg0[2]
    kurt0 = statg0[3]
    mn0 = statg0[4]
    mx0 = statg0[5]
    statsLPim = params.pixelLPStats
    skew0p = statsLPim[0]
    kurt0p = statsLPim[1]
    vHPR0 = params.varianceHPR
    acr0 = params.autoCorrReal
    ace0 = params.autoCorrMag
    magMeans0 = params.magMeans
    C0 = params.cousinMagCorr
    Cx0 = params.parentMagCorr
    Crx0 = params.parentRealCorr

    # Extract {Nsc, Nor, Na} from params
    tmp = params.autoCorrMag.shape
    Na = tmp[0]
    Nsc = tmp[2]
    Nor = tmp[len(tmp)-1]*(len(tmp) == 4) + (len(tmp) < 4)
    la = (Na-1) / 2

    # If im0 is a vector of length 2, create Gaussian white noise image of this
    # size, with desired pixel mean and variance.  If vector length is
    # 3,  use the 3rd element to seed the random number generator.
    if len(im0) <= 3:
        if len(im0) == 3:
            im0 = im0[0:2]
        im = mean0 + np.sqrt(var0)*np.random.randn(im0[0], im0[1])
    else:
        im = im0

    # If the spatial neighborhood Na is too big for the lower scales,
    # "modacor22.m" will make it as big as the spatial support at
    # each scale
    Ny, Nx = im.shape[0:2]
    nth = np.log2(min(Ny, Nx)/Na)
    if nth < Nsc+1:
        print ('Warning: Na will be cut off for levels above #%d !\n' % np.floor(nth))

    if imaskexist and (imask.size > 0):
        if imask.shape[0] != np.prod(im.shape):
            print ('imask size %d does not match image dimensions [%d,%d]' % (imask.shape[0], im.shape[0], im.shape[1]))
            return [], [], []
        if imask.shape[1] == 1:
            masky = np.array(map(lambda x: -np.pi/2 + 2*np.pi/Ny * x, range(Ny/2)))
            maskx = np.array(map(lambda x: -np.pi/2 + 2*np.pi/Nx * x, range(Nx/2)))
            mask = np.transpose(masky) * maskx
            mask = np.power(mask,2)
            aux = np.zeros(im.shape)
            aux[Ny/4:Ny/4+Ny/2+1,Nx/4:Nx/4+Nx/2+1] = mask
            mask = aux
        else:
            mask = np.reshape(imask[:, 0], im.shape, order='F')

    prev_im = im

    nq = 0
    Nq = int(np.floor(np.log2(Niter)))
    imS = np.zeros([Ny, Nx, Nq+1])

    # MAIN LOOP
    snr2 = np.zeros([Niter, Nsc+1])
    snr7 = np.zeros([Niter, 2*(Nsc+1)+4])
    snr3 = np.zeros([Niter, Nsc])
    snr4 = np.zeros([Niter, Nsc])
    snr4r = np.zeros([Niter, Nsc])
    snr1 = np.zeros([Niter, Nsc*Nor])
    snr6 = np.zeros([Niter, 1])
    for niter in range(Niter):
        p = 1
        # Build the steerable pyramid
        pyrm = ppt.SCFpyr(im, Nsc, Nor-1)
        pyr = pyrm.pyr
        pind = np.array(pyrm.pyrSize)

        if np.mod(pind, 4).any():
            print ('Algorithm will fail: band dimensions are not all multiples of 4!')
            return [], [], []

        # Subtract mean of lowBand
        nband = pind.shape[0] - 1
        pyr, m = subtract_mean(pyr, pind, nband)

        apyr = map(lambda i: np.abs(pyr[i]), range(len(pyr)))

        # Adjust autoCorr of lowBand
        nband = pind.shape[0] - 1
        ch = pyr[nband]
        Sch = min(ch.shape[0]/2, ch.shape[1]/2)
        nz = np.sum(~np.isnan(acr0[:, :, Nsc]))
        lz = (np.sqrt(nz)-1) / 2
        le = min(Sch/2-1, lz)
        im = np.real(ch)  # Reconstructed image: initialize to lowband
        mpyrm = ppt.SFpyr(im, 0, 0)
        mpyr = mpyrm.pyr
        mpind = mpyrm.pyrSize
        im = mpyr[1]
        vari = acr0[la, la, Nsc]

        if cmask[1]:
            if vari/var0 > 1e-4:
                im, snr2[niter, Nsc], Chf = modacor22(im, acr0[la-le:la+le+1, la-le:la+le+1, Nsc], p)
            else:
                im = im * np.sqrt(vari/var2.var2(im))
            if var2.var2(np.imag(ch)) / var2.var2(np.real(ch)) > 1e-6:
                print('Discarding non-trivial imaginary part, lowPass autoCorr!')
            im = np.real(im)

        if cmask[0]:
            if vari/var0 > 1e-4:
                im, snr7[niter, 2*Nsc] = modskew(im, skew0p[Nsc], p)	# Adjusts skewness 
                im, snr7[niter, 2*Nsc+1] = modkurt(im, kurt0p[Nsc], p)	# Adjusts kurtosis 

        # Subtract mean of magnitude
        if cmask[2]:
            magMeans = np.zeros([pind.shape[0], 1])
            for nband in range(pind.shape[0]):
                apyr, magMeans[nband] = subtract_mean(apyr, pind, nband)

        # Coarse-to-fine loop
        for nsc in range(Nsc-1, -1, -1):
            firstBnum = nsc*Nor + 1
            cousinSz = np.prod(pind[firstBnum])

            # Interpolate parents
            if cmask[2] or cmask[3]:
                if nsc < Nsc-1:
                    parents = np.zeros([cousinSz, Nor])
                    rparents = np.zeros([cousinSz, Nor*2])
                    for nor in range(Nor):
                        nband = (nsc+1)*Nor + nor + 1

                        tmp = expand(pyr[nband], 2) / 4
                        rtmp = np.real(tmp)
                        itmp = np.imag(tmp)
                        tmp = np.sqrt(rtmp**2 + itmp**2) * np.exp(2 * 1.j * np.arctan2(rtmp, itmp))
                        rparents[:, nor:nor+1] = np.reshape(np.real(tmp), [np.prod(tmp.shape), 1],
                                                            order='F')
                        rparents[:, Nor+nor:Nor+nor+1] = np.reshape(np.imag(tmp), [np.prod(tmp.shape),1], order='F')

                        tmp = np.abs(tmp)
                        parents[:, nor:nor+1] = np.reshape(tmp-np.mean(tmp),
                                                           [np.prod(tmp.shape), 1], order='F')
                else:
                    rparents = []
                    parents = []

            if cmask[2]:
                # Adjust cross-correlation with MAGNITUDES at other orientations/scales:
                cousins = np.zeros([cousinSz, Nor])
                for i in range(Nor):
                    tmp1 = np.reshape(apyr[firstBnum+i], [np.prod(pind[firstBnum+i]), 1], order='F')
                    cousins[:, i:i+1] = tmp1
                nc = cousins.shape[1]
                if len(parents) > 0: np1 = parents.shape[1]
                else: np1 = 0
                if np1 == 0:
                    cousins, snr3[niter, nsc], M = adjustCorr1s(cousins, C0[0:nc, 0:nc, nsc], 2, p)
                else:
                    cousins, snr3[niter,nsc], snr4[niter,nsc], Mx, My = adjustCorr2s(cousins, C0[0:nc,0:nc,nsc], parents, Cx0[0:nc,0:np1,nsc], 3, p)
                if var2.var2(np.imag(cousins)) / var2.var2(np.real(cousins)) > 1e-6:
                    print('Non-trivial imaginary part, mag crossCorr, lev=%d!\n' % nsc)
                else:
                    cousins = np.real(cousins)
                    for i in range(Nor):
                        apyr[firstBnum+i] = np.reshape(cousins[:, i], pind[firstBnum+i], order='F')

                # Adjust autoCorr of mag responses
                nband = nsc*Nor + 1
                Sch = min(pind[nband][0]/2, pind[nband][1]/2)
                nz = np.sum(~np.isnan(ace0[:, :, nsc, 0]))
                lz = (np.sqrt(nz)-1) / 2
                le = int(min(Sch/2-1, lz))
                for nor in range(Nor):
                    nband = nsc*Nor + nor + 1
                    ch = apyr[nband]
                    ch, snr1[niter, nband-1], Chf = modacor22(ch, ace0[la-le:la+le+1,
                                                                       la-le:la+le+1, nsc, nor], p)
                    ch = np.real(ch)
                    apyr[nband] = ch
                    # Impose magnitude:
                    mag = apyr[nband] + magMeans0[nband]
                    mag = mag * (mag > 0)
                    pyr[nband] = (pyr[nband] * (mag / (np.abs(pyr[nband]))) +
                                  (np.abs(pyr[nband]) < np.finfo(float).eps))

            # Adjust cross-correlation of REAL PARTS at other orientations/scales
            cousins = np.zeros([cousinSz, Nor])
            for i in range(Nor):
                tmp1 = np.reshape(np.real(pyr[firstBnum+i]),
                                  [np.prod(pind[firstBnum+i]), 1], order='F')
                cousins[:, i:i+1] = tmp1
            Nrc = cousins.shape[1]
            if len(rparents) > 0: Nrp = rparents.shape[1]
            else: Nrp = 0

            if cmask[3] and (Nrp != 0):
                a3 = 0
                a4 = 0
                for nrc in range(Nrc):
                    cou = cousins[:, nrc]
                    cou = np.reshape(cou, [len(cou), 1], order='F')
                    cou, s3, s4, Mx, My = adjustCorr2s(cou, np.mean(cou**2), rparents,
                                                       Crx0[nrc, 0:Nrp, nsc], 3)
                    a3 = s3 + a3
                    a4 = s4 + a4
                    cousins[:, nrc:nrc+1] = cou
                snr4r[niter, nsc] = a4 / Nrc

            if var2.var2(np.imag(cousins))/var2.var2(np.real(cousins)) > 1e-6:
                print('Non-trivial imaginary part, real crossCorr, lev=%d!\n' % nsc)
            else:
                # NOTE: THIS SETS REAL PART ONLY - signal is now NONANALYTIC!
                for i in range(Nor):
                    pyr[firstBnum+i] = np.reshape(cousins[:, i], pind[firstBnum+i], order='F')

            # Re-create analytic subbands
            dims = pind[firstBnum]
            ctr = np.int32(np.ceil((dims+0.5)/2))
            ang = mkAngle.mkAngle((dims[0], dims[1]), 0, ctr)
            ang[ctr[0]-1, ctr[1]-1] = -np.pi/2

            for nor in range(Nor):
                nband = nsc*Nor + nor + 1
                ch = pyr[nband]
                ang0 = np.pi*nor/Nor
                xang = np.mod(ang-ang0+np.pi, 2*np.pi) - np.pi
                amask = 2*(np.abs(xang) < np.pi/2) + (np.abs(xang) == np.pi/2)
                amask[ctr[0]-1, ctr[1]-1] = 1
                amask[:, 0] = 1
                amask[0, :] = 1
                amask = np.fft.fftshift(amask)
                ch = np.fft.ifft2(amask*np.fft.fft2(ch))	# "Analytic" version
                pyr[nband] = ch

            # Combine ori bands
            bandNums = np.array(range(Nor)) + nsc*Nor + 1  # ori bands only
            # Make fake pyramid, containing dummy hi, ori, lo
            fakePind = []
            fakePind.append(pind[bandNums[0]])
            for i in range(bandNums[0], bandNums[Nor-1]+2):
                fakePind.append(pind[i])

            fakePyr = []
            fakePyr.append(np.zeros([fakePind[0][0], fakePind[0][1]]))
            for i in range(bandNums[0], bandNums[Nor-1]+1):
                fakePyr.append(np.real(pyr[i]))
            fakePyr.append(np.zeros([fakePind[len(fakePind)-1][0], fakePind[len(fakePind)-1][1]]))

            mpyr1 = mpyrm
            mpyr1.pyr = fakePyr
            mpyr1.pyrSize = fakePind
            ch = mpyr1.reconSFpyr([1]) # recon ori bands only #####################

            im = np.real(expand(im, 2)) / 4
            im = im + ch
            vari = acr0[la, la, nsc]

            if cmask[1]:
                if vari/var0 > 1e-4:
                    im, snr2[niter,nsc], Chf = modacor22(im, acr0[la-le:la+le+1, la-le:la+le+1, nsc], p)
                else:
                    im = im * np.sqrt(vari/var2.var2(im))
            im = np.real(im)

            if cmask[0]:
                # Fix marginal stats
                if vari/var0 > 1e-4:
                    im, snr7[niter, 2*nsc-1] = modskew(im, skew0p[nsc], p)       # Adjusts skewness
                    im, snr7[niter, 2*nsc] = modkurt(im, kurt0p[nsc], p)         # Adjusts kurtosis

        # Adjust variance in HP, if higher than desired
        if cmask[1] or cmask[2] or cmask[3]:
            ch = pyr[0]
            vHPR = np.mean(np.power(ch, 2))
            if vHPR > vHPR0:
                ch = ch * np.sqrt(vHPR0/vHPR)
                pyr[0] = ch

        mpyr1 = pyrm
        mpyr1.pyr = pyr
        mpyr1.pyrSize = pind
        im = im + mpyr1.reconSFpyr([0])  # recon hi only

        # Pixel statistics
        means = np.mean(im)
        vars = var2.var2(im, means)
        snr7[niter, 2*(Nsc+1)] = snr(var0, var0-vars)
        im = im - means  			# Adjusts mean and variance
        mns, mxs = range2.range2(im + mean0)
        snr7[niter, 2*(Nsc+1)+1] = snr(mx0-mn0, np.sqrt((mx0-mxs)**2+(mn0-mns)**2))
        if cmask[0]:
            im = im * np.sqrt(((1-p)*vars + p*var0)/vars)
        im = im + mean0
        if cmask[0]:
            im, snr7[niter, 2*(Nsc+1)+2] = modskew(im, skew0, p) # Adjusts skewness (keep mean and variance)
            im, snr7[niter, 2*(Nsc+1)+3] = modkurt(im, kurt0, p) # Adjusts kurtosis (keep mean and variance, but not skewness)
            im = np.maximum(np.minimum(im, (1-p)*im.max()+p*mx0), (1-p)*im.min()+p*mn0)		# Adjusts range (affects everything)
        else:
            snr7[niter, 2*(Nsc+1)+2] = snr(skew0, skew0-skew2.skew2(im))
            snr7[niter, 2*(Nsc+1)+3] = snr(kurt0, kurt0-kurt2.kurt2(im))

        # Force pixels specified by image mask
        if imaskexist and (imask.size > 0):
            im = mask*np.reshape(imask[:, 2-(imask.shape[1] == 1)], im.shape, order='F') + (1-mask)*im

        snr6[niter, 0] = snr(im-mean0, im-prev_im)

        if np.floor(np.log2(niter)) == np.log2(niter):
            # imS[:,:,nq] = im
            nq += 1
            print nq

        tmp = prev_im
        prev_im = im

        # accelerator
        alpha = 0.8
        im = im + alpha*(im - tmp)

    im = prev_im

    snrP = [snr7, snr2, snr1, snr3, snr4, snr4r, snr6]

    return im, snrP, imS


