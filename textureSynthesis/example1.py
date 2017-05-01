import sys
sys.path.append('pyPyrTools')
import numpy as np
import matplotlib.pyplot as plt
import textureAnalysis as anal
import textureSynthesis as synth

def pgmRead(fname):

    pyver = 3

    f = open(fname, 'rb')
    fmt = str(f.readline())
    if pyver == 3:
        fmt = fmt[2:4]
    else:
        fmt = fmt[0:2]

    fmtList = ['P2', 'P5']
    if fmt not in fmtList:
        print ('PGM file must be of type P2 or P5')
        return

    # Any number of comment lines
    TheLine = str(f.readline())
    while ((pyver == 3) & (TheLine[2] == '#')) | ((pyver == 2) & (TheLine[0] == '#')):
        TheLine  = str(f.readline())

    # dimensions
    if pyver == 3:
        TheLine = TheLine[2:len(TheLine)-3]
    sz = TheLine.split(None, 2)
    xdim = int(sz[0])
    ydim = int(sz[1])
    sz = xdim * ydim

    # Maximum pixel value
    TheLine = str(f.readline())
    if pyver == 3:
        TheLine = TheLine[2:len(TheLine)-3]

    im = np.zeros([sz,1])
    if fmt[1] == '2':
        TheLine  = str(f.readline())
        if pyver == 3:
            TheLine = TheLine[2:len(TheLine)]
        for i in range(len(TheLine)):
            im[i] = int(TheLine[i])
    else:
        TheLine = f.readline()
        if pyver == 3:
            TheLine = TheLine[2:]
        for i in range(len(TheLine)):
            im[i] = TheLine[i]

    f.close()

    im = np.reshape(im,[ydim,xdim])

    return im


im0 = pgmRead('text.pgm')

plt.figure(1)
plt.imshow(im0, cmap='gray')

Nsc = 4 # Number of scales
Nor = 4 # Number of orientations
Na = 9  # Spatial neighborhood is Na x Na coefficients. It must be an odd number!

params = anal.textureAnalysis(im0, Nsc, Nor, Na)

Niter = 25	# Number of iterations of synthesis loop
Nsx = 192	# Size of synthetic image is Nsy x Nsx
Nsy = 128	# WARNING: Both dimensions must be multiple of 2^(Nsc+2)

res, snrP, imS = synth.textureSynthesis(params, [Nsy,Nsx], Niter)

plt.figure(2)
plt.imshow(res, cmap='gray')
plt.show()