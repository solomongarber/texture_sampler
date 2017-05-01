class textureParams:
    pixelStats = []
    pixelLPStats = []
    autoCorrReal = []
    autoCorrMag = []
    magMeans = []
    cousinMagCorr = []
    parentMagCorr = []
    cousinRealCorr = []
    parentRealCorr = []
    varianceHPR = 0

    #constructor
    def __init__(self, pixelStats, pixelLPStats, autoCorrReal, autoCorrMag, magMeans,
        cousinMagCorr, parentMagCorr, cousinRealCorr, parentRealCorr, varianceHPR):
        self.pixelStats = pixelStats
        self.pixelLPStats = pixelLPStats
        self.autoCorrReal = autoCorrReal
        self.autoCorrMag = autoCorrMag
        self.magMeans = magMeans
        self.cousinMagCorr = cousinMagCorr
        self.parentMagCorr = parentMagCorr
        self.cousinRealCorr = cousinRealCorr
        self.parentRealCorr = parentRealCorr
        self.varianceHPR = varianceHPR