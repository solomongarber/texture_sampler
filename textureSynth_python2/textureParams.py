


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
    def __init__(self, *args):
        if len(args) > 0: self.pixelStats = args[0]
        if len(args) > 1: self.pixelLPStats = args[1]
        if len(args) > 2: self.autoCorrReal = args[2]
        if len(args) > 3: self.autoCorrMag = args[3]
        if len(args) > 4: self.magMeans = args[4]
        if len(args) > 5: self.cousinMagCorr = args[5]
        if len(args) > 6: self.parentMagCorr = args[6]
        if len(args) > 7: self.cousinRealCorr = args[7]
        if len(args) > 8: self.parentRealCorr = args[8]
        if len(args) > 9: self.varianceHPR = args[9]
        


