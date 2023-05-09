class Weed():
    def __init__(self, patchnr,patchsize,spreadrange, reproductionrate,standarddeviation,saturation,plantattach):
        self.type = "weed"
        self.patchnr = patchnr
        self.patchsize = patchsize
        self.spreadrange = spreadrange
        self.reproductionrate = reproductionrate
        self.reproductionrateSTD = standarddeviation
        self.saturation = saturation
        self.plantattach = plantattach