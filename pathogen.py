class Pathogen():
    def __init__(self,patchnr,infectionduration,spreadrange,reproductionfraction,reproductionrate,standarddeviation, saturation):
        self.type = "pathogen"
        self.patchnr = patchnr
        self.infectionduration = infectionduration
        self.spreadrange = spreadrange
        self.reproductionfraction = reproductionfraction
        self.reproductionrate = reproductionrate
        self.reproductionrateSTD = standarddeviation
        self.saturation = saturation