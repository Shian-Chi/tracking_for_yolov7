from yolo.myProject.parameter import Parameters
para = Parameters()


class Distance:
    def __init__(self, objSize):
        self.distance = None
        self.objSize = objSize

    def getDistance(self, objPixel):
        self.distance = para.Focal_Length * self.objSize / objPixel
        return self.distance
