import aicsimage.io as io
import matplotlib.pyplot as plt

class RasterReader(object):
    def __init__(self, filepath):
        with io.rasterReader.RasterReader(filepath) as reader:
            self.raster = reader.rast.read()
            self.profile = reader.rast.profile

    def get_raster(self, start=0):
        """
        Returns the raster for the specified channels.
        start = 15 for input signal to ignore additional inform inputs
        """
        return self.raster[start:, ...]

