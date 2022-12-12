import rasterio


class RasterReader:
    """This class is used to open and process the contents of a raster file.

    Examples:
        reader = rasterReader.RasterReader(path="file.tif")
        file_image = reader.open()

        with rasterReader.RasterReader(path="file2.tif") as reader:
            file2_image = reader.read()

    The load function will get a 3D ZYX array from a raster file.
    """

    def __init__(self, file_path):
        # nothing yet!
        self.filePath = file_path
        self.rast = rasterio.open(self.filePath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.rast.close()

    def load(self):
        """This will get an entire z stack from a raster file.

        :return: A 3D ZYX slice from the raster file.
        """
        return self.rast.read()

    def load_slice(self, z=0, c=0, t=0):
        """This will get a single slice out of the z stack of a rast file.

        :param z: The z index within the tiff stack
        :param c: An arbitrary c index that does nothing
        :param t: An arbitrary t index that does nothing
        :return: A 2D YX slice from the tiff file.
        """
        index = z
        data = self.rast.read(index)
        return data

    def get_metadata(self):
        return None

    def size_z(self):
        return self.rast.count -14# len(self.rast.count)

    def size_c(self):
        return 1

    def size_t(self):
        return 1

    def size_x(self):
        return self.rast.shape[1]

    def size_y(self):
        return self.rast.shape[0]

    def dtype(self):
        return self.rast.dtypes[0]

    def profile(self):
        return self.rast.profile