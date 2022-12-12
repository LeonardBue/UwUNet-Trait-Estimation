import torch.utils.data
from fnet.data.fnetdataset import FnetDataset
from fnet.data.rasterreader import RasterReader

import fnet.transforms as transforms

import joblib
import pandas as pd
import sys

sys.path.append('../../code/')
from utils import make_scaler, id_from_path


class RasterDataset(FnetDataset):
    """Dataset for Rasters as Tif files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.do_nothing], 
                    transform_target = None, 
                    path_model: str = None,
                    scale_signal = 'MinMaxScaler',
                    scale_target = None,
                    validation = False,
                    start: int = 0):
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        assert all(i in self.df.columns for i in ['path_signal', 'path_target'])
        #print('transform_source is ' + str(transform_source))
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.path_model = path_model
        self.scale_signal = scale_signal
        self.scale_target = scale_target
        self.validation = validation
        self.start = start

    def __getitem__(self, index):
        """
        takes a Raster dataset and a global index value and creates a list of arrays that represent the signal and target images in the path to file csv

        RasterDataset -> list of arrays
        """
        element = self.df.iloc[index, :]
        #print("element['path_signal'] is: " + str(element['path_signal']))#
        im_out = [RasterReader(element['path_signal']).get_raster(start=self.start)]
        #print((im_out[0]).size)
        if isinstance(element['path_target'], str):
            im_out.append(RasterReader(element['path_target']).get_raster())
            #print((im_out[1]).size)
        
        if self.transform_source is not None:
            #print("TRANSFORMING SOURCE")
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if self.transform_target is not None and (len(im_out) > 1):
            #print("TRANSFORMING TARGET")
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])

        if self.scale_signal is not None:
            id = id_from_path(element['path_signal'])
            if not self.validation:
                path_img_scaler = '/'.join([self.path_model,'scaler','train',id+'_'+'signal'+'_scaler.pkl'])
            else:
                path_img_scaler = '/'.join([self.path_model,'scaler','train',id+'_'+'signal'+'_scaler.pkl'])
            scaler = joblib.load(path_img_scaler)
            array = im_out[0].reshape((-1, im_out[0].shape[1]*im_out[0].shape[2])).transpose()
            array = scaler.transform(array)
            im_out[0] = array.transpose().reshape((-1, im_out[0].shape[1], im_out[0].shape[2]))

        if self.scale_target is not None:
            id = id_from_path(element['path_target'])
            if not self.validation:
                path_img_scaler = '/'.join([self.path_model,'scaler','train',id+'_'+'target'+'_scaler.pkl'])
            else:
                path_img_scaler = '/'.join([self.path_model,'scaler','train',id+'_'+'target'+'_scaler.pkl'])
            scaler = joblib.load(path_img_scaler)
            array = im_out[1].reshape((-1, im_out[1].shape[1]*im_out[1].shape[2])).transpose()
            array = scaler.transform(array)
            im_out[1] = array.transpose().reshape((-1, im_out[1].shape[1], im_out[1].shape[2]))


        im_out = [torch.from_numpy(im).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        ##comment this line out for hyperspectral -Bryce
        #im_out = [torch.unsqueeze(im, 0) for im in im_out]
        #print("im-out is" + str(im_out))#
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index):
        return self.df.iloc[index, :].to_dict()
    
    def get_profile(self, index):
        element = self.df.iloc[index, :]
        return RasterReader(element['path_target']).profile
