from fnet.data.fnetdataset import FnetDataset
import numpy as np
import torch

from tqdm import tqdm

import pdb


class BufferedPatchDataset(FnetDataset):
    """Dataset that provides chunks/patchs from another dataset."""

    def __init__(self, 
                 dataset,
                 patch_size, 
                 final_chan,
                 buffer_size = 1,
                 buffer_switch_frequency = 720, 
                 npatches = 100000,
                 verbose = True,
                 transform = None,
                 shuffle_images = True,
                 dim_squeeze = None,
    ):
        
        self.counter = 0
        
        self.dataset = dataset
        self.transform = transform
        
        self.buffer_switch_frequency = buffer_switch_frequency
        
        self.npatches = npatches
        self.final_chan = final_chan
        self.buffer = list()
        
        self.verbose = verbose
        self.shuffle_images = shuffle_images
        self.dim_squeeze = dim_squeeze
        
        shuffed_data_order = np.arange(0, len(dataset))

        if self.shuffle_images:
            np.random.shuffle(shuffed_data_order)
        
        
        pbar = tqdm(range(0, buffer_size))
                       
        self.buffer_history = list()
            
        for i in pbar:
            #convert from a torch.Size object to a list
            if self.verbose: pbar.set_description("buffering images")

            datum_index = shuffed_data_order[i]
            datum = dataset[datum_index]
            
            datum_size = datum[0].size()
            
            self.buffer_history.append(datum_index)
            self.buffer.append(datum)
            
        self.remaining_to_be_in_buffer = shuffed_data_order[i+1:]
            
        self.patch_size = [datum_size[0]] + patch_size

            
    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        self.counter +=1
        
        if (self.buffer_switch_frequency > 0) and (self.counter % self.buffer_switch_frequency == 0):
            if self.verbose: print("Inserting new item into buffer")
                
            self.insert_new_element_into_buffer()
        
        return self.get_random_patch()
                       
    def insert_new_element_into_buffer(self):
        #sample with replacement
                       
        self.buffer.pop(0)
        
        if self.shuffle_images:
            
            if len(self.remaining_to_be_in_buffer) == 0:
                self.remaining_to_be_in_buffer = np.arange(0, len(self.dataset))
                np.random.shuffle(self.remaining_to_be_in_buffer)
            
            new_datum_index = self.remaining_to_be_in_buffer[0]
            self.remaining_to_be_in_buffer = self.remaining_to_be_in_buffer[1:]
            
        else:
            new_datum_index = self.buffer_history[-1]+1
            if new_datum_index == len(self.dataset):
                new_datum_index = 0
                             
        self.buffer_history.append(new_datum_index)
        self.buffer.append(self.dataset[new_datum_index])
        
        if self.verbose: print("Added item {0}".format(new_datum_index))


    def get_random_patch(self):
        
        buffer_index = np.random.randint(len(self.buffer))
                                   
        datum = self.buffer[buffer_index]

        starts = np.array([np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0 for d, p in zip(datum[0].size(), self.patch_size)])
        ends = starts + np.array(self.patch_size)
        
        #Set parameters here
        # patch_size = self.patch_size[1:] #dimension of the patch to be sampled in x or y # OV: 256
        # spec_channels = self.patch_size[0] #number of channels in starting tiff
        final_channels = self.final_chan #2 #number of channels in final tiff
        
        #thank you Rory for this weird trick
        ##Rory's weird tricks are no match for my hamfisitng
        index = [slice(s, e) for s,e in zip(starts,ends)]
        
        #index.insert(0, slice(0, spec_channels))
        patch = [(datum[0])[tuple(index)]]
        
        index.pop(0)
        index.insert(0, slice(0, final_channels))
        patch = patch + [(datum[1])[tuple(index)]]
    
        if self.dim_squeeze is not None:
            patch = [torch.squeeze(d, self.dim_squeeze) for d in patch]
        return patch
        
    
    def get_buffer_history(self):
        return self.buffer_history
    
# def _test():
#     # dims_chunk = (2,3,4)
#     dims_chunk = (4,5)
#     ds_test = ChunkDatasetDummy(
#         None,
#         dims_chunk = dims_chunk,
#     )
#     print('Dataset len', len(ds_test))
#     for i in range(3):
#         print('***** {} *****'.format(i))
#         element = ds_test[i]
#         print(element[0])
#         print(element[1])
    
# if __name__ == '__main__':
#     _test()

