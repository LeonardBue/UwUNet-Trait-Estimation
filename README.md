# U-within-U-Net: A Versatile Deep Learning Architecture for Analysis of Hyperspectral Images

## System Requirements
Installing on Linux is recommended (Our machine runs Ubuntu 18.04).

An nVIDIA graphics card with >10GB of GPU memory (Our machine is equipped with an nVIDIA Titan RTX with 24GB of memory).

## Installation
### Environment setup
- Install [Miniconda](https://conda.io/miniconda.html) or Anaconda if necessary.
- Create a Conda environment for the platform and install all the necessary dependencies:
```shell
conda env create -f environment.yml
```
- Activate the environment:
```shell
conda activate fnet
```
- Try executing the test script:
```shell
./scripts/test_run.sh
```
The installation was successful if the script executes without errors.

If installation continues to fail, try downloading the original pytorch-fnet-release_1 repository and replacing bufferedpatchdataset.py and fnet_nn_2d.py in that repo with the same files from this repo.

## The U-within-U-Net
The original U-within-U-Net was developed by:
Manifold, B., Men, S., Hu, R. et al. A versatile deep learning architecture for classification and label-free prediction of hyperspectral images. Nat Mach Intell 3, 306–315 (2021). https://doi.org/10.1038/s42256-021-00309-y

some functions were substituted by up-to-date packages and deprecated code implemntations adjusted.
The last activation function and batch normalization was removed to allow for regression.

The Hyperion images e.g., EO1H0450222004225110PZ, can be accessed here: https://earthexplorer.usgs.gov/


## Training with Hyperion Data for Trait Estimation
The training is separated into two steps, first spaial training and subsequently spectral training using synthetic data.

### Spatial Training
To train the model spatially, by reconstructing the image, execute
```sh
./code/uwu_net/scripts/train_model_UwU_spatial.sh <DATASET> 0
```
where you substitute `<DATASET>` with the correct Hyperion ID

To calculate reconstions of trainng and/or validation data, execute
```sh
./code/uwu_net/scripts/predict_UwU_spatial.sh <DATASET> 0
```

### Spectral Training
To generate synthetic data, execute 
```code/gen_inform_inputs.py```
This will generate all neccessary files to train th model for trait estimation.

Now, in order to train using synthetic data, try executing
```sh
./code/uwu_net/scripts/train_model_UwU_spectral.sh <DATASET> 0
```

### Trait estimation
Subsequently, you can estimate traits for synthetic data by executing
```sh
./code/uwu_net/scripts/predict_UwU_synthetic.sh <DATASET> 0
```
or for the Hyperion image by executing
```sh
./code/uwu_net/scripts/predict_UwU_real.sh <DATASET> 0
```


## Instructions to training/utilizing UwU-Net models on your data
See the AllenCellModeling/pytorch-fnet-release_1 github and readme for other basic functions and training of traditional U-Nets.

Create a 2 column csv file where the first row contains "path_signal,path_target". Then in each column place the corresponding path-to-files for the images to be used for model training. Save the file with the intended model name my_model.csv to the /data/csvs/ folder

Edit the /scripts/train_model_UwU.sh file to the desired training parameters. Edit the /fnet/data/bufferedpatchdataset.py file to match the patch size used in train_model_UwU.sh. Edit the fnet/nn_modules/fnet_nn_UwU.py file to match the desired initial, intermediate, and final channel size for the images and select the correct return line in forward() to match the chosen output channel dimensions.

Once all parameters have been set execute:
```shell
./code/uwu_net/scripts/train_model_UwU.sh my_model 0
```

Once you have trained a model you can execute:
```shell
./code/uwu_net/scripts/predict_UwU.sh my_model 0
```
to predict the withheld test images and the training images.

To use the trained model on new data, edit the test.csv or train.csv file to the desired path to files then reexecute the above script.

Note: reexecuting the prediction will require no existing folder for the model results in the /results/ folder. Rename previous executions to have the model continue predicting.


## Allen Institute Software License
This software license is the 2-clause BSD license plus clause a third clause that prohibits redistribution and use for commercial purposes without further permission.   
Copyright © 2018. Allen Institute.  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.  
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.  
3. Redistributions and use for commercial purposes are not permitted without the Allen Institute’s written permission. For purposes of this license, commercial purposes are the incorporation of the Allen Institute's software into anything for which you will charge fees or other compensation or use of the software to perform a commercial service for a third party. Contact terms@alleninstitute.org for commercial licensing opportunities.  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
