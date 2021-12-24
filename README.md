## LBH Measure Python
This repository contains the SDK for LBH Measure. 
It can be used for training and deployment 

## LBH Measure SDK
### Model Builder
This class contains all the training related code. It's a `pytorch-lightning` module. 
It expects a config file for training. The file is preset in `lbh_measure/conf.yml`
```
# model_path: /Users/nikhil.k/data/dev/lbh/dgcnn.pytorch/pretrained/semseg/model_6.t7
# model_path: /Users/nikhil.k/data/dev/lbh/dgcnn.pytorch/pretrained/model.partseg.t7

model: dgcnn
cuda: False
#pcd_dir: '/Users/nikhil.k/data/dev/lbh/converted-bag/'
pcd_dir: '/Users/nikhil.k/data/dev/lbh/udaan-measure/annotation/converted_pcd_nov_25'
#train_data: '/Users/nikhil.k/data/dev/lbh/converted-bag/labels/train_labels'
train_data: '/Users/nikhil.k/data/dev/lbh/udaan-measure/annotation/labels'
test_data: '/Users/nikhil.k/data/dev/lbh/udaan-measure/annotation/labels'
#test_data: '/Users/nikhil.k/data/dev/lbh/converted-bag/labels/test_labels'
model_path: '/Users/nikhil.k/Downloads/epoch=72-step=802.ckpt'
embs: 1024
dropout: 0.5
emb_dims: 1024
k: 9
lr: 1e-3
scheduler: cos
epochs: 100
exp_name: test_1
batch_size: 4
num_workers: 16
test_batch_size: 1
downsample_factor: 0.01
use_sgd: False
```
Here, the model hyper parameters are defined. These parameters are logged to MLFlow as well.

### BagDataset
This class is the cornerstone for handling and pre-processing all the data to the model.
The class has static methods that can are called during training and while serving. 

### Model - DGCNN_SemSeg
The model code is in this file. The details of the implementation are found [here](https://github.com/AnTao97/dgcnn.pytorch). 

### model_inference
The `main()` function is the common function to invoke the model. 
