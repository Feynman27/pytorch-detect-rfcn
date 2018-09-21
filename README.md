A pytorch implementation of the paper 
[RFCN](https://arxiv.org/pdf/1605.06409.pdf).

## Introduction

This project is a (refactored) pytorch implementation of RFCN. 
This repository is influenced by the following implementations:

* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), based on Pytorch

* [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), based on Pycaffe + Numpy

* [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), based on Pytorch + Numpy

* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), based on TensorFlow + Numpy

* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), Pytorch + TensorFlow + Numpy

During our implementing, we referred the above implementations, 
especially 
[jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). 
As in that implementation, this repository has the following qualities: 

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

* **It supports multi-image batch training**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to support multiple images in each minibatch.

* **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

Furthermore, since the Detect to Track and Track to Detect implementation 
originally used an R-FCN siamese network and correlation layer, we've added/modified the following:
* **Supports multiple images per roidb entry**. By default, we use 2 images in contiguous frames to define an roidb entry to faciliate
a forward pass through a two-legged siamese network. 

* **It is memory efficient**. We limit the aspect ratio of the images in each roidb and group images 
with similar aspect ratios into a minibatch. As such, we can train resnet101 with batchsize = 2 (4 images) on a 2 Titan X (12 GB). 

* **Supports 4 pooling methods**. roi pooling, roi alignment, roi cropping, and position-sensitive roi pooling. 
More importantly, we modify all of them to support multi-image batch training.

### prerequisites

* Python 2.7
* Pytorch 0.3.0+ (0.2.0 may work, but hasn't been tested)
* CUDA 8.0 or higher

### Data Preparation

### Pretrained Model

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, PSROI_POOLING, ROI_Pooing, ROI_Align and ROI_Crop. 
The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train 

Before training, set the right directory to save and load the trained models. 
Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train an RFCN D&T model with resnet-101 on Imagenet VID, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
    --cuda \
    --dataset imagenet_vid \
    --cag \
    --lr $LEARNING_RATE \
    --bs $BATCH_SIZE \
```
where 'bs' is the batch size with default 1. 
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. 
**On 2 Titan Xps with 12G memory, it can be up to 2 (4 images, 2 per GPU)**.

## Test

## Demo

## Authorship

Contributions to this project have been made by [Thomas Balestri](https://github.com/Feynman27) and 
[Jugal Sheth](https://github.com//jugalsheth92).

