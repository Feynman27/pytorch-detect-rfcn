A pytorch implementation of the baseline  
[RFCN](https://arxiv.org/pdf/1605.06409.pdf) approach used 
in the paper [https://arxiv.org/abs/1710.03958](https://arxiv.org/abs/1710.03958).

## Introduction

This project is a pytorch implementation of the baseline 
RFCN in the Detect to Track paper. 
This repository is influenced by the following implementations:

* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), based on Pytorch

* [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), based on Pycaffe + Numpy

* [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), based on Pytorch + Numpy

* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), based on TensorFlow + Numpy

* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), Pytorch + TensorFlow + Numpy

Our implementation stems heavily from the work 
[jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). 
As in that implementation, this repository has the following qualities: 

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

* **It supports multi-image batch training**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to support multiple images in each minibatch.

* **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It is memory efficient**. We limit the aspect ratio of the images in each roidb and group images 
with similar aspect ratios into a minibatch. As such, we can train resnet101 with batchsize = 2 (4 images) on a 2 Titan X (12 GB). 

* **Supports 4 pooling methods**. roi pooling, roi alignment, roi cropping, and position-sensitive roi pooling. 
More importantly, we modify all of them to support multi-image batch training.

### prerequisites

* Python 2.7
* Pytorch 0.3.0 (0.4.0 may work, but hasn't been tested)
* CUDA 8.0 or higher

### Pretrained Model
The RFCN network weights are initialized using the ImageNet resnet-101 weights. 
The pretrained resnet-101 model can be accessed from 
[here](https://drive.google.com/drive/u/0/folders/1TM9bJ1mod2EipgXHhYscRxkJhrtOGSju) under
the name `res101.pth`

### Training

Below are instructions for training an RFCN network on Imagenet VID+DET.

```
cd pytorch-detect-rfcn
mkdir data
```

Download the ILSVRC VID and DET (TODO: Add public link).

Untar the file:
```bash
tar xf ILSVRC2015.tar.gz
```

We'll refer to this directory as `$DATAPATH`.
Make sure the directory structure looks something like:
```bash
|--ILSVRC2015
|----Annotations
|------DET
|--------train
|--------val
|------VID
|--------train
|--------val
|----Data
|------DET
|--------train
|--------val
|------VID
|--------train
|--------val
|----ImageSets
|------DET
|------VID
```

Create a soft link under `pytorch-detect-rfcn/data`:
```bash
ln -s $DATAPATH/ILSVRC2015 ./ILSVRC
```

Create a directory called `pytorch-detect-rfcn/data/pretrained_model`,
and place the pretrained models into this directory.

Before training, set the correct directory to save and load the trained models.
The default is `./output/models`.
Change the arguments "save_dir" and "load_dir" in trainval_net.py and 
test_net.py to adapt to your environment.

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

### Results
Imagenet VID+DET (Train/Test: imagenet_vid_train+imagenet_det_train/imagenet_vid_val,
scale=600, PS ROI Pooling).

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
Res-101     | 2 | 2 | 1e-3 | 5   | 11   |  -- | 8021MiB   | 70.3

### Build 

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

## Authorship

Contributions to this project have been made by [Thomas Balestri](https://github.com/Feynman27) and 
[Jugal Sheth](https://github.com//jugalsheth92).

