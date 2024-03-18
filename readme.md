
# Discriminability-Driven Channel Selection for Out-of-Distribution Detection

This is the source code for paper [Discriminability-Driven Channel Selection for Out-of-Distribution Detection]
by Yue Yuan, Rundong He, YiCong Dong, Zhongyi Han and Yilong Yin.

In this work, we propose propose a new activation-based OOD detection method called DDCS, which introduces a newly-designed function to rectify the extremely low and high activations at the same time.
In this way, LHAct solves the problem of model underconfidence on ID data and overconfidence on OOD data.

## Usage

### 1. Dataset Preparation

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/id_data/ILSVRC-2012/train` and  `./datasets/id_data/ILSVRC-2012/val`, respectively.


#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/ood_data/`.

### 2. Pre-trained Model Preparation

The model we used in the paper is the pre-trained ResNet-50 provided by Pytorch. The download process
will start upon running.


### 3. Activation Calculation
To get activations on the penultimate layer, please run:

```
python get_activation.py 
```

### 4. OOD Detection Evaluation

To reproduce our results, please run:
```
python eval.py 
```


## 5. OOD Detection Results

LHAct achieves state-of-the-art performance averaged on the 4 OOD datasets.

(./results.png)


## The code is improved based on [ReAct: Out-of-distribution Detection With Rectified Activations](https://openreview.net/pdf?id=IBVBtz_sRSm)'s source code.
