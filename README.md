# Improving training of deep neural networks via Singular Value Bounding

This is the code release for the Singular Value Bounding (SVB) and Bounded Batch Normalization (BBN) methods proposed in the CVPR2017 paper "Improving training of deep neural networks via Singular Value Bounding", authored by Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu.

This work investigates *solution properties of neural networks* that can potentially lead to good performance. Inspired by [orthogonal weight initialization](https://arxiv.org/abs/1312.6120), we propose to constrain the solutions of weight matrices in the orthogonal feasible set during the whole process of network training.

We achieve this by a simple yet effective method called Singular Value Bounding (SVB). In SVB, *all singular values of each weight matrix* are simply bounded in a narrow band around the value of 1. Based on the same motivation, we also propose Bounded Batch Normalization (BBN), which improves [Batch Normalization](https://arxiv.org/abs/1502.03167) by removing its potential risk of ill-conditioned layer transform.

We present both theoretical and empirical results to justify our proposed methods. In particular, *we achieve the state-of-the-art results of 3.06% error rate on CIFAR10 and 16.90% on CIFAR100*, using off-the-shelf network architectures ([Wide ResNets](https://arxiv.org/abs/1605.07146)).

###### Project page: [http://www.aperture-lab.net/research/svb/](http://www.aperture-lab.net/research/svb/)

# Results

#### Controlled studies on CIFAR10 using 20-layer (left) and 38-layer (right) ConvNets (VGG)

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

#### Ablation studies on CIFAR10 using a 68-layer ResNet

#### Wide ResNets on CIFAR10 and CIFAR100

#### Preliminary results on ImageNet 

# Usage

# Acknowledgements
