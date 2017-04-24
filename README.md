# Improving training of deep neural networks via Singular Value Bounding

This is the code release for the Singular Value Bounding (SVB) and Bounded Batch Normalization (BBN) methods proposed in the CVPR2017 paper "Improving training of deep neural networks via Singular Value Bounding", authored by Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu.

This work investigates solution properties of neural networks that can potentially lead to good performance. Inspired by orthogonal weight initialization, we propose to constrain the solutions of weight matrices in the orthogonal feasible set during the whole process of network training.

We achieve this by a simple yet effective method called Singular Value Bounding (SVB). In SVB, all singular values of each weight matrix are simply bounded in a narrow band around the value of $1$. Based on the same motivation, we also propose Bounded Batch Normalization (BBN), which improves Batch Normalization by removing its potential risk of ill-conditioned layer transform.

We present both theoretical and empirical results to justify our proposed methods. In particular, we achieve the state-of-the-art results of 3.06% error rate on CIFAR10 and 16.90% on CIFAR100, using off-the-shelf network architectures (Wide ResNets).
