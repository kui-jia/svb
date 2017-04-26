# Improving training of deep neural networks via Singular Value Bounding

This is the code release for the Singular Value Bounding (SVB) and Bounded Batch Normalization (BBN) methods proposed in the CVPR2017 paper "Improving training of deep neural networks via Singular Value Bounding", authored by Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu.

This work investigates *solution properties of neural networks* that can potentially lead to good performance. Inspired by [orthogonal weight initialization](https://arxiv.org/abs/1312.6120), we propose to constrain the solutions of weight matrices in the orthogonal feasible set during the whole process of network training.

We achieve this by a simple yet effective method called Singular Value Bounding (SVB). In SVB, *all singular values of each weight matrix* are simply bounded in a narrow band around the value of 1. Based on the same motivation, we also propose Bounded Batch Normalization (BBN), which improves [Batch Normalization](https://arxiv.org/abs/1502.03167) by removing its potential risk of ill-conditioned layer transform.

We present both theoretical and empirical results to justify our proposed methods. In particular, *we achieve the state-of-the-art results of 3.06% error rate on CIFAR10 and 16.90% on CIFAR100*, using off-the-shelf network architectures ([Wide ResNets](https://arxiv.org/abs/1605.07146)).

###### Project page: [http://www.aperture-lab.net/research/svb/](http://www.aperture-lab.net/research/svb/)

# Results

#### Controlled studies on CIFAR10 using 20-layer (left) and 38-layer (right) ConvNets (VGG)

![alt text](http://www.aperture-lab.net/research/svb/ConvNetStudies.png)

Validation curves on CIFAR10 using two ConvNets of 20 and 38 weight layers respectively. Blue lines are results by SGD with momentum. Red lines are results by SVB at different values of \epsilon (0.01, 0.05, 0.2, 0.5, 1) in Algorithm 1 of the paper. Black lines are results using both SVB (fixing \epsilon = 0.05) and BBN at different values of \tilde{\epsilon} (0.01, 0.05, 0.2, 0.5, 1)) in Algorithm 2 of the paper. The left two figures are from the 20-layer ConvNet, and the right two ones are from the 38-layer ConvNet.

#### Ablation studies on CIFAR10 using a 68-layer ResNet

| Training methods        | Error rate (%)           | 
| ------------- |:-------------:| 
| SGD with momentum + BN      | 6.10 (6.22 +/- 0.14) | 
| SVB + BN      | 5.65 (5.79 +/- 0.10)      |  
| SVB + BBN | 5.37 (5.49 +/- 0.11)     | 

Ablation studies on CIFAR10, using a pre-activation ResNet with 68 weight layers of 3 x 3 convolutional filters. Results are in the format of *best (mean + std)* over 5 runs. Standard data augmentation (4 pixels zero-padding plus horizontal flipping) is used.

#### Results on CIFAR10 and CIFAR100 using Wide ResNets

| Methods                    | CIFAR10           | CIFAR100            | # layers           | # params              | 
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| 
| Wide ResNet W/O SVB+BBN    | 3.78 | 19.92 | 28 | 36.5M |
| Wide ResNet WITH SVB+BBN   | 3.24 | 17.47 | 28 | 36.5M |
| Wider ResNet W/O SVB+BBN   | 3.64 | 19.25 | 28 | 94.2M |
| Wider ResNet WITH SVB+BBN  | 3.06 | 16.90 | 28 | 94.2M |

*Wide ResNet* and *Wider ResNet* in the table above respectively refer to the architectures of “WRN-28-10” and “WRN-28-16” as in [Wide Residual Networks](https://arxiv.org/abs/1605.07146). Standard data augmentation (4 pixels zero-padding plus horizontal flipping) is used. 

#### Preliminary results on ImageNet 

| Training methods        | Top-1 error (%)           |  Top-1 error (%)    |
| ------------- |:-------------:|:-------------:| 
| Our Inception-ResNet     | 21.61 | 5.91 |
| Our Inception-ResNet WITH SVB+BN      | 21.20 | 5.57 |

Results of single-model ([Inception-ResNet](https://arxiv.org/abs/1602.07261)) and single-crop testing on the ImageNet validation set.   

# Usage

#### Installation 

The code depends on the [Torch library](http://torch.ch/). Please install torch first.

#### Training

We take a pre-activation version of ResNet as the example to explain how to train a deep network using SVB and BBN methods. 

SGD with momentum

`th main.lua -cudnnSetting deterministic -netType PreActResNet -ensembleID 1 -BN true -nBaseRecur 11 -kWRN 1 -lrDecayMethod exp -lrBase 0.5 -lrEnd 0.001 -batchSize 128 -nEpoch 160 -nLRDecayStage 80 -weightDecay 0.0001 -svBFlag false -bnsBFlag false`

SGD with momentum with SVB turned on

`th main.lua -cudnnSetting deterministic -netType PreActResNet -ensembleID 1 -BN true -nBaseRecur 11 -kWRN 1 -lrDecayMethod exp -lrBase 0.5 -lrEnd 0.001 -batchSize 128 -nEpoch 160 -nLRDecayStage 80 -weightDecay 0.0001 -svBFlag true -svBFactor 1.5 -svBIter 391 -bnsBFlag false`

SGD with momentum with both SVB and BBN turned on

`th main.lua -cudnnSetting deterministic -netType PreActResNet -ensembleID 1 -BN true -nBaseRecur 11 -kWRN 1 -lrDecayMethod exp -lrBase 0.5 -lrEnd 0.001 -batchSize 128 -nEpoch 160 -nLRDecayStage 80 -weightDecay 0.0001 -svBFlag true -svBFactor 1.5 -svBIter 391 -bnsBFlag true -bnsBFactor 2 -bnsBType BBN`

`svbFactor` is set as 1 plus the \epsilon value of Algorithm 1. `bnsBFactor` is set as 1 plus the \tilde{\epsilon} value of Algorithm 2. One may also set `bnsBType` as `rel` to get even better performance. Please refer to the file `optsArgParse.lua` for setting of other hyperparameters.   


# Acknowledgements
