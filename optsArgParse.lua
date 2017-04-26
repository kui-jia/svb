--[[

Initialzie parameter settings, taking either default values or change by command line

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding", CVPR 2017.
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine() -- initialize the torch CmdLine class
   cmd:text() 
   cmd:text('Parameter settings used for this experiment')
   
    ------------ General options --------------------
   cmd:option('-dataFolder', 'Data', 'Path to dataset')
   cmd:option('-gpuStartID', 1,          'This program will use the GPUs with IDs of [opts.gpuStartID, opts.gpuStartID+opts.nGPU-1]')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')                                              -- speed relevant
   cmd:option('-expFolder',  'Exps', 'Directory in which to save resulting experimental files/checkpoints')
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'true', 'Use optnet to reduce memory usage')                                 -- speed relevant?
   
   cmd:option('-dataset',         'cifar10',        'imagenet | cifar10 | cifar100')
   cmd:option('-manualSeed',      0,                'Manually set RNG seed')
   cmd:option('-nGPU',            8,                'Number of GPUs to use by default')                        -- speed relevant
   cmd:option('-cudnnSetting',    'deterministic',  'Options: fastest | default | deterministic')              -- speed relevant?
   cmd:option('-nThreads',        4,                'number of data loading threads')                          -- speed relevant
   
   ------------- Training options --------------------
   cmd:option('-testFlag',      'false',   'true for only testing on the validation/test data')
   cmd:option('-contPoint',     0,         'to specify an epoch/point to continue the training if > 0' )
   cmd:option('-startEpoch',    1,         'useful to specify a starting epoch to train')
   cmd:option('-batchSize',     128,      'mini-batch size (1 = pure stochastic)')                             -- speed relevant
   cmd:option('-alpha',         0.1,      'learning rate decay rate')
   cmd:option('-momentum',      0.9,   'momentum')
   cmd:option('-weightDecay',   1e-4,  'weight decay')
   
   cmd:option('-lrDecayMethod', 'exp',   'step or exp decay')
   cmd:option('-lrBase',        0.5,      'initial learning rate')
   cmd:option('-lrEnd',         0.001,      'the final learning rate when using exponential decay')
   cmd:option('-nEpoch',        160,       'Number of total epochs to run')                                    -- speed relevant
   cmd:option('-nLRDecayStage', 80,      'the number of training stages with different learing rates')
   cmd:option('-svBFlag',       'true',  'flag to do singular value bounding')                                 -- speed relevant
   cmd:option('-svBFactor',     1.5,  'singular value bounding factor applied to weight matrices of FC/conv layers')
   cmd:option('-svBIter',       391,   'perform singular value bounding per svbIter iterations')               -- speed relevant
   cmd:option('-bnsBFlag',      'true', 'flag to do scaling bounding of BN layers')
   cmd:option('-bnsBFactor',    2,  'bounding factor for scaling parameter of BN layer')
   cmd:option('-bnsBType',      'rel',  'relative (rel) or BBN ways for bounding scaling parameter of BN layer')  
   
   ---------- Model options ----------------------------------
   cmd:option('-netType',       'PreActResNet',    'PreActResNet or other types of networks')
   cmd:option('-ensembleID',    0,                 'the IDs of the same network architecture' )
   cmd:option('-kWRN',          1,                 'the k value in Wide ResNet, k =1 for ResNet')
   cmd:option('-nBaseRecur',    9,       '1 | 3 | 5 | 7 ...')
   cmd:option('-BN',            'true',   'true to use batch normalization before or after conv. layer')
   cmd:option('-paramInitMethod', 'orthogonal',   'xavierimproved | orthogonal')
   cmd:text()
   
   local opts = cmd:parse(arg or {}) -- take tuning arguments from command line
   
   -- converting the command line arguments type of string to boolean
   opts.shareGradInput = opts.shareGradInput ~= 'false' 
   opts.optnet = opts.optnet ~= 'false'
   opts.testFlag = opts.testFlag ~= 'false'
   opts.BN = opts.BN ~= 'false'
   opts.svBFlag = opts.svBFlag ~= 'false'
   opts.bnsBFlag = opts.bnsBFlag ~= 'false'
   
   if opts.dataset == 'cifar10' then
   elseif opts.dataset == 'ImageNet' then
   else
      cmd:error('The specified dataset is not supported!')
   end
   
   if opts.shareGradInput and opts.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end
   
   return opts
end

return M