--[[

The entry code, change parameter settings in 'optsArgParse.lua' 

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding", CVPR 2017.
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local initTimer = torch.Timer()

initTimer:reset()		
require 'torch'
require 'cutorch'
require 'paths'
require 'optim'
require 'nn'
print(('The time of requring torch packages is: %.3f'):format(initTimer:time().real))	  

local optsArgParse = require 'optsArgParse'
local opts = optsArgParse.parse(arg)
print(opts)

initTimer:reset()
local dataLoader = require 'dataLoader' -- cifar10Init is called inside
local netBuilder = require(opts.dataset .. '_' .. opts.netType)
local cnnTrainer = require 'cnnTrain'
local checkpoint = require 'checkpoint'
local utils = require('Utils/' .. 'utilFuncs')
print(('The time of requring project packages is: %.3f'):format(initTimer:time().real))	
----------------------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1) 
torch.manualSeed(opts.manualSeed)
cutorch.manualSeedAll(opts.manualSeed)

local prefix
if opts.netType == 'PreActResNet' then
    if opts.lrDecayMethod == 'exp' then
        prefix = opts.netType .. opts.ensembleID .. '_nRecur' .. opts.nBaseRecur .. '_kWRN' .. opts.kWRN .. '_BN_' .. tostring(opts.BN) ..
               '_lr_' .. opts.lrDecayMethod .. '_base' .. opts.lrBase .. '_end' .. opts.lrEnd .. '_wDecay' .. opts.weightDecay ..
			   '_batch' .. opts.batchSize .. '_nEpoch' .. opts.nEpoch .. '_nStage' .. opts.nLRDecayStage .. 
			   '_svBFlag_' .. tostring(opts.svBFlag) .. '_factor' .. opts.svBFactor .. '_iter' .. opts.svBIter .. 
			   '_bnsBFlag_' .. tostring(opts.bnsBFlag) .. '_factor' .. opts.bnsBFactor .. '_' .. opts.bnsBType
    end	
end
print(prefix)			   

-- creating callback multi-threaded data loading functions
initTimer:reset()
local trnLoader, valLoader = dataLoader.create(opts)
print(('The time of creating dataLoader is: %.3f'):format(initTimer:time().real))	

-- loading the latest training checkpoint if it exists
initTimer:reset()
local latestpoint, optimState = checkpoint.latest(prefix, opts) -- returning nil if not existing 
print(('The time of loading latest checkpoint is: %.3f'):format(initTimer:time().real))	

-- loading the latest or create a new network model
initTimer:reset()
local net, criterion = netBuilder.netInit(opts, latestpoint) 
print(('The time of initializing network model is: %.3f'):format(initTimer:time().real))	

-- initialize the trainer, which handles training loop and evaluation on the validation set
initTimer:reset()
local trainer = cnnTrainer(net, criterion, optimState, opts) 
print(('The time of initializing the trainer is: %.3f'):format(initTimer:time().real))

-- do testing only if true
if opts.testFlag then
    local bestpoint = checkpoint.best(prefix, opts) -- returning nil if not existing 
    net, criterion = netBuilder.netInit(opts, bestpoint) -- loading the best model
    trainer = cnnTrainer(net, criterion, nil, opts) -- initialize the trainer with the best model
    local top1Err, top5Err = trainer:test(0, valLoader)
    print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
    return
end

-- start training 
local startEpoch = latestpoint and latestpoint.epoch + 1 or opts.startEpoch
local nTrnIterPerEpoch = trnLoader:epochIterNum()
local bestTop1Err = math.huge
local bestTop5Err = math.huge
local trnTimer = torch.Timer()
local testTimer = torch.Timer()
local svbTimer = torch.Timer()
local statsFPath = paths.concat(opts.expFolder, 'stats_' .. prefix .. '.txt')
if not paths.filep(statsFPath) then
	local statsFile = io.open(statsFPath, 'w') -- create a new one
	statsFile:close()
end	
for epoch = startEpoch, opts.nEpoch do
    -- Train for a single epoch
	trnTimer:reset()		
    local iter = (epoch-1) * nTrnIterPerEpoch -- the total number of iterations trained so far
    local trnTop1Err, trnTop5Err, trnLoss = trainer:epochTrain(epoch, iter, trnLoader)
	print(('| Training | Epoch: [%d]   Time %.3f   top1 %7.3f   top5 %7.3f   Loss %1.4f'):format(
                         epoch, trnTimer:time().real, trnTop1Err, trnTop5Err, trnLoss))	  
	utils.writeErrsToFile(statsFPath, epoch, trnTop1Err, trnTop5Err, 'train')
	
    -- Run model on validation set
	testTimer:reset()
    local testTop1Err, testTop5Err = trainer:test(epoch, valLoader)
	print(('                                                                | Testing | Epoch [%d]    Time %.3f   top1 %7.3f   top5 %7.3f'):format(
                      epoch, testTimer:time().real, testTop1Err, testTop5Err))	
    utils.writeErrsToFile(statsFPath, epoch, testTop1Err, testTop5Err, 'val')					  
	
	-- Save the best model and other statistics/results 
    local bestModelFlag = false
    if testTop1Err < bestTop1Err then
       bestModelFlag = true -- true to save the up to now best model
       bestTop1Err = testTop1Err
       bestTop5Err = testTop5Err
       print(' * Best model ', testTop1Err, testTop5Err)
	   utils.writeErrsToFile(statsFPath, '', testTop1Err, testTop5Err, 'best')		
    end

    checkpoint.save(net, trainer.optimState, epoch, bestModelFlag, prefix, opts)
	
	-- Applying Singular Value Bounding (and Bounded Batch Normalization) to (conv and fc) layer weights 
	-- at the end of each epoch, but not for the last epoch
	svbTimer:reset()
	if opts.svBFlag and epoch ~= opts.nEpoch then
	    trainer:fcConvWeightReguViaSVB()
		if opts.bnsBFlag then -- optionally do scaling bounding of BN layers
			trainer:BNScalingRegu()
		end
	end
	print(('The time of SVB operation at the end of each epoch: %.3f'):format(svbTimer:time().real))	  
end
trnTimer:reset()	
testTimer:reset()
svbTimer:reset()

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1Err, bestTop5Err))
utils.writeErrsToFile(statsFPath, '', bestTop1Err, bestTop5Err, 'final')		

