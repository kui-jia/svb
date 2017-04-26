--[[

Pre-activation version of ResNet

An implementation of the ResNet architectures described in the CVPR2017 paper 
Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding"
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local cfg = require('Utils/' .. 'configUtils')

local spatialConv = cudnn.SpatialConvolution
local spatialAvgPool = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local spatialBN = cudnn.SpatialBatchNormalization
local BN = cudnn.BatchNormalization

local linear = nn.Linear
local crossEntropyCriterion = nn.CrossEntropyCriterion

local nnView = nn.View

local convBlock = cfg.convBlock
local resUnit = cfg.resUnit
local initFcConvLayer = cfg.initFcConvLayer
local initBNLayer = cfg.initBNLayer

local M = {}

function M.netInit(opts, checkpoint)
    local net
	 
	-- Loading checkpoints or initialize a new network to train 
    if checkpoint then
	    local modelPath = paths.concat(opts.expFolder, checkpoint.modelFName)
		assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
		net = torch.load(modelPath):cuda()
	else
	    -- Initialize the network architecture --  
        local nV1Recur = opts.nBaseRecur
        local nV2Recur = opts.nBaseRecur
        local nV3Recur = opts.nBaseRecur
	
	    net = nn.Sequential() 
	
	    -- Stem 
	    net:add(convBlock({3, 16, 3, 3}, {1, 1}, {1, 1}, opts.BN)) -- grid size of 32 x 32
		if opts.kWRN > 1 then
		    net:add(convBlock({16, torch.ceil(16*opts.kWRN/2), 3, 3}, {1, 1}, {1, 1}, opts.BN))
		    net:add(convBlock({torch.ceil(16*opts.kWRN/2), 16*opts.kWRN, 3, 3}, {1, 1}, {1, 1}, opts.BN))
		end														   
	
	    -- V1 
	    for idx = 1, nV1Recur do
	        net:add(resUnit(16*opts.kWRN, 16*opts.kWRN, '', '', '', opts.BN, 'PreActOrig'))
        end	
	
	    -- V2
	    net:add(resUnit(16*opts.kWRN, 32*opts.kWRN, '', '', '', opts.BN, 'PreActOrig')) -- grid size 16 x 16
	    for idx = 2, nV2Recur do
	        net:add(resUnit(32*opts.kWRN, 32*opts.kWRN, '', '', '', opts.BN, 'PreActOrig'))
	    end
	
	    -- V3
	    net:add(resUnit(32*opts.kWRN, 64*opts.kWRN, '', '', '', opts.BN, 'PreActOrig')) -- grid size 8 x 8
	    for idx = 2, nV3Recur do
	        net:add(resUnit(64*opts.kWRN, 64*opts.kWRN, '', '', '', opts.BN, 'PreActOrig'))
	    end
	
    	-- additional BN and ReLU after the last ResUnit
	    net:add(spatialBN(64*opts.kWRN)) 
	    net:add(ReLU(true)) 

        -- average pooling
        net:add(spatialAvgPool(8, 8, 1, 1, 0, 0)) 
	
    	-- fc
	    net:add(nnView(64*opts.kWRN):setNumInputDims(3)) 
	    net:add(linear(64*opts.kWRN, 10))	

	    -- Initialize parameters of conv., linear, and BN layers --	
		for _, moduleTypeName in ipairs{'nn.SpatialConvolution', 'cunn.SpatialConvolution', 'cudnn.SpatialConvolution', 'fbnn.SpatialConvolution'} do
        	initFcConvLayer(net, moduleTypeName, opts.paramInitMethod)
		end
		for _, moduleTypeName in ipairs{'nn.Linear', 'cunn.Linear', 'cudnn.Linear', 'fbnn.Linear'} do
	        initFcConvLayer(net, moduleTypeName, opts.paramInitMethod)
		end
		for _, moduleTypeName in ipairs{'nn.SpatialBatchNormalization', 'cunn.SpatialBatchNormalization', 'cudnn.SpatialBatchNormalization', 'fbnn.SpatialBatchNormalization'} do 
	        initBNLayer(net, moduleTypeName)
		end
		for _, moduleTypeName in ipairs{'nn.BatchNormalization', 'cunn.BatchNormalization', 'cudnn.BatchNormalization', 'fbnn.BatchNormalization'} do
    	    initBNLayer(net, moduleTypeName)
		end
	
	    net.gradInput = nil    
		
		-- Push the network into GPU --
    	net:cuda()
    end
    
	-- remove the DataParallelTable for model replica, if any
    if torch.type(net) == 'nn.DataParallelTable' then
      net = net:get(1) 
    end
   
    if opts.optnet then
       local optnet = require 'optnet'
       local tmpsize = opts.dataset == 'imagenet' and 224 or opts.dataset == 'cifar10' and 32
       local tmpInput = torch.zeros(4,3,tmpsize,tmpsize):cuda() 
       optnet.optimizeMemory(net, tmpInput, {inplace = false, mode = 'training'})
    end
   
    if opts.shareGradInput then
       M.shareGradInput(net)
    end
   
    -- Set the CUDNN flags
    if opts.cudnnSetting == 'fastest' then
       cudnn.fastest = true
       cudnn.benchmark = true
    elseif opts.cudnnSetting == 'deterministic' then
       net:apply(function(m) 
                     if m.setMode then m:setMode(1, 1, 1) end
                 end) 
    end
   
    -- Wrap the network with DataParallelTable, if using more than one GPUs
    if opts.nGPU > 1 then
         local gpuIDs = torch.range(opts.gpuStartID, opts.gpuStartID+opts.nGPU-1):totable()
	     local fastest, benchmark = cudnn.fastest, cudnn.benchmark
	   
	     local netGPUReplicas = nn.DataParallelTable(1, true, true) 
		 netGPUReplicas:add(net, gpuIDs)	
       	 netGPUReplicas:threads(function() 
		                            local cudnn = require 'cudnn'
									cudnn.fastest, cudnn.benchmark = fastest, benchmark
								end) 
		 netGPUReplicas.gradInput = nil 					
		 
		 net = netGPUReplicas:cuda() -- push into GPUs		 
    end
	
	-- set up the training criterion module
	local criterion = crossEntropyCriterion() 
	criterion:cuda() -- push into GPU
	
    return net, criterion
end


function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end


return M
