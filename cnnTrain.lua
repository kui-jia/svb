--[[

SGD based CNN Training

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding", CVPR 2017.
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local optim = require 'optim'
local utils = require('Utils/' .. 'utilFuncs')

local M = {}
local cnnTrain = torch.class('cnnTrain', M)

function cnnTrain:__init(net, criterion, optimState, opts)
    self.net = net
    self.criterion = criterion
    self.optimState = optimState or {
	    learningRate = opts.lrBase,
		momentum = opts.momentum,
		weightDecay = opts.weightDecay,
		nesterov = true, -- enables Nesterov momentum, Nesterov momentum requires a momentum and zero dampening
		learningRateDecay = 0.0,
		dampening = 0.0
	    }    
	self.opts = opts
	self.params, self.gradParams = net:getParameters() -- returning concatenated vectors of parameters and gradients 
		                                               -- should only be called once for a network since storage may change  
end

function cnnTrain:epochTrain(epoch, startIter, dataLoader) -- training the network for one epoch
    -- setting learning rate for the present epoch
	self.optimState.learningRate = self:learningRateSchedule(epoch, self.opts.nEpoch)
	
	local timer = torch.Timer()
    local dataTimer = torch.Timer()
	
	local function feval() -- function handler to be called by optimizer, upon value instantiation of self.gradParams and self.criterion
        return self.criterion.output, self.gradParams
    end
	
	local nEpochIter = dataLoader:epochIterNum() -- no. of interations in this epoch 
    local top1Err, top5Err, loss = 0.0, 0.0, 0.0
	local nSample = 0 
   
    self.net:training() -- setting mode of modules/sub-modules to training mode (train = true), useful for Dropout or BatchNormalization
	
	for iKey, batchSample in dataLoader:run() do 
		
	    local dataTime = dataTimer:time().real
		
        self:copyBatchSample2GPU(batchSample) -- Copy input and target to the GPU
		
		-- forward pass
		local output = self.net:forward(self.input):float() 													
        local batchSize = output:size(1)
        local currLoss = self.criterion:forward(self.net.output, self.target)
	  
	    self.net:zeroGradParameters() 
		
		-- backward pass
        self.criterion:backward(self.net.output, self.target)
        self.net:backward(self.input, self.criterion.gradInput) -- this function does backward pass by calling updateGradInput and accGradParameters
	  
	    -- updating parameters via SGD with momentum
		optim.sgd(feval, self.params, self.optimState) -- returning the updated self.params
		                                               -- self.params, self.gradParams collectively store the network parameters and their gradients
		
		-- Applying Singular Value Bounding and (Bounded Batch Normalization) to (conv and fc) layer weights 
		-- at certain iterations, but not at the last epoch of iterations, and not when the current epoch of training appoaches ending. 
		if self.opts.svBFlag and math.fmod(iKey, self.opts.svBIter) == 0 
		                     and nEpochIter - iKey >= 100 and epoch ~= self.opts.nEpoch then
		    self:fcConvWeightReguViaSVB()
			if self.opts.bnsBFlag then -- optionally do scaling bounding of BN layers
			    self:BNScalingRegu()
			end
		end		
		
		-- reporting errors (based on network parameters of the previous iteration) and other learning statistics
		local currTop1Err, currTop5Err = self:computeErrors(output, batchSample.target, 1) 
		top1Err = top1Err + currTop1Err*batchSize
        top5Err = top5Err + currTop5Err*batchSize
        loss = loss + currLoss*batchSize
        nSample = nSample + batchSize
						 
        print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.4f  top1 %7.3f  top5 %7.3f'):format(
                         epoch, iKey, nEpochIter, timer:time().real, dataTime, currLoss, currTop1Err, currTop5Err))		
						 
		-- check that the storage didn't get changed due to an unfortunate getParameters call
        assert(self.params:storage() == self.net:parameters()[1]:storage())		

        timer:reset()
        dataTimer:reset()		
	end
	
	return top1Err / nSample, top5Err / nSample, loss / nSample
end


function cnnTrain:test(epoch, dataLoader) 
    -- computing the top-1 and top-5 errors on the validation set
	local timer = torch.Timer()
    local dataTimer = torch.Timer()
    
	local nEpochIter = dataLoader:epochIterNum() -- no. of interations
	local nCrops = 1
    local top1Err, top5Err = 0.0, 0.0
    local nSample = 0
	
	self.net:evaluate() -- setting mode of modules/sub-modules to evaluation mode, useful for Dropout or BatchNormalization
	
	for iKey, batchSample in dataLoader:run() do 
		local dataTime = dataTimer:time().real
		
		self:copyBatchSample2GPU(batchSample) -- Copy input and target to the GPU
		
		-- forward pass
		local output = self.net:forward(self.input):float() 
        local batchSize = output:size(1) / nCrops -- 'batchSize' becomes in terms of images/samples
        -- local currLoss = self.criterion:forward(self.net.output, self.target)
		
		local currTop1Err, currTop5Err = self:computeErrors(output, batchSample.target, nCrops) 
        top1Err = top1Err + currTop1Err*batchSize
        top5Err = top5Err + currTop5Err*batchSize
        nSample = nSample + batchSize
		
		-- print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
        --               epoch, iKey, nEpochIter, timer:time().real, dataTime, currTop1Err, top1Err/nSample, currTop5Err, top5Err/nSample))

        timer:reset()
        dataTimer:reset()
	end
	
	self.net:training() -- resetting the mode of modules/sub-modules to training mode
	
	return top1Err/nSample, top5Err/nSample
end


function cnnTrain:computeErrors(output, target, nCrops)
    -- computation of errors are detached from the network
    -- running on CPU	
	if nCrops > 1 then
        -- sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2) 
    end

    -- computing the top1 and top5 error rate
    local batchSize = output:size(1)

    local _ , predictions = output:float():sort(2, true) -- descending

    -- finding which predictions match the target
    local correct = predictions:eq(target:long():view(batchSize, 1):expandAs(output))

    -- Top-1 error
    local top1Err = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

    -- Top-5 error, if there are at least 5 classes
    local len = math.min(5, correct:size(2))
    local top5Err = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

    return top1Err * 100, top5Err * 100
end


function cnnTrain:copyBatchSample2GPU(batchSample)
    self.input = self.input or ( self.opts.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor() )
    self.target = self.target or torch.CudaTensor()
	
    self.input:resize(batchSample.input:size()):copy(batchSample.input)
    self.target:resize(batchSample.target:size()):copy(batchSample.target)
end

local function kron(X, Z)
    assert(X:dim() <= 2 and Z:dim() <= 2) -- should generalize this
	local N, M, P, Q
	if #X:size() > 1 then
	    N = X:size(1)
		M = X:size(2)
	else
	    N = X:size(1)
		M = 1
	end
	if #Z:size() > 1 then
        P = Z:size(1) 
		Q = Z:size(2)
	else
	    P = Z:size(1)
		Q = 1
	end
    local K = torch.Tensor(N*P, M*Q)
	if #X:size() > 1 then
        for row = 1,N do
            for col = 1,M do
                K[{{(row - 1)*P + 1, row*P},{(col - 1)*Q + 1, col*Q}}] = torch.mul(Z, X[row][col])
            end
        end
	else
	    for row = 1,N do
            K[{{(row - 1)*P + 1, row*P},{1, Q}}] = torch.mul(Z, X[row])
        end
	end
    return K
end

function cnnTrain:learningRateSchedule(epoch, nTotalEpoch)
    local decay = 0
	local learningRate = self.opts.lrBase
   
    if self.opts.lrDecayMethod == 'step' then 
	    -- step decay
		if nTotalEpoch == 160 then
            decay = epoch >= 121 and 2 or epoch >= 81 and 1 or 0
		elseif nTotalEpoch == 300 then
		    decay = epoch >= 226 and 2 or epoch >= 151 and 1 or 0
		end
		learningRate = learningRate * math.pow(self.opts.alpha, decay)
	elseif self.opts.lrDecayMethod == 'exp' then
	    -- exponential decay
		local lrSeries = kron(torch.logspace(math.log(self.opts.lrBase, 10), math.log(self.opts.lrEnd, 10), self.opts.nLRDecayStage), 
		                                                                          torch.ones(self.opts.nEpoch/self.opts.nLRDecayStage, 1))																				  
		learningRate = lrSeries[epoch]:squeeze() -- squeeze as a torch type of 'number'		
	end
   
   return learningRate
end

function cnnTrain:fcConvWeightReguViaSVB()
    -- regularizaing weight matrices via SVD on CPU and then copying back onto GPU 
    local tmpU, tmpS, tmpV
	
    -- applying to weights of fc layers
	for _, moduleTypeName in pairs{'nn.Linear', 'cudnn.Linear'} do
        for iKey, moduleVal in pairs(self.net:findModules(moduleTypeName)) do
		    tmpU, tmpS, tmpV = torch.svd(moduleVal.weight:t():float()) -- weight of fc layer is of the size nFeaOut x nFeaIn
		                                                               -- tmpS is a vector containing singular values													

			for idx = 1, tmpS:size(1), 1 do
	            if tmpS[idx] > self.opts.svBFactor then
                    tmpS[idx] = self.opts.svBFactor
                elseif tmpS[idx] < 1/self.opts.svBFactor then
                    tmpS[idx] = 1/self.opts.svBFactor
                end
	        end
			
		    moduleVal.weight:copy((tmpU*torch.diag(tmpS)*tmpV:t()):t())
	    end
	end	
	
	-- applying to weights of conv. layers 
	for _, moduleTypeName in pairs{'nn.SpatialConvolution', 'cudnn.SpatialConvolution'} do
	    for iKey, moduleVal in pairs(self.net:findModules(moduleTypeName)) do
		    tmpU, tmpS, tmpV = torch.svd(torch.reshape(moduleVal.weight, moduleVal.nOutputPlane, 
		                                  moduleVal.nInputPlane*moduleVal.kH*moduleVal.kW):t():float()) -- weight of conv layer is of the size 
		                                                                                                -- nOutputPlane x nInputPlane x kH x kW												

			for idx = 1, tmpS:size(1), 1 do
	            if tmpS[idx] > self.opts.svBFactor then
                    tmpS[idx] = self.opts.svBFactor
                elseif tmpS[idx] < 1/self.opts.svBFactor then
                    tmpS[idx] = 1/self.opts.svBFactor
                end
	        end
			
		    moduleVal.weight:copy(torch.reshape((tmpU*torch.diag(tmpS)*tmpV:t()):t(), 
		                          moduleVal.nOutputPlane, moduleVal.nInputPlane, moduleVal.kH, moduleVal.kW))
	    end
	end
	
end

local function BNScalingBounding(vec, runningVar, epsilon, sFactor, bType)

    if bType == 'rel' then
	    local m = torch.mean(vec)
		local relVec = torch.div(vec, m)
		
		for idx = 1, vec:size(1), 1 do
            if relVec[idx] > sFactor then
                vec[idx] = m * sFactor
            elseif relVec[idx] < 1/sFactor then
                vec[idx] = m / sFactor
            end
        end
	elseif bType == 'BBN' then
		
		local runningStd = torch.sqrt(torch.add(runningVar, epsilon))
		local m = torch.mean(torch.cdiv(vec, runningStd)) -- mean of gamma / std 
		for idx = 1, vec:size(1), 1 do
		    if vec[idx]/(runningStd[idx]*m) > sFactor then
			    vec[idx] = runningStd[idx] * m * sFactor
			elseif vec[idx]/(runningStd[idx]*m) < 1/sFactor then
			    vec[idx] = (runningStd[idx] * m) / sFactor
			end
		end
    else 	
	    error('BNScalingBounding must be either rel or BBN!')  
	end
	
	return vec
end

function cnnTrain:BNScalingRegu()
    -- applying to layer of BatchNormalization
	for _, moduleTypeName in pairs{'nn.BatchNormalization', 'cudnn.BatchNormalization'} do
        for iKey, moduleVal in pairs(self.net:findModules(moduleTypeName)) do
		    moduleVal.weight:copy(BNScalingBounding(moduleVal.weight, moduleVal.running_var, moduleVal.eps, self.opts.bnsBFactor, self.opts.bnsBType))
	    end
	end
	
	-- applying to layer of SpatialBatchNormalization
	for _, moduleTypeName in pairs{'nn.SpatialBatchNormalization', 'cudnn.SpatialBatchNormalization'} do 
        for iKey, moduleVal in pairs(self.net:findModules(moduleTypeName)) do
		    moduleVal.weight:copy(BNScalingBounding(moduleVal.weight, moduleVal.running_var, moduleVal.eps, self.opts.bnsBFactor, self.opts.bnsBType))
	    end
	end    
end

return M.cnnTrain