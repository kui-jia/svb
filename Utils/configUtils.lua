--[[

Utility functions for configuration of network architecture and initialization of network parameters

Copyright (C) 2016 Kui Jia 
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'

local spatialConv = cudnn.SpatialConvolution
local spatialAvgPool = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local spatialBN = cudnn.SpatialBatchNormalization
local BN = cudnn.BatchNormalization

local linear = nn.Linear
local crossEntropyCriterion = nn.CrossEntropyCriterion
local identity = nn.Identity

local M = {}

local function ShareGradInput(module, key)
    assert(key)
    module.__shareGradInputKey = key
    return module
end

function M.convBlock(size, stride, pad, BNFlag)
    -- size, a table of {nInputPlane, nOutputPlane, kernelWidth, kernelHeight}
	-- stride, a table of {strideWidth, strideHeight}
	-- pad, a table of {padWidth, padHeight}
	local nFeaIn, nFeaOut, kerW, kerH = table.unpack(size)
	local strideW, strideH = table.unpack(stride)
	local padW, padH = table.unpack(pad)
	
    local blk = nn.Sequential()
	blk:add(spatialConv(nFeaIn, nFeaOut, kerW, kerH, strideW, strideH, padW, padH)) -- conv. layer
	if BNFlag then
	    blk:add(spatialBN(nFeaOut)) -- The default momentum (for running average of mean and std) in BN layer is 0.1 
	                                                  -- The default eps in BN layer is 1e-5    
	end
	blk:add(ReLU(true)) -- true for in-place operation 			
	
	return blk
end

function M.preActConvBlock(size, stride, pad, BNFlag, BNKey, nConvGroup)

	local nFeaIn, nFeaOut, kerW, kerH = table.unpack(size)
	local strideW, strideH = table.unpack(stride)
	local padW, padH = table.unpack(pad)
	
	local blk = nn.Sequential()
	if BNFlag then
	    if BNKey ~= '' then
		    blk:add(ShareGradInput(spatialBN(nFeaIn), BNKey))
		else
	        blk:add(spatialBN(nFeaIn)) -- The default momentum (for running average of mean and std) in BN layer is 0.1 
	                                                      -- The default eps in BN layer is 1e-5   
        end														  
	end
	blk:add(ReLU(true)) -- true for in-place operation

    if not nConvGroup then
	    nConvGroup = 1
    end 	
	blk:add(spatialConv(nFeaIn, nFeaOut, kerW, kerH, strideW, strideH, padW, padH, nConvGroup)) -- conv. layer
	
	return blk
end

function M.resUnit(nFeaIn, nFeaOut, downSamplingFlag, nGroup, groupWidth, BNFlag, unitType)
    local unit = nn.Sequential() 
	
    if unitType == 'PreActOrig' then
	    local resBlk = nn.Sequential() -- the residual block container
	    local scBlk = nn.Sequential() -- the (identity or projection) shortcut block container 
	    -- local scresBlk = nn.ConcatTable() -- the path concatenation container
	
	    if nFeaIn == nFeaOut then
	        -- identity shortcut when input and output dimensions (and also sizes of input/output feature maps) are the same
	        resBlk:add(M.preActConvBlock({nFeaIn, nFeaIn, 3, 3}, {1, 1}, {1, 1}, BNFlag, ''))  
		    resBlk:add(M.preActConvBlock({nFeaIn, nFeaIn, 3, 3}, {1, 1}, {1, 1}, BNFlag, '')) -- the 2nd one shares the same key with other gradInput storage
	    
		    scBlk:add(identity()) -- identity shortcut
    	else
	        -- optionally doing projection shortcut when input and output dimensions (and also sizes of input/output feature maps) mismatch
		    resBlk:add(M.preActConvBlock({nFeaIn, nFeaOut, 3, 3}, {2, 2}, {1, 1}, BNFlag, '')) 
		    resBlk:add(M.preActConvBlock({nFeaOut, nFeaOut, 3, 3}, {1, 1}, {1, 1}, BNFlag, '')) -- the 2nd one shares the same key with other gradInput storage
		
		    scBlk:add(M.preActConvBlock({nFeaIn, nFeaOut, 1, 1}, {2, 2}, {0, 0}, BNFlag, '')) -- the 1x1 convolution shortcut block                                                                                  -- try BNKey = 'scPreActBN' if error happens 
	    end
		
		unit:add(nn.ConcatTable():add(resBlk):add(scBlk)):add(nn.CAddTable(true))
	else
	    error('The specified resUnit type is not provided!')
	end
	
	return unit
end




function M.initFcConvLayer(net, convLayerType, initMethod)
    for iKey, moduleVal in pairs(net:findModules(convLayerType)) do
		if initMethod == 'xavierimproved' then
		    local std 
		    if convLayerType == 'nn.Linear' or convLayerType == 'cunn.Linear' or convLayerType == 'cudnn.Linear' or convLayerType == 'fbnn.Linear' then
				std = math.sqrt(2/(moduleVal.weight:size(1))) -- for fc layer, the weight matrix of size nFeaOut x nFeaIn
			else
     		    std = math.sqrt(2/(moduleVal.kW * moduleVal.kH * moduleVal.nOutputPlane))  
			end
            moduleVal.weight:normal(0, std)
			
			if cudnn.version >= 4000 then
                moduleVal.bias = nil
                moduleVal.gradBias = nil
            else
                moduleVal.bias:zero()
            end
		elseif initMethod == 'orthogonal' then
		    local tmpU, tmpS, tmpV
			local nFeaIn, nFeaOut, kerH, kerW
			
		    if convLayerType == 'nn.Linear' or convLayerType == 'cunn.Linear' or convLayerType == 'cudnn.Linear' or convLayerType == 'fbnn.Linear' then 
			    -- for fc layer, the weight matrix of size nFeaOut x nFeaIn
			    nFeaIn = moduleVal.weight:size(2) 
				nFeaOut = moduleVal.weight:size(1)
				kerH = 1
				kerW = 1
			else -- SpatialConvolution layer
			    if moduleVal.groups then -- true when using 'cudnn.SpatialConvolution'
				    nFeaIn = moduleVal.nInputPlane / moduleVal.groups
				else
				    nFeaIn = moduleVal.nInputPlane
				end
				nFeaOut = moduleVal.nOutputPlane
				kerH = moduleVal.kH
				kerW = moduleVal.kW
			end
			
			tmpU, tmpS, tmpV = torch.svd(torch.randn(kerH*kerW*nFeaIn, nFeaOut), 'A') -- tmpS is a vector containing singular values	
			if kerH*kerW*nFeaIn > nFeaOut then
                tmpS = torch.cat(torch.eye(nFeaOut), torch.zeros(kerH*kerW*nFeaIn-nFeaOut, nFeaOut), 1)
			elseif kerH*kerW*nFeaIn == nFeaOut then
                tmpS = torch.eye(nFeaOut)		
            else
                tmpS = torch.cat(torch.eye(kerH*kerW*nFeaIn), torch.zeros(kerH*kerW*nFeaIn, nFeaOut-kerH*kerW*nFeaIn), 2)
            end
			
			if convLayerType == 'nn.Linear' or convLayerType == 'cunn.Linear' or convLayerType == 'cudnn.Linear' or convLayerType == 'fbnn.Linear' then 
			    -- for fc layer, the weight matrix of size nFeaOut x nFeaIn
				moduleVal.weight:copy((tmpU*tmpS*tmpV:t()):t())
			else -- weight of conv layer is of the size nOutputPlane x nInputPlane x kH x kW
				moduleVal.weight:copy(torch.reshape((tmpU*tmpS*tmpV:t()):t(), nFeaOut, nFeaIn, kerH, kerW))
			end
			
			if cudnn.version >= 4000 then
                moduleVal.bias = nil
                moduleVal.gradBias = nil
            else
                moduleVal.bias:zero()
            end
		else
		    error('The weight initialization method does not exist!')	
		end 		
	end
end

function M.initBNLayer(net, BNLayerType)
    for iKey, moduleVal in pairs(net:findModules(BNLayerType)) do
	    moduleVal.weight:fill(1) -- 'scaling' is initialized as 1
		moduleVal.bias:zero() -- 'shift' is initialized as 0
	end
end

return M
