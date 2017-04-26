--[[

Utility functions

Copyright (C) 2016 Kui Jia 
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local M = {}

function M.dataPreProcessing_CIFAR(trnData, testData)
    -- trnData, testData is of the BypeTensor type
    -- trnData of the size 50000 x 3 x 32 x 32, testData of the size 10000 x 3 x 32 x 32
	
	trnData = trnData:float()
    testData = testData:float()
	  
	local numTrain, dim, height, width = trnData:size(1), trnData:size(2), trnData:size(3), trnData:size(4)
    local numTest = testData:size(1) 

    -- remove data mean (across samples)
	local dataMean = torch.mean(trnData, 1)
	
	trnData:add(-1, dataMean:expand(numTrain, dim, height, width))
	testData:add(-1, dataMean:expand(numTest, dim, height, width))
	
	-- normalize by image mean and std as suggested in "An Analysis of Single-Layer Networks in Unsupervised Feature Learning", Adam Coates, Honglak Lee, Andrew Y. Ng
	local contrastNormalizationFlag = true 
	if contrastNormalizationFlag then
	    local num, z, n, dim, data
		
		data = torch.cat(trnData, testData, 1)
		
		z = data:view(data:size(1),-1)
		z:add(-1, torch.mean(z, 2):expand(z:size(1), z:size(2)))
		num, dim = z:size(1), z:size(2)
		n = torch.std(z, 2):expand(num, dim)
		z:cmul(torch.cdiv(torch.mean(n, 1):expand(num, dim), torch.clamp(n, 40, 1e100))) 
		data = z:view(-1, 3, height, width)
		
		-- extract training and test data
		trnData =  data:sub(1,trnData:size(1), 1,3, 1,32, 1,32)
		testData = data:sub(trnData:size(1)+1,trnData:size(1)+testData:size(1), 1,3, 1,32, 1,32)
		
		data = nil
	end
	
	local whitenDataFlag = true
	if whitenDataFlag then
	    local z, num, dim, W, V, E, mid, en
	    
		z = trnData:view(trnData:size(1),-1)
		num, dim = z:size(1), z:size(2)
		W = z:t() * z / z:size(1)
		E, V = torch.symeig(W, 'V') -- vector E contains eigenvalues, and matrix V contains eigenvectors
		-- the scale is selected to approximately preserve the norm of W
		en = torch.Tensor(1,1):fill(torch.sqrt(torch.mean(E)))  
		en = en:float()
		mid = V * torch.diag(torch.squeeze(torch.cdiv(en:expand(E:size(1),1), torch.clamp(torch.sqrt(torch.abs(E)), 10, 1e100)))) * V:t()
        z = z * mid 
        trnData = z:view(-1, 3, height, width) --for trnData
		
        z = testData:view(testData:size(1),-1)
        z = z * mid 
        testData = z:view(-1, 3, height, width) --for testData
	end
	  
	return trnData, testData
end

function M.writeErrsToFile(fpath, epoch, top1Err, top5Err, mode)
    -- mode ~ 'train' | 'val' | 'best' | 'final'
	local file
	
    if paths.filep(fpath) then
	    file = io.open(fpath, 'a') -- append
	else
	    file = io.open(fpath, 'w') -- create a new one
    end	
	
	if mode == 'train' then
	    file:write(string.format('Training-Epoch:%d  top1:%7.3f  top5:%7.3f\n', epoch, top1Err, top5Err))	
	elseif mode == 'val' then
	    file:write(string.format('                                   Testing-Epoch:%d  top1:%7.3f  top5:%7.3f\n', epoch, top1Err, top5Err))					  
	elseif mode == 'best' then
	    file:write(string.format('                                                                             Best so far | top1:%7.3f  top5:%7.3f |\n', 
	                                                     top1Err, top5Err))
	elseif mode == 'final' then
	    file:write(string.format('----- The final best result | top1:%7.3f  top5:%7.3f | -----\n', top1Err, top5Err))
	else
	    error('The mode of writing erros to file must be either train, val, best, or final!')
	end
	
	file:close()
end


function M.writeSingularValueToFile(fpath, SVs)
    -- mode ~ 'train' | 'val' | 'best' | 'final'
	local file
	
    if paths.filep(fpath) then
	    file = io.open(fpath, 'a') -- append
	else
	    file = io.open(fpath, 'w') -- create a new one
    end	
	
	file:write(string.format('Layer: '))	
	for idx = 1, SVs:size(1), 1 do
	    file:write(string.format('%7.4f ', SVs[idx]))	
	end
	file:write(string.format('\n'))	
	
	file:close()
end

return M