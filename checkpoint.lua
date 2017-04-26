--[[

Checkpoint code 

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding", CVPR 2017.
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local M = {}

local function netRecursiveCopy(net)
    local copyNet = {}
    for iKey, moduleVal in pairs(net) do
        if type(moduleVal) == 'table' then
		    copyNet[iKey] = netRecursiveCopy(moduleVal)
		else
		    copyNet[iKey] = moduleVal 
		end
	end    
	
	if torch.typename(net) then
        torch.setmetatable(copyNet, torch.typename(net))
    end
    return copyNet
end

function M.latest(fNamePrefix, opts)
    local latestPath = paths.concat(opts.expFolder, 'latest_' .. fNamePrefix .. '.t7')
    if not paths.filep(latestPath) then
        return nil
    end
	
	-- loading from the latest or the specified checkpoint
	local latest, optimState
	if opts.contPoint > 0 then
	    print('=> Loading checkpoint ' .. opts.contPoint)
		local modelFName = 'net_' .. fNamePrefix .. '_' .. opts.contPoint .. '.t7'
	    local optimFName = 'optState_' .. fNamePrefix .. '_' .. opts.contPoint .. '.t7'
		latest = {epoch = opts.contPoint, modelFName = modelFName, optimFName = optimFName}
		optimState = torch.load(paths.concat(opts.expFolder, latest.optimFName))
	else
	    print('=> Loading checkpoint ' .. latestPath)
        latest = torch.load(latestPath)
	    optimState = torch.load(paths.concat(opts.expFolder, latest.optimFName))
	end
	
	return latest, optimState
end

function M.best(fNamePrefix, opts)
    local bestPath = paths.concat(opts.expFolder, 'best_' .. fNamePrefix .. '.t7')
    if not paths.filep(bestPath) then
        return nil
    end
	
	-- loading from the bestpoint
	print('=> Loading bestpoint ' .. bestPath)
    local best = torch.load(bestPath)
	
	return best
end

function M.save(net, optimState, epoch, bestModelFlag, fNamePrefix, opts)
    -- don't save the DataParallelTable for easier loading on other machines
    if torch.type(net) == 'nn.DataParallelTable' then
        net = net:get(1)
    end
	
	-- create a clean copy on the CPU without modifying the original network
    net = netRecursiveCopy(net):float():clearState() -- Clears intermediate module states such as output, gradInput and others
	
	local modelFName = 'net_' .. fNamePrefix .. '_' .. epoch .. '.t7'
	local optimFName = 'optState_' .. fNamePrefix .. '_' .. epoch .. '.t7'
	
	torch.save(paths.concat(opts.expFolder, modelFName), net)
	torch.save(paths.concat(opts.expFolder, optimFName), optimState)
	
	-- for latest checking
	torch.save(paths.concat(opts.expFolder, 'latest_' .. fNamePrefix .. '.t7'), {epoch = epoch, modelFName = modelFName, optimFName = optimFName})
	
    if bestModelFlag then -- bestModelFlag is specified when calling this save function 
	    local bestModelFName = 'net_' .. fNamePrefix .. '_best.t7'
        torch.save(paths.concat(opts.expFolder, bestModelFName), net)
	    torch.save(paths.concat(opts.expFolder, 'best_' .. fNamePrefix .. '.t7'), {epoch = epoch, modelFName = bestModelFName}) -- for testing using the best model
    end
   
end

return M