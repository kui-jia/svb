--[[

Multi-threaded data loader to make loading of large-size images efficient 

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding", CVPR 2017.
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]

local imdbTrnVal = require 'cifar10Init'
local threads = require 'threads'
threads.serialization('threads.sharedserialize') -- Specify which serialization scheme should be used

local M = {}
local dataLoader = torch.class('dataLoader', M)

function dataLoader.create(opts)
	local trnValDataLoaders = {}
	for iKey, value in ipairs({'train', 'val'}) do
	    local imdb = imdbTrnVal.create(opts, value)
		trnValDataLoaders[iKey] = M.dataLoader(imdb, opts, value)
	end
	
    return table.unpack(trnValDataLoaders)
end


function dataLoader:__init(imdb, opts, trnValSplit)
    --[[ 
	imdb contains data and info. for either 'train' or 'val' samples
	trnValSplit = 'train' or 'val'
	--]]
    local manualSeed = opts.manualSeed 
    local function init()
     	require(opts.dataset .. 'Init') 
	end
	local function main(threadid) 
	    torch.manualSeed(manualSeed + threadid) 
		torch.setnumthreads(1) 
		_G.imdb = imdb 
		_G.preprocess = imdb:preprocess() 
		return imdb:size() 
	end
	
	-- initialize a pool of threads
	local threadPool, nImgInIMDBTable = threads.Threads(opts.nThreads, init, main) 
	self.nCrops = 1 
	self.threadPool = threadPool
	self.nImgInIMDB = nImgInIMDBTable[1][1] 
	self.batchSize = math.floor(opts.batchSize / self.nCrops) 
end

function dataLoader:epochIterNum() 
    return math.ceil(self.nImgInIMDB / self.batchSize) 
end

function dataLoader:run() -- callback function for data loading during training/inference
    local threadPool = self.threadPool 
	local nImgInIMDB, batchSize = self.nImgInIMDB, self.batchSize
	local tmpindices = torch.randperm(nImgInIMDB) 
	
	local batchImgSamples 
	local idx = 1 
    local iter = 0 
	
	local function enqueue()
	    -- distributing the jobs of loading and pre-processing an epoch of mini-batches of image samples over a pool of threads 
        while idx <= nImgInIMDB and threadPool:acceptsjob()	do -- acceptsjob() return true if the pool of thread queues is not full
		    local tmpbatchindices = tmpindices:narrow(1, idx, math.min(batchSize, nImgInIMDB-idx+1))
            -- distributing the following jobs of mini-batches to the pool of threads
			threadPool:addjob(
			    function(tmpindices, nCrops) -- callback function, executed on each threads
                    local nImgSample = tmpindices:size(1)	
                    local target = torch.IntTensor(nImgSample) -- variable for hosting training targets/labels of image samples
                    local batchImgData, tmpsizes
                    for iKey, idxValue in ipairs(tmpindices:totable()) do
					    local currImgSample = _G.imdb:get(idxValue) 
						local currInput = _G.preprocess(currImgSample.input) -- do data augmentation on the fly
						if not batchImgData then
						    tmpsizes = currInput:size():totable() 
							batchImgData = torch.FloatTensor(nImgSample, nCrops, table.unpack(tmpsizes))
						end
						batchImgData[iKey]:copy(currInput)
						target[iKey] = currImgSample.target
                    end
                    collectgarbage() -- automtic management/freeing of garbage memory oppupied by the preceding operations 
                    return {input = batchImgData:view(nImgSample*nCrops, table.unpack(tmpsizes)), target = target}					
                end, 
				
                function(_batchImgSamples_) -- endcallback function whose argument is from the return of callback function, executed on the main thread, 
                    batchImgSamples = _batchImgSamples_ -- pass the mini-batch of image samples to the main thread
				end, 
				
				tmpbatchindices, -- arguments of callback function  
				self.nCrops
			)
			
            idx = idx + batchSize			
        end		
	end
	
	
	local function loop()
	    enqueue() -- loading and processing a mini-batch of image samples over a free thread 
		if not threadPool:hasjob() then -- true if there is still any job unfinished
		    return nil -- finish the 'loop' function when all jobs are done  
		end 
		
		threadPool:dojob() -- to tell the main thread to execute the next endcallback in the queue
		if threadPool:haserror() then
		    threadPool:synchronize()
		end
		
		enqueue() 
		iter = iter + 1
		
		return iter, batchImgSamples 
	end
	
	return loop
end

return M.dataLoader