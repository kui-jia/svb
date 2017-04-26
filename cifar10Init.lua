--[[

This file loads and pre-processes cifar10 data 

Kui Jia, Dacheng Tao, Shenghua Gao, and Xiangmin Xu, "Improving training of deep neural networks via Singular Value Bounding", CVPR 2017.
http://www.aperture-lab.net/research/svb

This code is based on the fb.resnet.torch package (https://github.com/facebook/fb.resnet.torch)
Copyright (c) 2016, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the 
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

--]]



local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
-- color statistics from the entire CIFAR-10 training set
local meanstd = { mean = {125.3, 123.0, 113.9}, std  = {63.0,  62.1,  66.7} }

local t = require('Utils/' .. 'transforms')
local utils = require('Utils/' .. 'utilFuncs')

local M = {}
local imdb = torch.class('imdb', M)

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- The downloaded files have labels 0-9, which do not work with CrossEntropyCriterion
   labels:add(1)

   return {data = data:contiguous():view(-1, 3, 32, 32), labels = labels} 
end

local function loadingRawData(fileName)
    print('=> Downloading CIFAR-10 dataset from ' .. URL)
	local ok = os.execute('curl ' .. URL .. ' | tar xz -C Data/cifar10-raw/')
	-- local ok = os.execute('tar xz -C Data/cifar10-raw/')
	assert(ok == true or ok == 0, 'error downloading CIFAR-10')
	
	print(" | combining dataset into a single file")
    local trnData = convertToTensor( {
       'Data/cifar10-raw/cifar-10-batches-t7/data_batch_1.t7',
       'Data/cifar10-raw/cifar-10-batches-t7/data_batch_2.t7',
       'Data/cifar10-raw/cifar-10-batches-t7/data_batch_3.t7',
       'Data/cifar10-raw/cifar-10-batches-t7/data_batch_4.t7',
       'Data/cifar10-raw/cifar-10-batches-t7/data_batch_5.t7',
    } )
    local testData = convertToTensor( {
       'Data/cifar10-raw/cifar-10-batches-t7/test_batch.t7',
    } )
	
	print(' | saving CIFAR-10 dataset to ' .. fileName)
    torch.save(fileName, {train = trnData, val = testData})
end


function imdb.create(opts, trnValSplit) -- will be called when building up multi-threaded dataLoader
    local tmpFileName = paths.concat('Data', 'cifar10.t7')
	if not paths.filep(tmpFileName) then
		loadingRawData(tmpFileName) -- downloading the data and combining and saving as a single file 'cifar10.t7'
	end
	local images = torch.load(tmpFileName) -- loading the train and val data 

	local imdbTrnVal = M.imdb(images, opts, trnValSplit) -- returning imdb class instance for either 'train' or 'val' data
	return imdbTrnVal
end


function imdb:__init(images, opts, trnValSplit)
    assert(images[trnValSplit], trnValSplit)
    self.images = images[trnValSplit]
    self.trnValSplit = trnValSplit
end

function imdb:get(i)
    local img = self.images.data[i]:float()
    local label = self.images.labels[i]

    return {input = img, target = label}
end

function imdb:size()
   return self.images.data:size(1) -- the number of images
end

function imdb:preprocess() 
   
   if self.trnValSplit == 'train' then
      return t.Compose{
             -- t.ColorNormalize(meanstd), 
             t.HorizontalFlip(0.5),
             t.RandomCrop(32, 4),
             }
   elseif self.trnValSplit == 'val' then
      return t.IdentityMap() -- do nothing transformation
	  -- return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.trnValSplit)
   end
end


return M.imdb