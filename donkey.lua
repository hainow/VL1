--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'
require 'nn' 

print('calling dataset.lua')
paths.dofile('dataset.lua')
print('end dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "../cache"
os.execute('mkdir -p '..cache)
local trainCache = paths.concat(cache, 'trainCache_assignment2.t7')


-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '../logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}
local labelSampleSize = {3, opt.labelSize}

-- read the codebook (40 * 3)

local codebooktxt = '/usr0/home/htpham/Downloads/824/a1/data/list/codebook_40.txt'
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 

  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()
end


local div_num, sub_num
div_num = 127.5
sub_num = -1


local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize)
   input = input * 255
   return input
end


local function loadLabel_high(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.labelSize, opt.labelSize )
   input = input * 255
   return input
end


--*** WARNING: matrix implementation is **NOT** FASTER than loop implementation 
-- LUA really needs a library as numpy for python 


-- transpose codebook for matrix multiplication 
--codebook = codebook:transpose(1,2) 
function makeData_cls2(img, label)
    -- #label = opt.batchSize * 3 * opt.labelSize * opt.labelSize = N, 3, a a
    
    -- transpose (2, 3) and (3, 4) => N, a, a, 3
    --label = label:transpose(2, 3); label = label:transpose(3, 4);
    --label = label:reshape(opt.batchSize * opt.labelSize * opt.labelSize, 3) 
    
    label = nn.Transpose({2,3}, {3,4}):forward(label) 
    label = nn.View(opt.batchSize * opt.labelSize * opt.labelSize, 3):forward(label)

    -- dot product: change 4D tendor to 2D tensor to do dot product 
    local dot_prod = label * codebook 
    _, label = torch.max(dot_prod, 2) --label = argmax (index)
    
    dot_prod = nil 
    return {img, label}
end

function makeData_cls_pre(img, label)
    -- TODO: almost same as makeData_cls, need to convert img from RGB to BGR for caffe pre-trained model
    img = img:index(2, torch.LongTensor{3, 2, 1})
    return makeData_cls(img, label)
end


-- this version is as fast as matrix implementation above !!! 
function makeData_cls(img, label)
    -- TODO: the input label is a 3-channel real value image, quantize each pixel into classes (1 ~ 40)
    -- resize the label map from a matrix into a long vector
    -- hint: the label should be a vector with dimension of: opt.batchSize * opt.labelSize * opt.labelSize
    new_label = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

    local count = 1
    for i = 1, opt.batchSize do
        for j = 1, opt.labelSize do
            for k = 1, opt.labelSize do
                local current_label_pixel = torch.reshape(label[{{i}, {}, {j}, {k}}], 3) -- 3 x 1
                local dot_product = codebook * current_label_pixel
                local _, max_idx = torch.max(dot_product, 1) -- y dimension
                -- update new_label
                new_label[count] = max_idx
                count = count + 1
            end
        end
    end

    return {img, new_label}
end
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, imgpath, lblpath)
   collectgarbage()
   local img = loadImage(imgpath)
   local label = loadLabel_high(lblpath)
   img:add( - 127.5 )
   label:div(div_num)
   label:add(sub_num)

   return img, label

end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
   trainLoader.labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()
