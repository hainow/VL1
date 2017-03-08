require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


local sanitize = require('sanitize')


----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp [[
  -s,--save          (default "./SAVEDMODEL_P")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -r,--learningRate  (default 0.001)      learning rate
  -b,--batchSize     (default 10)         batch size
  -m,--momentum      (default 0.9)         momentum term of adam
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 1)          gpu to run on (default cpu)
  --scale            (default 512)          scale of images to train on
  --epochSize        (default 2000)        number of samples per epoch
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --weightDecay      (default 0.0005)        weight decay
  --classnum         (default 40)    
  --classification   (default 1)
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.loadSize = opt.scale
-- TODO: setup the output size 
-- opt.labelSize = ?
opt.labelSize = 16
 
opt.manualSeed = torch.random(1, 10000)
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
    cutorch.setDevice(opt.gpu + 1)
    print('<gpu> using device ' .. opt.gpu)
    torch.setdefaulttensortype('torch.CudaTensor')
else
    torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = { 3, opt.scale, opt.scale }
opt.outDim = opt.classnum


local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.01)
        m.bias:fill(0)
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end


if opt.network == '' then
    ---------------------------------------------------------------------
    -- TODO: load pretrain model and add some layers, let's name it as model_FCN for the sake of simplicity
    -- hint: you might need to add large padding in conv1 (perhaps around 100ish? )
    -- hint2: use ReArrange instead of Reshape or View
    -- hint3: you might need to skip the dropout and softmax layer in the model

    --[[
            nn.Sequential {
          [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) -> output]
          (1): nn.SpatialConvolution(3 -> 96, 11x11, 4,4)
          (2): nn.ReLU
          (3): nn.SpatialCrossMapLRN
          (4): nn.SpatialMaxPooling(3x3, 2,2)
          (5): nn.SpatialConvolution(96 -> 256, 5x5, 1,1, 2,2)
          (6): nn.ReLU
          (7): nn.SpatialCrossMapLRN
          (8): nn.SpatialMaxPooling(3x3, 2,2)
          (9): nn.SpatialConvolution(256 -> 384, 3x3, 1,1, 1,1)
          (10): nn.ReLU
          (11): nn.SpatialConvolution(384 -> 384, 3x3, 1,1, 1,1)
          (12): nn.ReLU
          (13): nn.SpatialConvolution(384 -> 256, 3x3, 1,1, 1,1)
          (14): nn.ReLU
          (15): nn.SpatialMaxPooling(3x3, 2,2)
          (16): nn.View(-1)
          (17): nn.SpatialConvolution(256 -> 4096, 6x6)
          (18): nn.ReLU
          (19): nn.Dropout(0.500000)
          (20): nn.SpatialConvolution(4096 -> 4096, 1x1)
          (21): nn.ReLU
          (22): nn.Dropout(0.500000)
          (23): nn.Linear(4096 -> 1000)
          (24): nn.SoftMax
        }
    ]]--

    model_FCN = torch.load("/usr0/home/htpham/Downloads/824/a1/AlexNet")
    model_FCN:remove(19)
    model_FCN:remove(16)
    model_FCN:remove()
    model_FCN:remove()
    model_FCN:remove()
    model_FCN:add(nn.SpatialConvolution(4096, opt.classnum, 1,1,1,1))
    model_FCN:add(nn.Transpose({2,3},{3,4}))
    model_FCN:add(nn.View(-1, opt.classnum))
    model_FCN:add(nn.LogSoftMax())

    -- TODO: padding for conv1
    model_FCN['modules'][1].padW = 100
    model_FCN['modules'][1].padH = 100
    print('done with padding and model')
else
    features = nn.Sequential()
    print('<trainer> reloading previously trained network: ' .. opt.network)
    tmp = torch.load(opt.network)
    model_FCN = tmp.FCN
end


-- TODO: loss function
criterion = nn.ClassNLLCriterion()
print('done with loss function')

print(model_FCN)
--model_FCN:apply(weights_init)


-- TODO: convert model and loss function to cuda
print("Migrating model to GPU...")
model_FCN:cuda()
criterion:cuda()


-- TODO: retrieve parameters and gradients
print('getting params')
params_FCN, gradParams_FCN = model_FCN:getParameters()
print('done with getting params')

-- print networks
print('fcn network:')
print(model_FCN)

-- TODO: setup dataset, use data.lua
print('calling data.lua')
paths.dofile('data.lua')
print('done with data.lua')

-- TODO: setup training functions, use fcn_train_cls.lua
print('calling fcn_train_cls.lua')
fcn = require('fcn_train_cls.lua')
print('done with fcn_train_cls.lua')

local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local function train()
    print('\n<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')

    model_FCN:training()
    batchNumber = 0
    for i = 1, opt.epochSize do
        donkeys:addjob(function()
            return makeData_cls_pre(trainLoader:sample(opt.batchSize))
            --return makeData_cls(trainLoader:sample(opt.batchSize))
        end,
            fcn.train)
    end
    donkeys:synchronize()
    cutorch.synchronize()
end

epoch = 1
-- training loop
while true do
    -- train/test
    train()

    if epoch % opt.saveFreq == 0 then
        local filename = paths.concat(opt.save, string.format('fcn_%d.net', epoch))
        os.execute('mkdir -p ' .. sys.dirname(filename))
        if paths.filep(filename) then
            os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainer> saving network to ' .. filename)
        torch.save(filename, { FCN = sanitize(model_FCN), opt = opt })
        --torch.save(filename, { FCN = model_FCN, opt = opt })

        	
    end

    print("at epoch" .. epoch .. " loss = " .. criterion.output)

    epoch = epoch + 1

    -- plot errors
    if opt.plot and epoch and epoch % 1 == 0 then
        torch.setdefaulttensortype('torch.FloatTensor')

        if opt.gpu then
            torch.setdefaulttensortype('torch.CudaTensor')
        else
            torch.setdefaulttensortype('torch.FloatTensor')
        end
    end
end
