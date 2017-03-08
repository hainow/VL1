--local dbg = require('debugger.lua')
require 'torch'
require 'optim'
require 'pl'
require 'paths'

local fcn = {} -- new class

local inputs = torch.CudaTensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
local targets = torch.CudaTensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- training function
function fcn.train(inputs_all)
    -- inputs_all = {input, target}
    cutorch.synchronize()
    epoch = epoch or 1
    local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
    local dataBatchSize = opt.batchSize

    inputs:copy(inputs_all[1])
    targets:copy(inputs_all[2])

    -- TODO: implemnet the training function
    local err, outputs 
    function opfunc(params_FCN)
	collectgarbage()
        -- look up online
        -- set grad = 0
        -- if x ~= parameters_FCN then -- get new parameters
        --     params_FCN:copy(x)
        -- end
        gradParams_FCN:zero()

        -- calculate ouputs
        outputs = model_FCN:forward(inputs)

        -- does forward_path is err
        err = criterion:forward(outputs, targets)
        --print(string.format('FCN loss: %f', err))
        -- calculate backward df
        local df_samples = criterion:backward(outputs, targets)

        -- change gradParams by calling model_FCN:backward(inputs, df)
        model_FCN:backward(inputs, df_samples)

        return err, gradParams_FCN
    end

    --inputs:copy(inputs_all[1])
    --targets:copy(inputs_all[2])

    optim.sgd(opfunc, params_FCN, optimState)
    batchNumber = batchNumber + 1
    cutorch.synchronize(); collectgarbage();
    print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f, loss %.3f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime, err))
    dataTimer:reset()
end


return fcn


