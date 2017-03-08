require 'image' 
require 'nn' 

path = 'disp_conv/'
model = 'SAVEDMODEL_P/fcn_40.net' 

print('loading model to retrieve weights') 
net = torch.load(model).FCN 
conv1_W = net:get(1).weight 
print(#conv1_W) -- 96 x 3 x 11 x 11 

for i = 1, 96 do 
    x = conv1_W[i] -- 3 x 11 x 11
    save_path = path .. i .. '.png'
    --  rescale 
    image.save(save_path, x)

    -- rescale by loading 
    x = image.load(save_path, 3, 'byte') 

    -- save it again as a true png 
    image.save(save_path, x)
    image.save(path .. i .. '.jpg', x)  
end 

