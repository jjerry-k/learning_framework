from tensorflow.keras import layers, models, losses, optimizers

def build_vgg(input_shape=(None, None, 3), num_layer=16, name='vgg'):
    num_layer_list = [11, 13, 16, 19]
    
    blocks_dict = {
        11: [1, 1, 2, 2, 2],
        13: [2, 2, 2, 2, 2], 
        16: [2, 2, 3, 3, 3], 
        19: [2, 2, 4, 4, 4]
    }

    num_channel_list = [64, 128, 256, 512, 512]

    def conv_block(input_layer, out_channel, num_iters, name=name+"_block1"):
        prev_layer = input_layer
        for i in range(num_iters):
            output = layers.Conv2D(out_channel, 3, strides=1, padding='same', activation='relu', name=name+"_conv%d"%(i+1))(prev_layer)
            prev_layer = output
        return output

    assert num_layer in  num_layer_list, "Number of layer must be in %s"%num_layer_list
    
    input_layer = layers.Input(shape=input_shape, name=name+str(num_layer)+"_input")
    prev_layer = input_layer
    for idx, num_iter in enumerate(blocks_dict[num_layer]):
        output = conv_block(prev_layer, num_channel_list[idx], num_iter, name=name+str(num_layer)+"_block%d"%(idx+1))
        output = layers.MaxPool2D(name=name+str(num_layer)+"_block%d_pool"%(idx+1))(output)
        prev_layer = output
    
    return models.Model(inputs= input_layer, outputs=output, name=name+str(num_layer))

    def build_fcn(input_shape= (None, None, 3), fcn_architecture= 8,  name='fcn'):
        architecture_list = [8, 16, 32]
        blocks_dict = {
            8: 3, 
            16: 2, 
            32: 1}

        assert fcn_architecture in  architecture_list, "FCN Architecture must be in %s"%architecture_list

        base_net = build_vgg(num_layer=16, name=name+str(fcn_architecture)+"_vgg")
        prev_layer = base_net.output

        for idx in range(blocks_dict[fcn_architecture]):
            