from tensorflow.keras import layers, models, activations

def conv_block(x, num_filters, ksize, strides=(1, 1), padding='same', activation='relu', name='conv_block'):
    output = layers.Conv2D(num_filters, ksize, strides=strides, padding=padding, activation="linear", name=name+"_conv")(x)
    output = layers.BatchNormalization(name=name+"_bn")(output)
    if activation == 'leaky_relu':
        output = layers.LeakyReLU(alpha=0.01, name=name+"_"+activation)(output)
    else:
        output = layers.Activation(activation, name=name+"_"+activation)(output)
    return output

def residual_block(x, num_filters, strides=(1, 1), activation='relu', use_branch=True, name='res_block'):
    
    if use_branch: 
        branch1 = conv_block(x, num_filters, 1, strides=strides, padding='valid', activation='linear', name=name+"_branch1")
    else : 
        branch1 = x
        
    branch2 = conv_block(x, num_filters//4, 1, strides=strides, padding='valid', activation=activation, name=name+"_branch2a")
    branch2 = conv_block(branch2, num_filters//4, 3, activation=activation, name=name+"_branch2b")
    branch2 = conv_block(branch2, num_filters, 1, activation='linear', name=name+"_branch2c")

    output = layers.Add(name=name+"_add")([branch1, branch2])
    if activation == 'leaky_relu':
        output = layers.LeakyReLU(alpha=0.01, name=name+"_"+activation)(output)
    else:
        output = layers.Activation(activation, name=name+"_"+activation)(output)
    return output

def build_resnet(num_layer = 50, input_shape=(None, None, 3, ), activation='relu', name="Net"): 
    num_layer_list = [50, 101, 152]
    
    blocks_dict = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3], 
        152: [3, 8, 36, 3]
    }

    num_channel_list = [256, 512, 1024, 2048]
    block_name = ['a', 'b', 'c', 'd']
    assert num_layer in  num_layer_list, "Number of layer must be in %s"%num_layer_list
    
    name = name+str(num_layer)

    _input = layers.Input(shape=input_shape, name=name+"_input")

    x = layers.ZeroPadding2D((3, 3), name=name+"_pad")(_input)
    x = conv_block(x, 64, 7, (2, 2), 'valid', activation, name=name+'_stem')
    x = layers.MaxPool2D(name=name+'_pool')(x)
    
    for idx, num_iter in enumerate(blocks_dict[num_layer]):
        for j in range(num_iter):
            if j==0:
                x = residual_block(x, num_channel_list[idx], activation=activation,  strides=(2, 2), name=name+'_res_'+block_name[idx]+str(j))
            else:
                x = residual_block(x, num_channel_list[idx], activation=activation, use_branch=False, name=name+'_res_'+block_name[idx]+str(j))

    return models.Model(_input, x, name=name)

model = build_resnet(50, (224, 224, 3), activation='leaky_relu', name="Self")
model.summary()