from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization 
from keras.layers import Activation, MaxPool2D, Conv2DTranspose 
from keras.layers import Concatenate, Dropout,GroupNormalization

def conv_block(input,num_filters,activation='relu'):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = GroupNormalization(groups=num_filters)(x)
    x = Activation(activation)(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = GroupNormalization(groups=num_filters)(x)
    x = Activation(activation)(x)

    return x

def enconder_block(input, num_filters, activation):
    x = conv_block(input, num_filters, activation)
    p = MaxPool2D((2,2))(x)
    
    return x, p

def decoder_block(input, skip_features, num_filters, activation):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    x = Concatenate()([x,skip_features])
    x = conv_block(x, num_filters, activation)
    
    return x

def unet_model(input_shape, activation):
    inputs = Input(input_shape)

    s1,p1 = enconder_block(inputs, 64, activation)
    s2,p2 = enconder_block(p1, 128, activation)
    s3,p3 = enconder_block(p2, 256, activation)
    s4,p4 = enconder_block(p3, 512, activation)
    
    b1 = conv_block(p4,1024)

    d2 = decoder_block(b1, s4, 512, activation)
    d3 = decoder_block(d2, s3, 256, activation)
    d4 = decoder_block(d3, s2, 128, activation)
    d5 = decoder_block(d4, s1, 64, activation)

    output = Conv2D(1,3,padding='same',activation='sigmoid')(d5)
  
    model = Model(inputs=inputs,outputs=output)
    model.summary()
    return model 