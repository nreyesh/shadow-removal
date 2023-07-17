from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate


def conv_block(input,num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def enconder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2,2))(x)
    
    return x, p

def decoder_block(input, skip_featrues, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), stride=2, padding='same')(input)
    x = Concatenate()([x,skip_featrues])
    x = conv_block(x, num_filters)
    
    return x

def build_model(input_shape):
    inputs = Input(input_shape)

    s1,p1 = enconder_block(inputs, 64)
    s2,p2 = enconder_block(p1, 128)
    s3,p3 = enconder_block(p2, 256)
    s4,p4 = enconder_block(p3, 512)

    b1 = conv_block(p4,1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    output = Conv2D(3,3,padding='same',activation='sigmoid')(d4)
  
    model = Model(inputs=inputs,outputs=output)
    model.summary()
    return model 