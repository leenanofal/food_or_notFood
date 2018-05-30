#####
#Author: Leena Nofal
#This code is an implementation of a deep learning classifier 
#which classifies images from the Yelp Dataset as 'food' or 'not food' 
#The code is based on the inception-resnet-v2 architecture described in:
#Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning by Szegedy, et al. 
#Copyright Leena Nofal (C) 2018 all rights reserved. 
######

import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Input, Concatenate, Add, Lambda
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras import backend as K

class LeenaNet:
    
    #helper function 
    @staticmethod
    def conv2d(x,filters, kernel_size,strides,padding, activation='relu'):

        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,padding=padding)(x)
        if activation is not None:
            x = Activation(activation)(x)
        return x

    #block one of inception-resnet-v2. 'stem' name comes from paper 
    @staticmethod
    def stem(x):
        
        #add BN ?

        #layer one: 3x3 Conv, 32 filers, stride 2, valid pad
        #output 149 x 149 x 32
        x = LeenaNet.conv2d(x, 32, (3,3),2, 'valid')

        #layer 2
        #output 147 x 147 x 32
        x = LeenaNet.conv2d (x, 32, (3,3), 1, 'valid') 

        #layer 3
        #output 147 x 147 x 64
        x = LeenaNet.conv2d (x, 64, (3,3), 1, 'same')

        # x will go into two different layers and be concatenated 
        #x output 73 x 73 x 160 
        x_one = LeenaNet.conv2d(x, 96, (3,3), 2, 'valid') #output of x_one 73x73x96
        x_two = MaxPooling2D((3,3), 2, 'valid')(x)
        #concatenate x_one and x_two (maxpooling layer) 
        x = Concatenate()([x_one, x_two])
        
        #layers split
        x_a = LeenaNet.conv2d(x, 64, (1,1), 1, 'same') # 73x73x64
        x_a = LeenaNet.conv2d(x_a, 96, (3,3), 1, 'valid') #71x71x96
        x_b = LeenaNet.conv2d(x, 64, (1,1), 1, 'same') #73x73x64
        x_b = LeenaNet.conv2d(x_b, 64, (7,1), 1, 'same') #73x73x64
        x_b = LeenaNet.conv2d(x_b, 64, (1,7), 1, 'same') #73x73x64
        x_b = LeenaNet.conv2d(x_b, 96, (3,3), 1, 'valid') #71x71x96
        #concatenate x_a and x_b
        #output 71 x 71 x 192
        x = Concatenate()([x_a, x_b])
        #final layer of 'stem'
        x_c = LeenaNet.conv2d(x, 192, (3,3), 2, 'valid') ##MISTAKE IN PAPER 
        ##in the above layer the picture in the paper inaccurately shows that it is stride 1
        ##but to get an output of 32x32x192 it needs to be stride 2
        x_d = MaxPooling2D((2,2), 2, 'valid')(x) #paper didn't specifiy MaxPool size 
        x = Concatenate()([x_c, x_d]) # 35 x 35 x 384
            
        #return output of stem layer 
        return x
    
    #block 2 of inception-resnet-v2
    @staticmethod
    def incep_resnet_a(x, scale=0.2):
        #x = Activation('relu')(x)
        x_orig = x
        x_a = LeenaNet.conv2d(x, 32, (1,1), 1, 'same') #35x35x32
        x_a = LeenaNet.conv2d(x_a, 48, (3,3), 1, 'same') #35x35x48
        x_a = LeenaNet.conv2d(x_a, 64, (3,3), 1, 'same') #35x35x64
        x_b = LeenaNet.conv2d(x, 32, (1,1), 1, 'same')  #35x35x32
        x_b = LeenaNet.conv2d(x_b, 32, (3,3), 1, 'same') #35x35x32
        x_c = LeenaNet.conv2d(x, 32, (1,1), 1, 'same') #35x35x32
        x = Concatenate()([x_a, x_b, x_c]) 
        x = LeenaNet.conv2d(x, 384, (1,1), 1, 'same') #35 x 35 x 384
        x_orig_scaled = Lambda(lambda z: z * scale)(x_orig)
        x = Add()([x, x_orig_scaled])
        x = Activation('relu')(x)
        return x

    #block3 
    @staticmethod
    def reduction_a(x):

        x_a = LeenaNet.conv2d(x, 256, (1,1), 1, 'same') #35x35x256
        x_a = LeenaNet.conv2d(x_a, 256, (3,3), 1, 'same') #35x35x256 
        x_a = LeenaNet.conv2d(x_a, 384, (3,3), 2, 'valid') #17x17x384
        x_b = LeenaNet.conv2d(x, 384, (3,3), 2, 'valid') #17x17x384
        x_c = MaxPooling2D((3,3), 2, 'valid')(x) #17x17x384
        x = Concatenate()([x_a, x_b, x_c]) # final output 17x17x1152
        return x

    #block4
    @staticmethod
    def incep_resnet_b(x, scale=0.2):
        x_orig = x
        print("original x")
        print(x_orig)
        x_a = LeenaNet.conv2d(x, 128, (1,1), 1, 'same') #17x17x128
        x_a = LeenaNet.conv2d(x_a, 160, (1,7), 1, 'same') #17x17x160
        x_a = LeenaNet.conv2d(x_a, 192, (7,1), 1, 'same') #17x17x192
        x_b = LeenaNet.conv2d(x, 192, (1,1), 1, 'same') #17x17x192
        x = Concatenate()([x_a, x_b])
        ###Another mistake in paper. It says the number of filters 
        ###should be 1154 but that won't work with the previous layer's output 
        ###which is 1152
        x = LeenaNet.conv2d(x, 1152, (1,1), 1, 'same', None) #17x17x1154
        x_orig_scaled = Lambda(lambda z: z * scale)(x_orig)
        print('new x: ',x)
        x = Add()([x, x_orig_scaled])
        x = Activation('relu')(x)
        return x
        

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        scale = 0.2 
        #Building Model 
        x = Input(shape=inputShape)
        y = x
        #block1
        y = LeenaNet.stem(y) #output 35x35x384
        y = Activation('relu')(y) #activation f(n) after block1
        #block2 5x
        for i in range(5):
            y = LeenaNet.incep_resnet_a(y, scale)
        #block3 reduction
        y = LeenaNet.reduction_a(y) #output 17x17x1152
        y = Activation('relu')(y) #activation fn after block3
        #block4 10x
        ###mistake2 in the Inception-ResNet-v2 paper's diagram of block4
        ###says 1154 filters which means the output is 17x17x1154, however, 
        ###the model will not compile because the previous layer's output 
        ###is 1152
        for i in range(10):
            y = LeenaNet.incep_resnet_b(y, scale) #17x17x1152
        y = Flatten()(y)
        y = Dense(1)(y)
        y = Activation('sigmoid')(y)
        #print(y)
        model = Model(inputs=[x], outputs=[y])
        
        
        # return the constructed network architecture
        return model


