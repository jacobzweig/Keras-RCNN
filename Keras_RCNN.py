'''
2016 by Jacob Zweig @jacobzweig
build RCNN networks in keras
'''
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import merge, Convolution2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=5,
		 nbFilters=128, filtersize = 3):


	model = BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize):
    
    def RCL_block(l_settings, l, pool=True, increase_dim=False):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters
		   
        conv1 = Convolution2D(out_num_filters, 1, 1, border_mode='same')
        stack1 = conv1(l)   	
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)
        
        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
        stack4 = conv2(stack3)
        stack5 = merge([stack1, stack4], mode='sum')
        stack6 = BatchNormalization()(stack5)
        stack7 = PReLU()(stack6)
    	
        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack8 = conv3(stack7)
        stack9 = merge([stack1, stack8], mode='sum')
        stack10 = BatchNormalization()(stack9)
        stack11 = PReLU()(stack10)    
        
        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack12 = conv4(stack11)
        stack13 = merge([stack1, stack12], mode='sum')
        stack14 = BatchNormalization()(stack13)
        stack15 = PReLU()(stack14)    
        
        if pool:
            stack16 = MaxPooling2D((2, 2), border_mode='same')(stack15) 
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = Dropout(0.1)(stack15)
            
        return stack17

    #Build Network
    input_img = Input(shape=(nbChannels, shape1, shape2))
    conv_l = Convolution2D(nbFilters, filtersize, filtersize, border_mode='same', activation='relu')
    l = conv_l(input_img)
    
    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=False)
        else:
            l = RCL_block(conv_l, l, pool=True)
    
    out = Flatten()(l)        
    l_out = Dense(nbClasses, activation = 'softmax')(out)
    
    model = Model(input = input_img, output = l_out)
    
    return model
