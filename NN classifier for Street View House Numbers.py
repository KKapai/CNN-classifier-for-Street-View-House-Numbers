import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt # plt for displaying pictures
import matplotlib.image as mpimg # mpimg is used to read pictures
import scipy.io as sio
nb_class = 10#There are 10 categories in total
nb_epoch = 4#The number of times the network was trained.
batchsize = 128#The number of samples used for each model training is 128
img_rows, img_cols = 32, 32#Image data size
# Read the training data set and convert the data format
Train_matFile = 'train_32x32.mat'
Train_datas = sio.loadmat(Train_matFile);#Read data in mat format
XTrain = Train_datas['X'];#Training image data
Ytrain= Train_datas['y'];#Training label data
x_train = np.arange(75015168).reshape(73257,32,32);#Initialize a 73257 * 32 * 32   array
y_train = [];#Storing training labels
for i in range(0,73257):#Read 73257 image data circularly
    x_Train_temp = XTrain[:,:,0,i];#Read one of the image data
    x_train[i,:,:] = x_Train_temp;#Store the data in x_train
x_train = np.array(x_train,dtype='uint8'); #Convert image data to uint8 type
y_train = Ytrain.flatten();#Convert training labels to a one-dimensional array
y_train = y_train-1;#Convert training labels to 0-9 label format types for keras
# Read the testing data set and convert the data format
Test_matFile = 'test_32x32.mat'
Test_datas = sio.loadmat(Test_matFile);#Read data in mat format
XTest = Test_datas['X'];#Testing image data
Ytest= Test_datas['y'];#Testing label data
x_test = np.arange(26656768).reshape(26032,32,32);#Initialize a 26032 * 32 * 32  array
y_test = [];#Storing testing labels
for i in range(0,26032):#Read 26032 image data circularly
    x_temp = XTest[:,:,1,i];#Read one of the image data
    x_test[i,:,:] = x_temp;#Store the data in x_test
x_test = np.array(x_test,dtype='uint8'); #Convert image data to uint8 type
y_test = Ytest.flatten();#Convert testing labels to a one-dimensional array
y_test = y_test - 1;#Convert testing labels to 0-9 label format types for keras


# setup data shape
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)  # 32*32:sample size,1:single channel
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)  # 32*32:sample size,1:single channel
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# one-hot
y_train = np_utils.to_categorical(y_train, nb_class)#Converts a category vector to a binary (only 0 and 1) matrix type representation. Its performance is to convert the original category vector into a form of one-hot encoding
y_test = np_utils.to_categorical(y_test, nb_class)#Converts a category vector to a binary (only 0 and 1) matrix type representation. Its performance is to convert the original category vector into a form of one-hot encoding

# setup model
model = Sequential()#Instantiate model objects using the Sequential method of the keras.model library

# 1st conv2d layer
model.add(Convolution2D(
        filters = 32, #32 filters-> 32 depths
        kernel_size = [5, 5], #Filter window size (5, 5)
        padding = 'same', # Filter mode
        input_shape = (img_rows, img_cols, 1)# Input shape is picture shape # default data_format: channels_last (rows, cols, channels)
))

model.add(Activation('relu')) # Activation function is relu

model.add(MaxPool2D(
    pool_size=(2, 2), #Represents the downsampling factor in two directions (vertical, horizontal), such as (2, 2) will make the picture become half the original length in both dimensions
    strides=(2, 2), #strides represents the step size of the slide
    padding='same'# Filter mode
))

# 2nd conv2d layer
model.add(Convolution2D(
    filters=64, #64 filters-> 64 depths
    kernel_size=(5, 5), #Filter window size (5, 5)
    padding='same'# Filter mode
))

model.add(Activation('relu'))# Activation function is relu

model.add(MaxPool2D(
    pool_size=(2, 2), #Represents the downsampling factor in two directions (vertical, horizontal), such as (2, 2) will make the picture become half the original length in both dimensions
    strides=(2, 2), #strides represents the step size of the slide
    padding='same'# Filter mode
))

# 1st fully connected dense
model.add(Flatten())#The Flatten layer is used to "flatten" the input and one-dimensionalize the multi-dimensional input.
model.add(Dense(2048))#2048 neurons
model.add(Activation('relu'))# Activation function is relu

# 2nd fully connected dense
model.add(Dense(10))#10 neurons
model.add(Activation('softmax')) # the last activation function definitely is softmax

# define optimizer and setup param
adam = Adam(lr=0.0003)#Adam Optimizer,lr is the learning rate and learning step

# compile model
model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# run network
model.fit(
    x = x_train,
    y = y_train,
    epochs = nb_epoch,
    batch_size = batchsize,
    validation_data = (x_test,y_test)
)
model.save('model')
#Classification accuracy using trained models
new_model = load_model('model')
(loss, accuracy) = new_model.evaluate(x_test, y_test)
print('loss is:=>>' , loss)
print('accuracy is:=>>' , accuracy)


