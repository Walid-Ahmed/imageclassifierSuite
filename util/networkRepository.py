from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation, Dense 
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import plot_model
import os
import cv2
from keras.layers import BatchNormalization
from keras.regularizers import l2
nb_filters=32
nb_conv=3
img_channels=3
num_classes=2
img_rows, img_cols=16,16

reg=l2(0.0002)


def createNet12():
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	model.add(Dropout(0.1))

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.1))

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.1))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.1))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))

	#model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#12")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	print("model image saved to file "+pathToSaveModelImage)
	return model
	

def createNet12BN():
	# Create the model
	model = Sequential()
	
	#model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	#model.add(Dropout(0.1))
	
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels),padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	
	
	
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.1))
	
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	
	

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.1))
	
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	
	#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.1))
	
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())

	#model.add(Dropout(0.5))
	#model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	

	model.add(Dense(num_classes))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	
	print("Network#12BN")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model
		
# Create the model from net11
def createNet11():
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#11 ")
	#plot_model(model, to_file=pathToSaveModelImage,show_shapes=True,show_layer_names=False)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file "+pathToSaveModelImage)
	return model

def createNet11BN():
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(num_classes))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	print("Network#11BN ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model

def createNet10():
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#10 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model


def createNet9():
# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#9 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model




def createNet8():
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	#model.add(Dropout(0.5))
	#model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#8")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model



def createNet7():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	#model.add(Dropout(0.5))
	#model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#7")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	print("model image saved to file")
	return model


def createNet6():
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	#model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#6 ")
	return model



def createNet5():
	# Create the model
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#5 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model


def createNet4():
	# Create the model
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	#model.add(Dropout(0.5))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#4 ")
	return model

def createNet3():
	img_channels=3
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#3 ")
	#plot_model(model, to_file=pathToSaveModelImage,show_shapes=True, show_layer_names=False)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file "+pathToSaveModelImage)
	return model



def createNet2():
	# Create the model
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	print("Network#2 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model



def createNet1():
	# Create the model
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	#model.add(Dropout(0.5))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#1 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model



def createNet13():  #network_tmp
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=( img_rows, img_cols, img_channels)))
		model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
		model.add(Dropout(0.75))
		model.add(Flatten())
		model.add(Dense(num_classes, activation='softmax'))
		return model


def createNet14():
		model = Sequential()
		inputShape = (img_rows, img_cols, img_channels)
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(64, (8, 8), input_shape=inputShape,padding="same", kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(128, (5, 5), padding="same",kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# third (and final) CONV => RELU => POOL layers
		model.add(Conv2D(256, (3, 3), padding="same",kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# first and only set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(num_classes))
		model.add(Activation("softmax"))
		return model

def createNet15():    #based on net#1
		# Create the model
		model = Sequential()
		model.add(Conv2D(64, (8, 8), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_regularizer=reg))
		#model.add(Dropout(0.5)
		model.add(Conv2D(128, (5, 5), activation='relu', padding='same', kernel_regularizer=reg))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax',kernel_regularizer=reg))
		print("Network#15 ")
		#plot_model(model, to_file=pathToSaveModelImage)
		#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
		#print("model image saved to file")
		return model

def createNet16():  #based on net#2
	# Create the model
	model = Sequential()
	model.add(Conv2D(64, (8, 8), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (5, 5), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#16 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model



def createNet17():   #based on net#3
	img_channels=3
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#17 ")
	#plot_model(model, to_file=pathToSaveModelImage,show_shapes=True, show_layer_names=False)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file "+pathToSaveModelImage)
	return model


def createNet18():   #based on net#4
	# Create the model
	model = Sequential()
	model.add(Conv2D(64, (8, 8), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_regularizer=reg))
	#model.add(Dropout(0.5))
	model.add(Conv2D(128, (4, 4), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax' ,  kernel_regularizer=reg))
	print("Network#18 ")
	return model



def createNet19():    # #based on net#5
	# Create the model
	model = Sequential()
	model.add(Conv2D(64, (8, 8), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_regularizer=reg))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (4, 4), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax',kernel_regularizer=reg))
	print("Network#19 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model



def createNet20():  #based on  net#6
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#20 ")
	return model





def createNet21():      #based on net#7
	model = Sequential()
	model.add(Conv2D(32, (12, 12), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(Conv2D(32, (8, 8), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (6, 6), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	#model.add(Dropout(0.5))
	#model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#21")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	print("model image saved to file")
	return model
	
	
	
def createNet22():      #based on net#8
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (12, 12), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (8, 8), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (8, 8), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (6, 6), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	#model.add(Dropout(0.5))
	#model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#22")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model	


def createNet23():   #based on net9
# Create the model
	model = Sequential()
	model.add(Conv2D(32, (12, 12), input_shape=( img_rows, img_cols, img_channels), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (12, 12), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (9, 9), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(32, (6, 6), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	#model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(2048, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#23 ")
	#plot_model(model, to_file=pathToSaveModelImage)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file")
	return model
	


def createNet24():   #based on net11
	model = Sequential()
	model.add(Conv2D(32, (9, 9), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_regularizer=reg))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (6, 6), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#24 ")
	#plot_model(model, to_file=pathToSaveModelImage,show_shapes=True,show_layer_names=False)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file "+pathToSaveModelImage)
	return model


def createNet25():   
	model = Sequential()
	model.add(Conv2D(64, (9, 9), input_shape=( img_rows, img_cols, img_channels), padding='same', activation='relu', kernel_regularizer=reg))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (6, 6), activation='relu', padding='same', kernel_regularizer=reg))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_regularizer=reg))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	print("Network#24 ")
	#plot_model(model, to_file=pathToSaveModelImage,show_shapes=True,show_layer_names=False)
	#cv2.imshow(networkName,cv2.imread(pathToSaveModelImage))
	#print("model image saved to file "+pathToSaveModelImage)
	return model


modelMediator = {"net1" : createNet1,
           "net2" : createNet2,
           "net3" : createNet3,
           "net4" : createNet4,
           "net5" : createNet5,
           "net6" : createNet6,
           "net7" : createNet7,
           "net8" : createNet8,
           "net9" : createNet9,
           "net10"  : createNet10,
           "net11" : createNet11,
           "net11BN": createNet11BN,
           "net12" : createNet12,
           "net12BN" : createNet12BN,
           "net13" : createNet13,
           "net14" : createNet14,
           "net15" : createNet15,
           "net16" : createNet16,
           "net17" : createNet17,
           "net18" : createNet18,
           "net19" : createNet19,
           "net20" : createNet20,
           "net21" : createNet21,
           "net22" :createNet22,
           "net23" :createNet23,
           "net24" :createNet24,
           "net25" :createNet25
           
           
}

#"net1" ,

allNetworks=[
		#"net1" ,
           "net2" ,
           "net3" ,
           "net4" ,
           "net5" ,
           "net6" ,
           "net7" ,
           "net8" ,
           "net9" ,
           "net10" ,
           "net11" ,
           "net11BN",
           "net12" ,
           "net12BN" ,
           "net13" ,
           "net14","net15","net16","net17","net18","net19","net20","net21","net22","net23","net24","net25"
           
           ]
       
allNetworks=[
		   "net1" ,
           
           "net4" ,
           "net5" ,
           
          
          
           "net11" ,
           "net11BN",

           "net12BN" ,

           
           
           ]    
           
           
           
#allNetworks=["net25"]
