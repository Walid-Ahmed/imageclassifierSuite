
#usage python trainCIFAR10.py


from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels


from tensorflow.keras.datasets import mnist


from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from  util import  plotUtil

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import cv2
from keras.preprocessing.image import img_to_array

from keras.datasets import cifar10
from keras.datasets import fashion_mnist
from keras.datasets import cifar100
from keras.datasets import mnist

from sklearn.metrics import confusion_matrix

import argparse



import numpy as np

import pickle
import os
from imutils import build_montages


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()



def defineModel():   #Cifar10.    #https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
		weight_decay = 1e-4

		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=(32,32,3)))
		model.add(tf.keras.layers.Activation('elu'))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
		model.add(tf.keras.layers.Activation('elu'))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
		model.add(tf.keras.layers.Dropout(0.2))
		 
		model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
		model.add(tf.keras.layers.Activation('elu'))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
		model.add(tf.keras.layers.Activation('elu'))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
		model.add(tf.keras.layers.Dropout(0.3))
		 
		model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
		model.add(tf.keras.layers.Activation('elu'))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
		model.add(tf.keras.layers.Activation('elu'))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
		model.add(tf.keras.layers.Dropout(0.4))
		 
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(10, activation="softmax"))

		# return the constructed network architecture
		return model 

if __name__ == '__main__':

	INIT_LR = 1e-3
	EPOCHS=25
	BS=32


	(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()
	labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



	
	numPfSamples,imgWidth,imgHeight,numOfchannels=trainData.shape




	print("[INFO] Original cifar10 dataset of trainData shape {}".format(trainData.shape))
	print("[INFO] Original cifar10 dataset of trainLabels shape {}".format(trainLabels.shape))
	print("[INFO] Original cifar10 dataset of testData shape {}".format(testData.shape))
	print("[INFO] Original cifar10 dataset of testLabels shape {}".format(testLabels.shape))


	trainX = trainData.reshape((trainData.shape[0], imgWidth,imgHeight,  numOfchannels))
	testX = testData.reshape((testData.shape[0], imgWidth,imgHeight, numOfchannels))

	input("[INFO] Press anykey to start training")


	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0



	


	trainLabels=trainLabels.astype(str)
	testLabels=testLabels.astype(str)
	#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainLabels)
	testY = lb.fit_transform(testLabels)
	numOfOutputs=len(lb.classes_)
	#print(lb.classes_)






	model=defineModel()
	model.summary()

	opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


	# train the network
	print("[INFO] training network...")
	#model.fit(trainX, trainY, batch_size=128, epochs=20,verbose=1)

	aug = ImageDataGenerator()


	history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)


	

	# serialize the label binarizer to disk
	fileNameToSaveLabels=dataset+"_labels.pkl"
	fileNameToSaveLabels=os.path.join("Results",fileNameToSaveLabels)
	f = open(fileNameToSaveLabels, "wb")
	f.write(pickle.dumps(lb.classes_))
	f.close()


	info1=plotUtil.plotAccuracyAndLossesonSameCurve(history,dataset+"_")
	info2=plotUtil.plotAccuracyAndLossesonSDifferentCurves(history,dataset+"_")


	#sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False)
	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	y_true=testY.argmax(axis=1)
	y_pred=predictions.argmax(axis=1)


	print(type(lb.classes_))
	print(classification_report(y_true,y_pred, target_names=labels))

	


	fileToSaveModel=os.path.join("Results",dataset+"Lenet.keras2")
	model.save(fileToSaveModel)

	print("[INFO] Model saved to {}".format(fileToSaveModel))
	print("[INFO] Labels  saved to {}".format(fileNameToSaveLabels))
	print(info1)
	print(info2)


	images=[]
	for imageIndex in range(9):

		image = (testData[imageIndex]).astype("uint8")
		imgDisplay=image.copy()

		# pre-process the image for classification
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		#print(image.shape) (1,28,28,1)

		predictions = model.predict(image, batch_size=32)
		#print("predictions={}".format(predictions))

	 
		# show the image and prediction

		 # merge the channels into one image
		if(numOfchannels==1):
			imgDisplay = cv2.merge([imgDisplay] * 3)
		else:
			imgDisplay=cv2.cvtColor(imgDisplay, cv2.COLOR_RGB2BGR)		
		#print(imgDisplay.shape)
	 
		# resize the image from a 28 x 28 image to a 96 x 96 image so we
		# can better see it
		#imgDisplay = cv2.resize(imgDisplay, (96, 96), interpolation=cv2.INTER_LINEAR)
		y_pred_img=predictions.argmax(axis=1)
		predictedLabel=labels[y_pred_img[0]]
		actualLabel=labels[np.argmax(testY[imageIndex])]

		cv2.putText(imgDisplay, str(predictedLabel), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		images.append(imgDisplay)
		print("[INFO] Predicted: {}, Actual: {}".format(predictedLabel,actualLabel))
		#cv2.imshow("Digit", imgDisplay)
		#cv2.waitKey(0)

	montage = build_montages(images, (128, 128), (3, 3))
	cv2.imshow("Sample prediction  from {} testing  dataset".format(dataset), montage[0])
	fileToSaveResults=os.path.join("Results",dataset+"_result.png")
	cv2.imwrite(fileToSaveResults,montage[0])
	cv2.waitKey(1000)	
	print("[INFO] Sample results   saved to {}".format(fileToSaveResults))


	# Plot non-normalized confusion matrix
	cm=plot_confusion_matrix(y_true, y_pred, classes=labels,dataset=dataset,title=dataset+ '_Confusion matrix, without normalization') 
	cm = confusion_matrix(y_true, y_pred)
	print_cm(cm, labels)

	# Plot normalized confusion matrix
	#plot_confusion_matrix(y_true, y_pred, classes=labels, dataset=dataset,normalize=True, title=dataset+'_Normalized confusion matrix')

	#plt.show()


