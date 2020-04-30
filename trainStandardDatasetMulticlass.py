
#usage python trainStandardDatasetMulticlass.py --dataset MNIST  --networkID  LenetModel --EPOCHS 20
#usage python trainStandardDatasetMulticlass.py  --dataset fashion_mnist --networkID MiniVGG --EPOCHS 25
#usage python trainStandardDatasetMulticlass.py  --dataset CIFAR10 --networkID net5  --EPOCHS 25    
#usage python trainStandardDatasetMulticlass.py  --dataset CIFAR100 --networkID MiniVGG  --EPOCHS 25  


from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels


from tensorflow.keras.datasets import mnist


from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
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
from util import helper



import numpy as np

import pickle
import os
from imutils import build_montages
from  util import  plotUtil






if __name__ == '__main__':


	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True, help="name of the dataset")
	ap.add_argument("--networkID", required=True, help="name of the network")
	ap.add_argument("--EPOCHS", required=True, help="name of the network")


	args = vars(ap.parse_args())
	dataset=args["dataset"]
	networkID=args["networkID"]
	EPOCHS=int(args["EPOCHS"])

	BS=32
	INIT_LR = 1e-3

	labels=[]
	print("[INFO] downloading {0}...".format(dataset))

	if(dataset=="MNIST"):
		(trainX, trainY), (testX, testY) = mnist.load_data()
		labels =  ['0','1','2','3','4','5','6','7','8','9']




	if(dataset=="CIFAR10"):
		(trainX, trainY), (testX, testY) = cifar10.load_data()
		labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



	if(dataset=="CIFAR100"):
		(trainX, trainY), (testX, testY) = cifar100.load_data()
		labels=['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm']


	if(dataset=="fashion_mnist"):
		(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
		labels = ["top", "trouser", "pullover", "dress", "coat","sandal", "shirt", "sneaker", "bag", "ankle boot"]


	#get original img width , hight and number of channels
	try:
		numPfSamples,imgWidth,imgHeight,numOfchannels=trainX.shape
    
    '''
	consider the architecture of a CNN. We need to specify the number of channels and our input data must have the shape HxWxD where “H” is the height, 
	“W” is the width, and “D” is the depth. We add in that “D” channel dimension (setting D=1) in order to make the dataset compatible with our architecture. 
	If you used RGB images then D=3.  ref:https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/
    '''
	
	except:   #dataset is single channe;
		numPfSamples,imgWidth,imgHeight=trainX.shape
		numOfchannels=1
		trainX = np.expand_dims(trainX, axis=-1)
		testX = np.expand_dims(testX, axis=-1)


	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0

	print("[INFO] Original {} dataset of trainData shape {}".format(dataset,trainX.shape))
	print("[INFO] Original {} dataset of trainLabels shape {}".format(dataset,trainY.shape))
	print("[INFO] Original {} dataset of testData shape {}".format(dataset,testX.shape))
	print("[INFO] Original {} dataset of testLabels shape {}".format(dataset,testY.shape))
	input("press any key to continue")



	#Prepare images for grid show
	images=[]
	for  i in range(16):
		image=trainX[i]
		if(numOfchannels==1):
			image = cv2.merge([image] * 3)
		else:
			image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)	
		images.append(image)
	
	fileToSaveSampleImage=os.path.join("Results","sample_"+dataset+".png")
	plotUtil.drarwGridOfImagesFromImagesData(images,fileToSaveSampleImage)
	print("[INFO] Sample  image of standard dataset:{} is saved at {}".format(dataset,fileToSaveSampleImage))

	

	#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY) #Binary targets transform to a column vector. otherwise one hot vector
	testY = lb.fit_transform(testY)  #Binary targets transform to a column vector. otherwise one hot vector
	numOfOutputs=len(lb.classes_)
	#print(lb.classes_)






	model=modelsFactory.ModelCreator(numOfOutputs,imgWidth,imgHeight,numOfchannels,networkID).model

	model.summary()
	opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


	# train the network
	print("[INFO] training network...")
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

	


	fileToSaveModel=os.path.join("Results",dataset+"_"+networkID+".keras2")
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
	helper.plot_print_confusion_matrix(y_true, y_pred, classes=labels,dataset=dataset,title=dataset+ '_Confusion matrix, without normalization') 

	# Plot normalized confusion matrix
	#plot_confusion_matrix(y_true, y_pred, classes=labels, dataset=dataset,normalize=True, title=dataset+'_Normalized confusion matrix')

	#plt.show()


