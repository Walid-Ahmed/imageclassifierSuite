#Baseline for code from https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/. 



# USAGE
# python trainClassifier_flow_from_large_data.py    --EPOCHS 25   --width 28 --height 28 --channels 3 --datasetDir Santa --networkID LenetModel
# python trainClassifier_flow_from_large_data.py    --EPOCHS 25   --width 224 --height 224 --channels 3  --datasetDir SportsClassification --networkID Resnet50

# python trainClassifier_flow_from_large_data.py    --EPOCHS 25   --width 64 --height 64 --channels 1 --datasetDir SMILES --networkID LenetModel


# import the necessary packages
import tensorflow as tf
import pickle
from  util import  plotUtil
from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from  util import  plotUtil  
from  util import  helper  
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def data_generator(imagePaths, bs, lb, mode="train", aug=None):

	# loop indefinitely
	while True:
		# initialize our batches of images and labels


		# keep looping until we reach our batch size
		for imagePath in imagePaths:

			images = []
			labels = []		

			while len(images) < bs:
			# loop over the input images

				# extract the class label from the image path and update the labels list
				label = imagePath.split(os.path.sep)[-2]


				# load the image, pre-process it, and store it in the data list
				print("[INFO] Reading image from path: {}".format(imagePath))
				
				if(channels==3):
					image = cv2.imread(imagePath)
				else:
					image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
				
				image = cv2.resize(image, (width,height))
				image = img_to_array(image)	
				labels.append(label)
				#print(labels)
				data.append(image)

			# update our corresponding batches lists
			images.append(image)
			labels.append(label)

		# one-hot encode the labels
		labels = lb.transform(np.array(labels))
		# scale the raw pixel intensities to the range [0, 1]
		images = np.array(images, dtype="float") / 255.0


		# yield the batch to the calling function
		yield (np.array(images), labels)


def predictBinaryValue(probs,threshold=0.5):
	y_pred=[]
	for prob in probs:
		if (prob>threshold):
			y_pred.append(1)
		else:	
			y_pred.append(0)
	return np.asarray(y_pred)



if __name__ == "__main__":



	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()

	ap.add_argument("--width",  required=True,help="image width")
	ap.add_argument("--height",  required=True,help="image height")
	ap.add_argument("--EPOCHS",  required=True,help="number of epochs to train")
	ap.add_argument("--datasetDir",  required=True,help="path to dataset directory")
	ap.add_argument("--networkID", required=True, help="I.D. of the network")
	ap.add_argument("--channels", default=3,type=int,help="Number of channels in image")




	args = vars(ap.parse_args())

	width=int(args["width"])
	height=int(args["height"])
	EPOCHS =int(args["EPOCHS"])
	datasetDir=args["datasetDir"]
	networkID=args["networkID"]
	channels=args["channels"]





	root_dir="datasets"
	base_dir = os.path.join(root_dir,datasetDir)







	# initial learning rate, and batch size

	INIT_LR = 1e-3
	BS = 32

	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []
	train_labels_dir=[]
	folders=get_immediate_subdirectories(os.path.join(root_dir,datasetDir))
	

	#print(len(folders))
	#exit()




	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images(base_dir)))
	random.seed(42)
	random.shuffle(imagePaths)
	NUM_TRAIN_IMAGES=len(imagePaths)
	NUM_TEST_IMAGES=len(imagePaths)



	lb = LabelBinarizer() 
	trainnGen=data_generator(imagePaths,BS,lb)
	testGen=data_generator(imagePaths,BS,lb)

	print(imagePaths)
	exit()

 





	numOfOutputs=len(folders)

	if (numOfOutputs==2):  #Binary problem
		numOfOutputs=1  # use only 1 neuron in last layer

	#print("[INFO] Training with the following {} classes {}".format(numOfOutputs ,lb.classes_ ))   










	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	# initialize the model
	print("[INFO] compiling model...")
	#model = LeNet.build(width=28, height=28, depth=3, classes=1)
	#model=modelsFactory.ModelCreator(numOfOutputs,imgWidth,imgHeight,"Resnet50").model
	model=modelsFactory.ModelCreator(numOfOutputs,width,height,channels=channels,networkID=networkID).model
	model.summary()


	opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	if(numOfOutputs==1):
		model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
	else:
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	# train the network
	input("[MSG] Press enter to start training")

	print("[INFO] training network...")
	#history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)

	history = model.fit_generator(data_generator,steps_per_epoch=NUM_TRAIN_IMAGES // BS, validation_data=testGen,validation_steps=NUM_TEST_IMAGES // BS, epochs=EPOCHS)

	# save the model to disk
	fileNameToSaveModel="{}_binaryClassifier.keras2".format(datasetDir)
	fileNameToSaveModel=os.path.join("Results",fileNameToSaveModel)
	model.save(fileNameToSaveModel)
	print("[INFO] Model saved  to file {}".format(fileNameToSaveModel))

	# serialize the label binarizer to disk
	fileNameToSaveLabels=datasetDir+"_labels.pkl"
	fileNameToSaveLabels=os.path.join("Results",fileNameToSaveLabels)
	f = open(fileNameToSaveLabels, "wb")
	f.write(pickle.dumps(lb.classes_))
	f.close()
	print("[INFO] Labels saved  to file {}".format(fileNameToSaveLabels))


	if(numOfOutputs==1):
		labelsDict=dict()
		labelsDict[lb.classes_[0]]=0
		labelsDict[lb.classes_[1]]=1
		print(labelsDict)


	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	if (numOfOutputs==1):
		y_true=testY
		y_pred=predictBinaryValue(predictions)
	else:	
		y_true=testY.argmax(axis=1)
		y_pred=predictions.argmax(axis=1)


	print(classification_report(y_true,y_pred, target_names=lb.classes_))
	cm=confusion_matrix(y_true, y_pred)
	helper.print_cm(cm,lb.classes_)




	plotUtil.plotAccuracyAndLossesonSameCurve(history)
	plotUtil.plotAccuracyAndLossesonSDifferentCurves(history)


	# Plot non-normalized confusion matrix
	helper.plot_print_confusion_matrix(y_true, y_pred, classes=lb.classes_,dataset=datasetDir,title=datasetDir+ '_Confusion matrix, without normalization') 

