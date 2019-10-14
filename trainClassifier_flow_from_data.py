#Baseline for code from https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/. 



# USAGE
# python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 28 --height 28 --datasetDir Santa --networkID LenetModel
# python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 224 --height 224 --datasetDir SportsClassification --networkID LenetModel




import tensorflow as tf

import pickle
from  util import  plotUtil
from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from  util import  plotUtil  
from  util import  helper  



# import the necessary packages
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


root_dir="datasets"






# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("--width",  required=True,help="path to label list as pickle file")
ap.add_argument("--height",  required=True,help="path to label list as pickle file")
ap.add_argument("--EPOCHS",  required=True,help="path to label list as pickle file")
ap.add_argument("--datasetDir",  required=True,help="path to label list as pickle file")
ap.add_argument("--networkID", required=True, help="I.D. of the network")


args = vars(ap.parse_args())

width=int(args["width"])
height=int(args["height"])
EPOCHS =int(args["EPOCHS"])
datasetDir=args["datasetDir"]
networkID=args["networkID"]



def predictBinaryValue(probs,threshold=0.5):
	y_pred=[]
	for prob in probs:
		if (prob>threshold):
			y_pred.append(1)
		else:	
			y_pred.append(0)
	return np.asarray(y_pred)




base_dir = os.path.join(root_dir,datasetDir)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



# initialize the number of epochs to train for, initia learning rate,
# and batch size

INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
train_labels_dir=[]
folders=get_immediate_subdirectories(os.path.join(root_dir,datasetDir))



LABELS = set(["weight_lifting", "tennis", "football"])


#plotUtil.drarwGridOfImages(train_label1_dir,train_label2_dir)

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(base_dir)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:



	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]

	#if label not in LABELS:
		#continue


	# load the image, pre-process it, and store it in the data list
	print("[INFO] Reading image from path: {}".format(imagePath))
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (width,height))
	image = img_to_array(image)	
	labels.append(label)
	data.append(image)


# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)



#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
lb = LabelBinarizer() 
labels = lb.fit_transform(labels)  #Binary targets transform to a column vector. otherwise one hot vector
numOfOutputs=len(lb.classes_)





if (numOfOutputs==2):  #Binary problem
	numOfOutputs=1  # use only 1 neuron in last layer

print("[INFO] Training with the following {} classes {}".format(numOfOutputs ,lb.classes_ ))   #['football' 'tennis' 'weight_lifting']

#print(lb.get_params())   #{'neg_label': 0, 'pos_label': 1, 'sparse_output': False}. not very usefull



	#labels = to_categorical(labels) # convert the labels from integers to vectors
	# perform one-hot encoding on the labels








# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)



# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
#model = LeNet.build(width=28, height=28, depth=3, classes=1)
#model=modelsFactory.ModelCreator(numOfOutputs,imgWidth,imgHeight,"Resnet50").model
model=modelsFactory.ModelCreator(numOfOutputs,width,height,networkID=networkID).model


model.summary()


opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

if(numOfOutputs==1):
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# train the network
print("[INFO] training network...")
history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

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


#sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False)
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
helper.plot_confusion_matrix(y_true, y_pred, classes=labels,dataset=datasetDir,title=datasetDir+ '_Confusion matrix, without normalization') 

