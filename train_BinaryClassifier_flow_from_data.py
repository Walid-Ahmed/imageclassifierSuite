#Baseline for code from https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

# USAGE
# python train_network.py 



import tensorflow as tf


from  util import  plotUtil
from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from  util import  plotUtil


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

root_dir="datasets"
datasetDir='santa'
base_dir = os.path.join(root_dir,datasetDir)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

folders=get_immediate_subdirectories(os.path.join(root_dir,datasetDir))


train_label1_dir=os.path.join(root_dir,datasetDir,folders[0])
train_label2_dir=os.path.join(root_dir,datasetDir,folders[1])
plotUtil.drarwGridOfImages(train_label1_dir,train_label2_dir)

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(base_dir)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "santa" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors

#trainY = to_categorical(trainY, num_classes=2)
#testY = to_categorical(testY, num_classes=2)
#print(testY)
#exit()

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
#model = LeNet.build(width=28, height=28, depth=3, classes=1)
model=modelsFactory.ModelCreator("LenetModel").model

model.summary()


opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
fileNameToSaveModel="{}_{}_binaryClassifier.keras2".format(folders[0],folders[1])
fileNameToSaveModel=os.path.join("Results",fileNameToSaveModel)
model.save(fileNameToSaveModel)
print("[INFO] Model saved  to file {}".format(fileNameToSaveModel))



plotUtil.plotAccuracyAndLossesonSameCurve(history)
plotUtil.plotAccuracyAndLossesonSDifferentCurves(history)

