#Baseline for code from https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/. 



# USAGE
# python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 28 --height 28 --channels 3 --datasetDir Santa --networkID LenetModel --verbose False --ResultsFolder  Results/r2_santa --applyAugmentation True
# python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 224 --height 224 --channels 3  --datasetDir SportsClassification --networkID Resnet50 --verbose False 
# python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 64 --height 64 --channels 1 --datasetDir SMILES --networkID LenetModel --verbose False
# python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 28 --height 28 --channels 3 --datasetDir NIHmalaria --networkID LenetModel --verbose False

# python trainClassifier_flow_from_data.py    --EPOCHS 200   --width 48 --height 48 --channels 1 --datasetDir FacialExpression  --networkID net2 --verbose False --ResultsFolder  Results/r2_faceExp



# import the necessary packages
import tensorflow as tf
import pickle
from  util import  plotUtil
from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from  util import  helper  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import shutil

from util import paths
from tensorflow.keras.utils import plot_model



print(tf.keras.__version__) #2.2.4-tf





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()



ap.add_argument("--datasetDir", required=True, help="path to dataset directory with train and validation images")
ap.add_argument("--testDir", default=None, help="path to test directory with test images")
ap.add_argument("--networkID", required=True, help="I.D. of the network")
ap.add_argument("--EPOCHS", required=False, type=int, default=25, help="Number of maximum epochs to train")
ap.add_argument("--BS", required=False, default=16 , type=int, help="Batch size")
ap.add_argument("--width", required=True, help="width of image")
ap.add_argument("--height", required=True, help="height of image")
ap.add_argument("--patience", required=False, default=50, type=int,help="Number of epochs to wait without accuracy improvment")
ap.add_argument("--ResultsFolder", required=False, default="Results",help="Folder to save Results")
ap.add_argument("--lr", required=False, type=float, default=0.001,help="Initial Learning rate")
ap.add_argument("--channels", default=3,type=int,help="Number of channels in image")
ap.add_argument("--labelSmoothing", type=float, default=0, help="turn on label smoothing")
ap.add_argument("--applyAugmentation",  default="False",help="turn on apply Augmentation")
ap.add_argument("--continueTraining",  default="False",help="continue training a previous trained model")

ap.add_argument("--verbose", default="True",type=str,help="Print extra data")






args = vars(ap.parse_args())

width=int(args["width"])
height=int(args["height"])
EPOCHS =int(args["EPOCHS"])
datasetDir=args["datasetDir"]
networkID=args["networkID"]
channels=args["channels"]
patience=args["patience"]
testDir=args["testDir"]


verbose=args["verbose"]
ResultsFolder=args['ResultsFolder']
applyAugmentation=args["applyAugmentation"]
continueTraining=args["continueTraining"]




if (verbose=="True"):
	verbose=True
else:
	verbose=False






if(applyAugmentation=="True") or  (applyAugmentation=="True"):
        applyAugmentation=True
else:
        applyAugmentation=False

if(continueTraining=="True") or  (continueTraining=="True"):
        continueTraining=True
else:
        continueTraining=False
        


if(applyAugmentation):

        train_datagen = ImageDataGenerator(
              rescale=1./255,   #All images will be rescaled by 1./255
              rotation_range=40,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest')
else:
        train_datagen = ImageDataGenerator(
                  rescale=1./255,   #All images will be rescaled by 1./255
                  )

test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

 


if os.path.exists(ResultsFolder):
	print("[Warning]  Folder aready exists, All files in folder will be deleted")
	input("[msg]  Press any key to continue")
	shutil.rmtree(ResultsFolder)
os.mkdir(ResultsFolder)	


folderNameToSaveBestModel="{}_Best_classifier".format(datasetDir)
folderNameToSaveBestModel=os.path.join(ResultsFolder,folderNameToSaveBestModel)

es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0 ,  patience=patience , verbose=1)
mc = ModelCheckpoint(folderNameToSaveBestModel, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
tensorboard_callback = TensorBoard(log_dir=ResultsFolder)



def predictBinaryValue(probs,threshold=0.5):
	y_pred=[]
	for prob in probs:
		if (prob>threshold):
			y_pred.append(1)
		else:	
			y_pred.append(0)
	return np.asarray(y_pred)






def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]




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





fileToSaveSampleImage=os.path.join(ResultsFolder,"sample_"+datasetDir+".png")
plotUtil.drarwGridOfImages(base_dir,fileNameToSaveImage=fileToSaveSampleImage,channels=int(channels))

paths.getTrainStatistics2(base_dir)

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(base_dir)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:

	# extract the class label from the image path and update the labels list
	label = imagePath.split(os.path.sep)[-2]


	# load the image, pre-process it, and store it in the data list
	if(verbose):
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


# scale the raw pixel intensities to the range [0, 1]



#data = np.array(data, dtype="float") / 255.0
data = np.array(data, dtype="float")

labels = np.array(labels)
print(data.shape)  #(922, 28, 28, 3)



#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
lb = LabelBinarizer() 
labels = lb.fit_transform(labels)  #Binary targets transform to a column vector. otherwise one hot vector, you can also use from keras.utils.to_categorical to  perform one-hot encoding on the labels
numOfOutputs=len(lb.classes_)
#print(lb.classes_)
print("__________________________________________________________________________________________________________")


print("[INFO] Training with the following {} classes {}".format(numOfOutputs ,lb.classes_ ))   



if (numOfOutputs==2):  #Binary problem
	numOfOutputs=1  # use only 1 neuron in last layer



fileNameToSaveLabels=datasetDir+"_labels.pkl"
fileNameToSaveLabels=os.path.join(ResultsFolder,fileNameToSaveLabels)
f = open(fileNameToSaveLabels, "wb")
labeles_dictionary=dict()
outInt=0
for item in lb.classes_:
	labeles_dictionary[item]=outInt
	outInt=outInt+1

f.write(pickle.dumps(labeles_dictionary))   #['not_santa' 'santa']
f.close()
print("[INFO] Labels saved  to file {} as {}".format(fileNameToSaveLabels,labeles_dictionary))









# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
numPfSamples,imgWidth,imgHeight,numOfchannels=trainX.shape
    
print("__________________________________________________________________________________________________________")
print("[INFO] Dataset of train Data shape {}".format(trainX.shape))
print("[INFO] Dataset of train Labels shape {}".format(testX.shape))
print("[INFO] Dataset of test Data shape {}".format(trainY.shape))
print("[INFO] Dataset of test Labels shape {}".format(testY.shape))
print("__________________________________________________________________________________________________________")


# construct the image generator for data augmentation


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




fileToSaveModelPlot=os.path.join(ResultsFolder,'model.png')
plot_model(model, to_file=fileToSaveModelPlot,show_shapes="True")
print("[INFO] Model plot  saved to file  {} ".format(fileToSaveModelPlot))

print("[INFO] training network...")
history = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS),
	validation_data=test_datagen.flow(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[es, mc,tensorboard_callback])

for key in history.history:
	print(key)


# save the model to disk
folderNameToSaveModel="{}_Classifier".format(datasetDir)
folderNameToSaveModel=os.path.join(ResultsFolder,folderNameToSaveModel)
model.save(folderNameToSaveModel,save_format='tf') #model is saved in TF2 format (default)



fileNameToSaveModel="{}_Classifier.h5".format(datasetDir)
fileNameToSaveModel=os.path.join(folderNameToSaveModel,fileNameToSaveModel)
model.save(fileNameToSaveModel,save_format='h5') #model is saved in h5 format






# evaluate the network
print("[INFO] evaluating network...")
testX = testX.astype("float32") / 255.0

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




#plot and save training curves 
info1=plotUtil.plotAccuracyAndLossesonSameCurve(history,ResultsFolder)
info2=plotUtil.plotAccuracyAndLossesonSDifferentCurves(history,ResultsFolder)
print("*************************************************************************************************************")      
print(info1)
print(info2)
print("*************************************************************************************************************")  




# Plot non-normalized confusion matrix
helper.plot_print_confusion_matrix(y_true, y_pred, ResultsFolder,classes=lb.classes_,dataset=datasetDir,title=datasetDir+ '_Confusion matrix, without normalization') 
print("[INFO] Model saved  to folder {} in both .h5 and TF2 format".format(folderNameToSaveModel))
print("[INFO] Best Model saved  to folder {}".format(folderNameToSaveBestModel))







