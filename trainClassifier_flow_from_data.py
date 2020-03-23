#Baseline for code from https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/. 



# USAGE
#python trainClassifier_flow_from_data.py    --EPOCHS 15   --width 28 --height 28  --datasetDir Santa --networkID LenetModel --verbose False --ResultsFolder  Results/r2_santa --augmentationLevel 2 --useOneNeuronForBinaryClassification True   --opt Adam
#python trainClassifier_flow_from_data.py    --EPOCHS 15   --width 28 --height 28  --datasetDir Santa --networkID LenetModel --verbose False --ResultsFolder  Results/r2_santa --augmentationLevel 2 --useOneNeuronForBinaryClassification False   --opt Adam
#python trainClassifier_flow_from_data.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 25  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r29_FacialExpression   --augmentationLevel 1 --opt Adam
#python trainClassifier_flow_from_data.py  --datasetDir Cyclone_Wildfire_Flood_Earthquake_Database --networkID Resnet50  --EPOCHS 25  --width  224 --height  224  --BS 32  --ResultsFolder  Results/r22_Cyclone_Wildfire_Flood_Earthquake_Database  --augmentationLevel 1 --opt Adam





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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import shutil

from util import paths
from tensorflow.keras.utils import plot_model
from scipy.ndimage import imread
from callbacks  import  TrainingMonitor
from callbacks   import EpochCheckpoint


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical




import matplotlib.pyplot as plt


def changeStringLabelToInt(labels,labeles_dictionary):
	intLabels=[]
	for label in labels:
		intLabels.append(labeles_dictionary[label])


	labels=	intLabels
	return labels


def calculatePrecisionRecall(probs,y_true,y_pred,labels,ResultsFolder):
		print("[INFO] Evaluating  Precision-Recall curve")

		precision, recall, thresholds = precision_recall_curve(y_true, probs) #y_score    probabilities between 0 and 1
		average_precision = average_precision_score(y_true, probs)
		precision_value=precision_score(y_true, y_pred, average='macro')  
		print("[INFO] precision_value at threshold 0.5=".format(precision_value) )


		plt.step(recall, precision, color='b', alpha=0.2, where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title(' Precision-Recall curve for class {0}'.format(labels[1] +" vs " + labels[0]))
		fileName="Precision_Recall_curve_"+labels[1]+".png"
		fileName=os.path.join(ResultsFolder,fileName)

		plt.savefig(fileName)
		print("[INFO] Precision_Recall_curve_  plot is saved to {}" .format(fileName) )

		plt.show()
		return precision, recall, thresholds



#aim to get Reproducible Results with Keras
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

print("[INFO] tf.Keras   version is {}".format( tf.keras.__version__)) 




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()



if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetDir", required=True, help="path to dataset directory with train and validation images")
    ap.add_argument("--testDir", default=None, help="path to test directory with test images")
    ap.add_argument("--networkID", required=True, help="I.D. of the network, it can be any of [net1,net2,net3,net4,net5,LenetModel,Resnet50,net3,MiniVGG,VGG16]")
    ap.add_argument("--EPOCHS", required=False, type=int, default=25, help="Number of maximum epochs to train")
    ap.add_argument("--BS", required=False, default=16 , type=int, help="Batch size")
    ap.add_argument("--width", required=True, help="width of image")
    ap.add_argument("--height", required=True, help="height of image")
    ap.add_argument("--patience", required=False, default=50, type=int,help="Number of epochs to wait without accuracy improvment")
    ap.add_argument("--ResultsFolder", required=False, default="Results",help="Folder to save Results")
    ap.add_argument("--lr", required=False, type=float, default=0.001,help="Initial Learning rate")
    ap.add_argument("--new_lr", required=False, type=float, default=1e-4,help="restarting Learning rate")
    ap.add_argument("--labelSmoothing", type=float, default=0, help="turn on label smoothing")
    #ap.add_argument("--applyAugmentation",  default="False",help="turn on apply Augmentation")
    ap.add_argument("--continueTraining",  default="False",help="continue training a previous trained model")
    ap.add_argument("--modelcheckpoint", type=str, default=None ,help="path to *specific* model checkpoint to load")
    ap.add_argument("--startepoch", type=int, default=0, help="epoch to restart training at")
    ap.add_argument("--saveEpochRate", type=int, default=5, help="Frequency to save checkpoints")
    ap.add_argument("--opt", type=str, default="SGD", help="Type of optimizer")
    ap.add_argument("--augmentationLevel", type=int, default=0,help="turn on  Augmentation")
    ap.add_argument("--useOneNeuronForBinaryClassification", type=str, default="True",help="turn on  Augmentation")
    ap.add_argument("--display", type=str, default="True",help="turn on/off  display of plots")





    ap.add_argument("--verbose", default="True",type=str,help="Print extra data")




    #read the arguments
    args = vars(ap.parse_args())
    datasetDir=args["datasetDir"]
    networkID=args["networkID"]
    EPOCHS=args["EPOCHS"]
    width=int(args["width"])
    height=int(args["height"])
    testDir=args["testDir"]
    BS=args["BS"]
    patience=args["patience"]
    ResultsFolder=args["ResultsFolder"]
    learningRate=args["lr"]
    new_lr=args["new_lr"]
    labelSmoothing=args["labelSmoothing"]
    continueTraining=args["continueTraining"]
    modelcheckpoint=args["modelcheckpoint"]    
    startepoch=args["startepoch"]
    saveEpochRate=args['saveEpochRate']
    opt=args['opt']
    augmentationLevel=args["augmentationLevel"]
    useOneNeuronForBinaryClassification=args['useOneNeuronForBinaryClassification']
    display=args['display']


    verbose=args["verbose"]

    if (verbose=="True"):
    	verbose=True
    else:
    	verbose=False



    if(continueTraining=="True") or  (continueTraining=="true"):
	        continueTraining=True
    else:
    		continueTraining=False
        


    if(useOneNeuronForBinaryClassification=="True") :
        useOneNeuronForBinaryClassification=True
    else:
        useOneNeuronForBinaryClassification=False


    if(display=="True") :
        display=True
    else:
        display=False



    if(augmentationLevel==2):

        train_datagen = ImageDataGenerator(
              rescale=1./255,   #All images will be rescaled by 1./255
              rotation_range=40,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest')
   
    elif(augmentationLevel==1):
        train_datagen = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest",
        rescale=1./255, )

    else:    
        train_datagen = ImageDataGenerator(
                  rescale=1./255,   #All images will be rescaled by 1./255
                  )

    test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

	 


    if os.path.exists(ResultsFolder):
        print("[Warning]  Folder {} aready exists, All files in folder will be deleted".format(ResultsFolder))
        input("[msg]  Press any key to continue")
        shutil.rmtree(ResultsFolder)
    os.mkdir(ResultsFolder)	


    folderNameToSaveBestModel="{}_Best_classifier".format(datasetDir)
    folderNameToSaveBestModel=os.path.join(ResultsFolder,folderNameToSaveBestModel)
    folderNameToSaveModelCheckPoints=os.path.join(ResultsFolder,"checkPoints")
    os.mkdir(folderNameToSaveModelCheckPoints)
    plotPath=os.path.join(ResultsFolder,"onlineLossAccPlot.png")
    jsonPath=os.path.join(ResultsFolder,"history.json")

    os.mkdir(folderNameToSaveBestModel)	
    fileNameToSaveBestModel=os.path.join(folderNameToSaveBestModel,"best_classifier_"+datasetDir+".h5")


    earlyStopping = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0 ,  patience=patience , verbose=1)
    modelCheckpoint = ModelCheckpoint(fileNameToSaveBestModel, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    tensorboard_callback = TensorBoard(log_dir=ResultsFolder)
    epochCheckpoint=EpochCheckpoint(folderNameToSaveModelCheckPoints, every=saveEpochRate,startAt=startepoch)
    trainingMonitor=TrainingMonitor(plotPath,jsonPath=jsonPath,startAt=startepoch)


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
info,channels=plotUtil.drarwGridOfImages(base_dir,fileNameToSaveImage=fileToSaveSampleImage,display=display)

paths.getTrainStatistics2(base_dir)

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(base_dir)))
random.seed(42)
random.shuffle(imagePaths)


firstImage=imagePaths[0]
firstImage= imread(firstImage)
channels=len(firstImage.shape)
if (channels==2):
    channels=1

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





data = np.array(data, dtype="float")

labels = np.array(labels)



#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
lb = LabelBinarizer() 
#labels = lb.fit_transform(labels)  #Binary targets transform to a column vector. otherwise one hot vector, you can also use from keras.utils.to_categorical to  perform one-hot encoding on the labels even if there are only 2 classes but deal with it as categorical
lb.fit_transform(labels)  #Binary targets transform to a column vector. otherwise one hot vector, you can also use from keras.utils.to_categorical to  perform one-hot encoding on the labels even if there are only 2 classes but deal with it as categorical

numOfOutputs=len(lb.classes_)



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


print("[INFO] Training with the following labels".format(lb.classes_))
print("__________________________________________________________________________________________________________")

# convert the labels from integers to vectors

print("[INFO] Training with the following {} classes {}".format(numOfOutputs ,lb.classes_ ))   


targetNames=lb.classes_


if (numOfOutputs==2) and useOneNeuronForBinaryClassification:  #Binary problem
	numOfOutputs=1  # use only 1 neuron in last layer
	lossFun=BinaryCrossentropy(label_smoothing=labelSmoothing)
	labels = lb.fit_transform(labels)  #Binary targets transform to a column vector. otherwise one hot vector, you can also use from keras.utils.to_categorical to  perform one-hot encoding on the labels even if there are only 2 classes but deal with it as categorical

else:
	labels=changeStringLabelToInt(labels,labeles_dictionary)  # change labels from String to int
	labels = to_categorical(labels, num_classes=numOfOutputs)    #np_utils.to_categorical takes y of datatype int
	
	#labels = lb.fit_transform(labels)  #Binary targets transform to a column vector. otherwise one hot vector, you can also use from keras.utils.to_categorical to  perform one-hot encoding on the labels even if there are only 2 classes but deal with it as categorical
	
	lossFun = CategoricalCrossentropy(label_smoothing=labelSmoothing) 






print("[INFO] lossFun is  {} ".format(lossFun))















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




#setup optimizer
if (opt=="RMSprop"):
	opt=RMSprop(learning_rate=learningRate, rho=0.9)
elif(opt=="Adam"):
	opt=Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, amsgrad=False,decay=learningRate / EPOCHS)
elif(opt=="SGD"):
	opt=SGD(learning_rate=learningRate, momentum=0.0, nesterov=False)



model.compile(loss=lossFun, optimizer=opt, metrics=["accuracy"])





fileToSaveModelPlot=os.path.join(ResultsFolder,'model.png')
plot_model(model, to_file=fileToSaveModelPlot,show_shapes="True")
print("[INFO] Model plot  saved to file  {} ".format(fileToSaveModelPlot))

print("[INFO] training network...")
history = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS),
	validation_data=test_datagen.flow(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, 
    callbacks=[earlyStopping, modelCheckpoint,tensorboard_callback,trainingMonitor,epochCheckpoint]

	)




# save the model to disk in tf 
folderNameToSaveModel="{}_Classifier".format(datasetDir)
folderNameToSaveModel=os.path.join(ResultsFolder,folderNameToSaveModel)
model.save(folderNameToSaveModel,save_format='tf') #model is saved in TF2 format (default)



fileNameToSaveModel="{}_Classifier.h5".format(datasetDir)
fileNameToSaveModel=os.path.join(folderNameToSaveModel,fileNameToSaveModel)
model.save(fileNameToSaveModel,save_format='h5') #model is saved in h5 format






# evaluate the network
print("[INFO] evaluating network...")
testX = testX.astype("float32") / 255.0

probs = model.predict(testX, batch_size=32)
if (numOfOutputs==1):
	y_true=testY
	y_pred=predictBinaryValue(probs)
else:	
	y_true=testY.argmax(axis=1)
	y_pred=probs.argmax(axis=1)


#print classfication report
print(classification_report(y_true,y_pred, target_names=targetNames))
print("*************************************************************************************************************")      
#exit()

#cm=confusion_matrix(y_true, y_pred)
#helper.print_cm(cm,lb.classes_)




#plot and save training curves 
title=datasetDir

fileToSaveLossAccCurve=os.path.join(ResultsFolder,title+"plot_loss_accu.png")
plotUtil.plotAccuracyAndLossesonSameCurve(history,title,fileToSaveLossAccCurve,display=display)

fileToSaveAccuracyCurve=os.path.join(ResultsFolder,title+"plot_acc.png")
fileToSaveLossCurve=os.path.join("Results",title+"plot_loss.png")
plotUtil.plotAccuracyAndLossesonSDifferentCurves(history,title,fileToSaveAccuracyCurve,fileToSaveLossCurve,display=display)






# Plot non-normalized confusion matrix
helper.plot_print_confusion_matrix(y_true, y_pred, ResultsFolder,classes=targetNames,dataset=datasetDir,title=datasetDir+ '_Confusion matrix, without normalization') 

#calculate precision recall
if(numOfOutputs==1):  #binary classification
	calculatePrecisionRecall(probs,y_true,y_pred,targetNames,ResultsFolder)


print("[INFO] Loss and accuracy  curve saved to {}".format(fileToSaveLossAccCurve))
print("[INFO] Loss curve saved to {}".format(fileToSaveLossCurve))
print("[INFO] Accuracy  curve saved to {}".format(fileToSaveAccuracyCurve))
print("[INFO] Best Model saved   as h5 file:  {}".format(fileNameToSaveBestModel))
print("[INFO] Model check points saved to folder  {}  each  {} epochs ".format(folderNameToSaveModelCheckPoints,saveEpochRate))
print("[INFO] Final model saved  to folder {} in both .h5 and TF2 format".format(folderNameToSaveModel))
print("[INFO] Sample images from dataset saved to file  {} ".format(fileToSaveSampleImage))
print("[INFO] History of loss and accuracy  saved to file  {} ".format(jsonPath))
print("*************************************************************************************************************")      







