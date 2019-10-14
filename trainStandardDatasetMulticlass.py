
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



import numpy as np

import pickle
import os
from imutils import build_montages


def plot_confusion_matrix(y_true, y_pred, classes,dataset,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fileToSaveConfusionMatrix=os.path.join("Results",dataset+'_Confusion Matrix.png')
    plt.savefig(fileToSaveConfusionMatrix)
    print("[INFO] Confusion matrix   saved to {}".format(fileNameToSaveLabels))

    plt.show()
    plt.savefig('books_read.png')


    return ax


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
		(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
		labels =  ['0','1','2','3','4','5','6','7','8','9']




	if(dataset=="CIFAR10"):
		(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()
		labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



	if(dataset=="CIFAR100"):
		(trainData, trainLabels), (testData, testLabels) = cifar100.load_data()
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
		(trainData, trainLabels), (testData, testLabels) = fashion_mnist.load_data()
		labels = ["top", "trouser", "pullover", "dress", "coat","sandal", "shirt", "sneaker", "bag", "ankle boot"]


	#get original img width , hight and number of channels
	try:
		numPfSamples,imgWidth,imgHeight,numOfchannels=trainData.shape

	except:
		numPfSamples,imgWidth,imgHeight=trainData.shape
		numOfchannels=1


	print("[INFO] Original {} dataset of trainData shape {}".format(dataset,trainData.shape))
	print("[INFO] Original {} dataset of trainLabels shape {}".format(dataset,trainLabels.shape))
	print("[INFO] Original {} dataset of testData shape {}".format(dataset,testData.shape))
	print("[INFO] Original {} dataset of testLabels shape {}".format(dataset,testLabels.shape))


	trainX = trainData.reshape((trainData.shape[0], imgWidth,imgHeight,  numOfchannels))
	testX = testData.reshape((testData.shape[0], imgWidth,imgHeight, numOfchannels))
	images=[]
	for  i in range(9):
		image=trainData[i]
		if(numOfchannels==1):
			image = cv2.merge([image] * 3)
		else:
			image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)	
		images.append(image)
	
	montage = build_montages(images, (128, 128), (3, 3))
	cv2.imshow("Sample images from {} training dataset".format(dataset), montage[0])
	sampleImage=montage[0]
	fileToSaveSampleImage=os.path.join("Results","sample_"+dataset+".png")
	cv2.imwrite(fileToSaveSampleImage,sampleImage)
	print("[INFO] Sample  image of standard dataset:{} is saved at {}".format(dataset,fileToSaveSampleImage))
	print("[INFO] Press anykey to start training")

	cv2.waitKey(0)

	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0


	# transform the training and testing labels into vectors in the
	# range [0, classes] -- this generates a vector for each label,
	# where the index of the label is set to `1` and all other entries
	# to `0`; in the case of MNIST, there are 10 class labels
	


	trainLabels=trainLabels.astype(str)
	testLabels=testLabels.astype(str)



	#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainLabels) #Binary targets transform to a column vector. otherwise one hot vector
	testY = lb.fit_transform(testLabels)  #Binary targets transform to a column vector. otherwise one hot vector


	numOfOutputs=len(lb.classes_)
	#print(lb.classes_)






	model=modelsFactory.ModelCreator(numOfOutputs,imgWidth,imgHeight,numOfchannels,networkID).model

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
	plot_confusion_matrix(y_true, y_pred, classes=labels,dataset=dataset,title=dataset+ '_Confusion matrix, without normalization') 

	# Plot normalized confusion matrix
	#plot_confusion_matrix(y_true, y_pred, classes=labels, dataset=dataset,normalize=True, title=dataset+'_Normalized confusion matrix')

	#plt.show()


