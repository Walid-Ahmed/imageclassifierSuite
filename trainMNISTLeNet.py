
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.datasets import mnist


from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from  util import  plotUtil
from sklearn.preprocessing import LabelBinarizer





if __name__ == '__main__':

	numOfOutputs=10
	imgWidth,imgHeight=28,28
	EPOCHS=25
	BS=32
	INIT_LR = 1e-3


	print("[INFO] downloading MNIST...")
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	trainData = trainData.reshape((trainData.shape[0], imgWidth,imgHeight, 1))
	testData = testData.reshape((testData.shape[0], imgWidth,imgHeight, 1))

	# scale data to the range of [0, 1]
	trainX = trainData.astype("float32") / 255.0
	testX = testData.astype("float32") / 255.0


	# transform the training and testing labels into vectors in the
	# range [0, classes] -- this generates a vector for each label,
	# where the index of the label is set to `1` and all other entries
	# to `0`; in the case of MNIST, there are 10 class labels
	
	#trainY = np_utils.to_categorical(trainLabels, 10)
	#testY = np_utils.to_categorical(testLabels, 10)



	#lb.classes_  will be  the labels with the same order in one hot vector--->. label = lb.classes_[i]
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainLabels)
	testY = lb.fit_transform(testLabels)
	numOfOutputs=len(lb.classes_)



	model=modelsFactory.ModelCreator(numOfOutputs,imgWidth,imgHeight,1,"LenetModel").model

	model.summary()
	opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


	# train the network
	print("[INFO] training network...")
	#model.fit(trainX, trainY, batch_size=128, epochs=20,verbose=1)

	aug = ImageDataGenerator()


	history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)


	# save the model to disk
	fileNameToSaveModel="MNIST_Classifier.keras2"
	fileNameToSaveModel=os.path.join("Results",fileNameToSaveModel)
	model.save(fileNameToSaveModel)
	print("[INFO] Model saved  to file {}".format(fileNameToSaveModel))

	# serialize the label binarizer to disk
	fileNameToSaveLabels="MNIST_labels.pkl"
	fileNameToSaveLabels=os.path.join("Results",fileNameToSaveLabels)
	f = open(fileNameToSaveLabels, "wb")
	f.write(pickle.dumps(lb.classes_))
	f.close()


	print("[INFO] Labels saved  to file {}".format(fileNameToSaveLabels))


	plotUtil.plotAccuracyAndLossesonSameCurve(history)
	plotUtil.plotAccuracyAndLossesonSDifferentCurves(history)

