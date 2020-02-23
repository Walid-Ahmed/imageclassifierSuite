
#usage python trainCIFAR10.py


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import numpy as np









if __name__ == '__main__':



    #load cifar10 dataset
	(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()
	labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
	numPfSamples,imgWidth,imgHeight,numOfchannels=trainData.shape
	print("[INFO] Original cifar10 dataset of train Data shape {}".format(trainData.shape))
	print("[INFO] Original cifar10 dataset of train Labels shape {}".format(trainLabels.shape))
	print("[INFO] Original cifar10 dataset of test Data shape {}".format(testData.shape))
	print("[INFO] Original cifar10 dataset of test Labels shape {}".format(testLabels.shape))


    #get data ready for training
	trainX = trainData.reshape((trainData.shape[0], imgWidth,imgHeight,  numOfchannels))
	testX = testData.reshape((testData.shape[0], imgWidth,imgHeight, numOfchannels))
	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0
	trainLabels=trainLabels.astype(str)
	testLabels=testLabels.astype(str)
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainLabels)
	testY = lb.fit_transform(testLabels)
	print("[INFO] Data ready for training")




	#tuning parameters
	weight_decay = 1e-4
	INIT_LR = 1e-3
	EPOCHS=2
	BS=32


	#define model
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
	model.summary()


    #compile model
	opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


	# train the network
	print("[INFO] training network...")
	aug = ImageDataGenerator()
	history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)
	model.save("cifar10model.keras2")
	print("[INFO] Model saved to {}".format("cifar10model.keras2"))
  
	#draw training curves	
	acc      = history.history[     'accuracy' ]
	val_acc  = history.history[ 'val_accuracy' ]
	loss     = history.history[    'loss' ]
	val_loss = history.history['val_loss' ]   
	epochs   = range(len(acc)) # Get number of epochs
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(epochs, loss, label="train_loss")
	plt.plot(epochs, val_loss, label="val_loss")
	plt.plot(epochs, acc, label="train_acc")
	plt.plot(epochs, val_acc, label="val_acc")
	plt.title("CIFAR10 Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.show()




	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	y_true=testY.argmax(axis=1)
	y_pred=predictions.argmax(axis=1)
	print(classification_report(y_true,y_pred, target_names=labels))
	print(confusion_matrix(y_true, y_pred))
	print(labels)





