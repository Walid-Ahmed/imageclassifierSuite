
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
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle


def plot_print_confusion_matrix(y_true, y_pred, classes,dataset,
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

        print_cm(cm,classes)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
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
    fileToSaveConfusionMatrix=os.path.join("Results",dataset+'_ConfusionMatrix.png')
    plt.savefig(fileToSaveConfusionMatrix)
    print("[INFO] Confusion matrix   saved to {}".format(fileToSaveConfusionMatrix))

    plt.show()


    return ax







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
	print(lb.classes_)


	fileNameToSaveLabels="CIFAR10"+"_labels.pkl"
	fileNameToSaveLabels=os.path.join("Results",fileNameToSaveLabels)
	f = open(fileNameToSaveLabels, "wb")
	labeles_dictionary=dict()	
	outInt=0
	for item in labels:
		labeles_dictionary[item]=outInt
		outInt=outInt+1
	f.write(pickle.dumps(labeles_dictionary))   #['not_santa' 'santa']
	f.close()
	print("[INFO] Labels saved  to file {} as {}".format(fileNameToSaveLabels,labeles_dictionary))
	input("[INFO] Data ready for training")


	folderNameToSaveBestModel="{}_Best_classifier".format("CIFAR10")
	folderNameToSaveBestModel=os.path.join("Results",folderNameToSaveBestModel)
	es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1 ,  patience=200)
	mc = ModelCheckpoint(folderNameToSaveBestModel, monitor='val_loss', mode='min', save_best_only=True)




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
	history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1,callbacks=[es, mc])

	pathToSaveModel=os.path.join("Results","cifar10model.h5")
	model.save(pathToSaveModel,save_format='h5')
	print("[INFO] Model saved to {}".format(pathToSaveModel))


	pathToSaveModel=os.path.join("Results","cifar10model")
	model.save(pathToSaveModel,save_format='tf')

	print("[INFO] Model saved in h5 and tf format to {}".format("Results"))
  
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
	fileToSaveLossAccCurve=os.path.join("Results","CIFAR10_"+"plot_loss_accu.png")
	print("[INFO] Loss curve saved to {}".format(fileToSaveLossAccCurve))
	plt.savefig(fileToSaveLossAccCurve)




	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	y_true=testY.argmax(axis=1)
	y_pred=predictions.argmax(axis=1)
	print(classification_report(y_true,y_pred, target_names=labels))
	print(confusion_matrix(y_true, y_pred))
	print(labels)




	plot_print_confusion_matrix(y_true, y_pred, classes=labels_,dataset="CIFAR10",title="CIFAR10"+ '_Confusion matrix, without normalization') 






