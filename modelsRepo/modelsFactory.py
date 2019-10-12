import tensorflow as tf
from  tensorflow.keras.applications import  ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

class  ModelCreator:


	def __init__(self, numOfOutputs,width,height,channels=3,networkID="default"):

		#self.imgWidth,self.imgHeight=imgSize
		self.numOfOutputs=numOfOutputs
		self.imgWidth=width
		self.imgHeight=height
		self.channels=channels

		if(self.numOfOutputs>1):
			self.finalActivation='softmax'
		else:
			self.finalActivation='sigmoid'


		if (networkID)=="net1":
			self.model=self.defineNet1()
			print("[INFO]  Net1 Model created")

		elif (networkID)=="net2":
			self.imgWidth=width
			self.imgHeight=height
			self.model=self.defineNet2()
			print("[INFO]  Net2 Model created")

		elif networkID=="LenetModel":
			self.imgWidth=width
			self.imgHeight=height
			self.model=self.defineLenetModel()
			print("[INFO]  Lenet created")


		elif networkID=="Resnet50":
			self.imgWidth=width
			self.imgHeight=height
			self.model=self.defineResnet50()
			print("[INFO]  Resnet50 created")	

		elif networkID=="net3":
			self.model=self.defineNet3()
			print("[INFO]  Net3 created")		


		elif networkID=="MiniVGG":
			self.model=self.defineMiniVGG()
			print("[INFO]  MiniVGG Model created")

		elif networkID=="net4":
			self.model=self.defineNet4()
			print("[INFO]  Net4 created")	

		elif networkID=="net5":
			self.model=self.defineNet5()
			print("[INFO]  Net5 created")	

		elif networkID=="VGG16":
			self.model=self.defineVGG16()
			print("[INFO]  VGG16 created")	

	def defineLenetModel(self):   #can work with 28*28 
		model = tf.keras.models.Sequential()

		# first set of CONV => RELU => POOL layers
		model.add(tf.keras.layers.Conv2D(20, (5, 5),  activation='relu',padding="same",input_shape=(self.imgWidth,self.imgHeight, self.channels)))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(tf.keras.layers.Conv2D(50, (5, 5),   activation='relu',padding="same"))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(500,activation='relu'))

		# sigmoid classifier
		model.add(tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation))

        input("press")

		# return the constructed network architecture
		return model

	def defineNet1(self):   #suitable for HoursedVsHumanModel
		model = tf.keras.models.Sequential([
		# Note the input shape is the desired size of the image 300x300 with 3 bytes color
		# This is the first convolution
		tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(self.imgWidth,self.imgHeight, self.channels)),
		tf.keras.layers.MaxPooling2D(2, 2),
		# The second convolution
		tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		# The third convolution
		tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		# The fourth convolution
		tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		# The fifth convolution
		tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		# Flatten the results to feed into a DNN
		tf.keras.layers.Flatten(),
		# 512 neuron hidden layer
		tf.keras.layers.Dense(512, activation='relu'),
		# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
		tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation)])
		return model





	def defineNet2(self):   #suitable for catsvsdogs

		model = tf.keras.models.Sequential([
	    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
	    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(self.imgWidth,self.imgHeight, self.channels)),
	    tf.keras.layers.MaxPooling2D(2,2),
	    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
	    tf.keras.layers.MaxPooling2D(2,2), 
	    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
	    tf.keras.layers.MaxPooling2D(2,2),
	    # Flatten the results to feed into a DNN
	    tf.keras.layers.Flatten(), 
	    # 512 neuron hidden layer
	    tf.keras.layers.Dense(512, activation='relu'), 
	    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
	    tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation)  ])
		return model



	def defineResnet50(self):   #https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/
		baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=tf.keras.layers.Input(shape=(self.imgWidth,self.imgHeight, self.channels)))
		# construct the head of the model that will be placed on top of the
		# the base model
		headModel = baseModel.output
		headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
		headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
		headModel = tf.keras.layers.Dense(512, activation="relu")(headModel)
		headModel = tf.keras.layers.Dropout(0.5)(headModel)
		headModel = tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation)(headModel)

		# place the head FC model on top of the base model (this will become
		# the actual model we will train)
		model = tf.keras.models.Model(inputs=baseModel.input, outputs=headModel)
		return model


	def defineNet3(self):  #should be suitable with CIFAR10.  https://ermlab.com/en/blog/nlp/cifar-10-classification-using-keras-tutorial/
		model = tf.keras.models.Sequential()
		 
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(self.imgWidth,self.imgHeight, self.channels)))
		model.add(tf.keras.layers.Dropout(0.2))
		 
		model.add(tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu'))
		model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
		 
		model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))
		model.add(tf.keras.layers.Dropout(0.2))
		 
		model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))
		model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
		 
		model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))
		model.add(tf.keras.layers.Dropout(0.2))
		 
		model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))
		model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
		 
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dropout(0.2))
		model.add(tf.keras.layers.Dense(1024,activation='relu',kernel_constraint=tf.keras.constraints.MaxNorm(3)))
		model.add(tf.keras.layers.Dropout(0.2))
		model.add(tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation))	
		return model


	def defineMiniVGG(self): #src https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning. #Fashionmnist
		# first CONV => RELU => CONV => RELU => POOL layer set
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same",input_shape=(self.imgWidth,self.imgHeight, self.channels)))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
		model.add(tf.keras.layers.Dropout(0.25))

		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
		model.add(tf.keras.layers.Dropout(0.25))
 
		# first (and only) set of FC => RELU layers
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(512))
		model.add(tf.keras.layers.Activation("relu"))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Dropout(0.5))
 
		# softmax classifier
		model.add(tf.keras.layers.Dense(self.numOfOutputs))
		model.add(tf.keras.layers.Activation(self.finalActivation))
 
		# return the constructed network architecture
		return model 

	def defineNet4(self):     #src https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py	   #CIFAR100
		model = tf.keras.models.Sequential()

		model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(self.imgWidth,self.imgHeight, self.channels)))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.Convolution2D(32, (3, 3)))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(tf.keras.layers.Dropout(0.25))

		model.add(tf.keras.layers.Convolution2D(64, 3, 3, padding='same'))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.Convolution2D(64, 3, 3,padding='same'))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(tf.keras.layers.Dropout(0.25))

		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(512))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.Dropout(0.5))
		model.add(tf.keras.layers.Dense(self.numOfOutputs))
		model.add(tf.keras.layers.Activation(self.finalActivation))

		# return the constructed network architecture
		return model 

	def defineNet5(self):   #Cifar10.    #https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
		weight_decay = 1e-4

		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=(self.imgWidth,self.imgHeight, self.channels)))
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
		model.add(tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation))

		# return the constructed network architecture
		return model 


	 
	
	def defineVGG16(self):
		# load model
		model = VGG16(include_top=False, input_shape=(self.imgWidth,self.imgHeight, self.channels))  #224,224,3
		# mark loaded layers as not trainable
		for layer in model.layers:
			layer.trainable = False
		# add new classifier layers
		flat1 = tf.keras.layers.Flatten()(model.layers[-1].output)
		class1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
		output = tf.keras.layers.Dense(self.numOfOutputs, activation=self.finalActivation)(class1)
		# define new model
		model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

		return model	
			 
	