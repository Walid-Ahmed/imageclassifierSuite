import tensorflow as tf
from  tensorflow.keras.applications import  ResNet50


class  ModelCreator:


	def __init__(self, numOfOutputs,width,height,channels=3,NNTitle="default"):

		#self.imgWidth,self.imgHeight=imgSize
		self.numOfOutputs=numOfOutputs
		self.imgWidth=width
		self.imgHeight=height
		self.channels=channels

		if(self.numOfOutputs>1):
			self.finalActivation='softmax'
		else:
			self.finalActivation='sigmoid'


		if (NNTitle)=="HoursedVsHumanModel":
			self.model=self.defineNet1()
			print("[INFO]  HoursedVsHuman Model created")

		elif (NNTitle)=="CatsvsDogsModel":
			self.imgWidth=150
			self.imgHeight=150
			self.model=self.defineNet2()
			print("[INFO]  CatsvsDogs Model created")

		elif NNTitle=="LenetModel":
			self.imgWidth=width
			self.imgHeight=height
			self.model=self.defineLenetModel()
			print("[INFO]  Lenet created")


		elif NNTitle=="Resnet50":
			self.imgWidth=width
			self.imgHeight=height
			self.model=self.defineResnet50()
			print("[INFO]  Resnet50 created")	

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



		# return the constructed network architecture
		return model

	def defineNet1(self):   #suitable for HoursedVsHumanModel
		model = tf.keras.models.Sequential([
		# Note the input shape is the desired size of the image 300x300 with 3 bytes color
		# This is the first convolution
		tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(self.imgWidth,self.imgHeight, 3)),
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
	    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(self.imgWidth,self.imgHeight, 3)),
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



	def defineResnet50(self):
		baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=tf.keras.layers.Input(shape=(self.imgWidth,self.imgHeight, 3)))
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

	