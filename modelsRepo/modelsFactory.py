import tensorflow as tf

class  ModelCreator:


	def __init__(self, NNTitle):
		#self.imgWidth,self.imgHeight=imgSize
		if (NNTitle)=="HoursedVsHumanModel":
			self.model=self.defineHoursedVsHumanModel()
		elif (NNTitle)=="CatsvsDogsModel":
			self.model=self.defineCatsvsDogsModel()
		elif NNTitle=="LenetModel":

			self.imgWidth=28
			self.imgHeight=28
			self.model=self.defineLenetModel()
			print("[INFO]  Lenet created")

	def defineLenetModel(self):
		model = tf.keras.models.Sequential()

		# first set of CONV => RELU => POOL layers
		model.add(tf.keras.layers.Conv2D(20, (5, 5),  activation='relu',padding="same",input_shape=(self.imgWidth,self.imgHeight, 3)))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(tf.keras.layers.Conv2D(50, (5, 5),   activation='relu',padding="same"))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(500,activation='relu'))

		# sigmoid classifier
		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



		# return the constructed network architecture
		return model

	def defineHoursedVsHumanModel(self):
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
		tf.keras.layers.Dense(1, activation='sigmoid')])
		return model





	def defineCatsvsDogsModel(self):

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
	    tf.keras.layers.Dense(1, activation='sigmoid')  ])
		return model