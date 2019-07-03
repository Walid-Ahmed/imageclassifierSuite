

import os
from imutils import paths
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from  modelsRepo import modelsFactory


BS = 20
numberOfEpochs=15
imgSize=(300,300)



root_dir="datasets"

datasetDir='cats_and_dogs'
labels=["cats","dogs"]
testDir="test_images_cats_and_dogs"

'''
datasetDir="horse-or-human"
labels=["horses","humans"]  #todo read them from directory names
testDir="test_horses_or_Human"
'''


base_dir = os.path.join(root_dir,datasetDir)
labels.sort()
path_test=os.path.join(root_dir,testDir)
totalTest = len(list(paths.list_images(path_test)))
print('[INFO] Total images in test  dataset '+path_test+ 'images :', totalTest)



train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_label1_dir = os.path.join(train_dir, labels[0])
train_label2_dir = os.path.join(train_dir, labels[1])

# Directory with our validation cat/dog pictures
validation_label1_dir = os.path.join(validation_dir, labels[0])
validation_label2_dir = os.path.join(validation_dir, labels[1])

"""Now, let's see what the filenames look like in the `cats` and `dogs` `train` directories (file naming conventions are the same in the `validation` directory):"""

train_label1_fnames = os.listdir( train_label1_dir )
train_label2_fnames = os.listdir( train_label2_dir )

print(train_label1_fnames[:10])
print(train_label2_fnames[:10])

"""Let's find out the total number of cat and dog images in the `train` and `validation` directories:"""
totalImages = len(os.listdir(train_label1_dir ) )+ len(os.listdir(train_label2_dir ) )+len(os.listdir( validation_label1_dir ) )+len(os.listdir( validation_label2_dir ) )
print('[INFO] Total images in dataset '+datasetDir+ 'images :', totalImages)

print('[INFO] Total training '+labels[0]+ ' images :', len(os.listdir(train_label1_dir ) ))
print('[INFO] Total training ' + labels[1]+ ' images :', len(os.listdir(train_label2_dir ) ))

print('[INFO] Total validation '+labels[0]+ ' images :', len(os.listdir( validation_label1_dir ) ))
print('[INFO] Total validation '+ labels[1]+ ' images :', len(os.listdir( validation_label2_dir ) ))

"""For both cats and dogs, we have 1,000 training images and 500 validation images.

Now let's take a look at a few pictures to get a better sense of what the cat and dog datasets look like. First, configure the matplot parameters:
"""

# %matplotlib inline



# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

"""Now, display a batch of 8 cat and 8 dog pictures. You can rerun the cell to see a fresh batch each time:"""

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_label1_pix = [os.path.join(train_label1_dir, fname) 
                for fname in train_label1_fnames[ pic_index-8:pic_index] 
               ]

next_label2_pix = [os.path.join(train_label2_dir, fname) 
                for fname in train_label2_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_label1_pix+next_label2_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()



import tensorflow as tf

"""Next we will define a Sequential layer as before, adding some convolutional layers first. Note the input shape parameter this time. In the earlier example it was 28x28x1, because the image was 28x28 in greyscale (8 bits, 1 byte for color depth). This time it is 150x150 for the size and 3 (24 bits, 3 bytes) for the color depth.

We then add a couple of convolutional layers as in the previous example, and flatten the final result to feed into the densely connected layers.

Finally we add the densely connected layers. 

Note that because we are facing a two-class classification problem, i.e. a *binary classification problem*, we will end our network with a [*sigmoid* activation](https://wikipedia.org/wiki/Sigmoid_function), so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
"""

model=modelsFactory.ModelCreator("HoursedVsHumanModel",imgSize).model
"""The model.summary() method call prints a summary of the NN"""

model.summary()



"""The "output shape" column shows how the size of your feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by a bit due to padding, and each pooling layer halves the dimensions.

Next, we'll configure the specifications for model training. We will train our model with the `binary_crossentropy` loss, because it's a binary classification problem and our final activation is a sigmoid. (For a refresher on loss metrics, see the [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture).) We will use the `rmsprop` optimizer with a learning rate of `0.001`. During training, we will want to monitor classification accuracy.

**NOTE**: In this case, using the [RMSprop optimization algorithm](https://wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp) is preferable to [stochastic gradient descent](https://developers.google.com/machine-learning/glossary/#SGD) (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as [Adam](https://wikipedia.org/wiki/Stochastic_gradient_descent#Adam) and [Adagrad](https://developers.google.com/machine-learning/glossary/#AdaGrad), also automatically adapt the learning rate during training, and would work equally well here.)
"""

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])

"""### Data Preprocessing

Let's set up data generators that will read pictures in our source folders, convert them to `float32` tensors, and feed them (with their labels) to our network. We'll have one generator for the training images and one for the validation images. Our generators will yield batches of 20 images of size 150x150 and their labels (binary).

As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. (It is uncommon to feed raw pixels into a convnet.) In our case, we will preprocess our images by normalizing the pixel values to be in the `[0, 1]` range (originally all values are in the `[0, 255]` range).

In Keras this can be done via the `keras.preprocessing.image.ImageDataGenerator` class using the `rescale` parameter. This `ImageDataGenerator` class allows you to instantiate generators of augmented image batches (and their labels) via `.flow(data, labels)` or `.flow_from_directory(directory)`. These generators can then be used with the Keras model methods that accept data generators as inputs: `fit_generator`, `evaluate_generator`, and `predict_generator`.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=imgSize)     
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size=imgSize)

"""### Training
Let's train on all 2,000 images available, for 15 epochs, and validate on all 1,000 test images. (This may take a few minutes to run.)

Do note the values per epoch.

You'll see 4 values per epoch -- Loss, Accuracy, Validation Loss and Validation Accuracy. 

The Loss and Accuracy are a great indication of progress of training. It's making a guess as to the classification of the training data, and then measuring it against the known label, calculating the result. Accuracy is the portion of correct guesses. The Validation accuracy is the measurement with the data that has not been used in training. As expected this would be a bit lower. You'll learn about why this occurs in the section on overfitting later in this course.
"""


# initialize the testing generator
test_generator = test_datagen.flow_from_directory(
  path_test,
  class_mode="binary",
  target_size=imgSize,
  batch_size=20)

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=numberOfEpochs,
                              validation_steps=50,
                              verbose=2)

"""###Running the Model

Let's now take a look at actually running a prediction using the model. This code will allow you to choose 1 or more files from your file system, it will then upload them, and run them through the model, giving an indication of whether the object is a dog or a cat.
"""


model.save("{}_{}_binaryClassifier.keras2".format(labels[0],labels[1]))





# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
test_generator.reset()
predIdxs = model.predict_generator(test_generator,steps=(totalTest // BS) + 1)

predictedLabels=[]


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
for  predIdx in predIdxs:

  if predIdx>0.5:     #1  is a labels[1]
      print(" belongs to {}".format(labels[1]))
      predictedLabels.append(1)
      
  else:
      print( " belongs to  {}".format(labels[0]))
      predictedLabels.append(0)





# show a nicely formatted classification report
print(classification_report(test_generator.classes, predictedLabels,target_names=test_generator.class_indices.keys()))
#exit()


'''
import numpy as np

from keras.preprocessing import image


for file in os.listdir(path_test):
 
  # predicting images
  imgPath=os.path.join(path_test,file)

  if (".DS_Store") in imgPath:
    continue
  print(imgPath)
  img=image.load_img(imgPath, target_size=(150, 150))
  
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(type(classes))  
  
  print(classes[0])
  
  if classes[0]>0:     #1  is a labels[1]
    print(file + " belongs to {}".format(labels[1]))
    
  else:
    print(file + " belongs to  {}".format(labels[0]))


'''


### Evaluating Accuracy and Loss for the Model

#Let's plot the training/validation accuracy and loss as collected during training:


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc ,label="train_acc")
plt.plot  ( epochs, val_acc, label="val_acc" )
plt.title ('Training and validation accuracy')
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.savefig("plot_acc.png")
plt.legend(loc="upper left")



plt.show()

plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss ,label="train_loss")
plt.plot  ( epochs, val_loss ,label="val_loss")
plt.title ('Training and validation loss'   )
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig("plot_loss.png")
plt.legend(loc="upper left")


plt.show()



