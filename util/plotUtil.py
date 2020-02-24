
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import matplotlib
#import  helper
from util import paths
import random
from PIL import Image
import matplotlib.cm as cm
import numpy as np
import cv2

matplotlib.use("Qt5Agg")
print("[INFO] matplotlib BACKEND IS {}".format(matplotlib.get_backend())) #[INFO] matplotlib BACKEND IS agg
info=""

def plotAccuracyAndLossesonSDifferentCurves(history,title=""):

  info=""
    #Let's plot the training/validation accuracy and loss as collected during training:
  plt.style.use("ggplot")


  #-----------------------------------------------------------
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  #-----------------------------------------------------------

  try:
    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

  except:  
    acc      = history.history[     'acc' ]
    val_acc  = history.history[ 'val_acc' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]


  epochs   = range(len(acc)) # Get number of epochs

  #------------------------------------------------
  # Plot training and validation accuracy per epoch
  #------------------------------------------------
  plt.figure()

  plt.plot  ( epochs,     acc ,label="train_acc")
  plt.plot  ( epochs, val_acc, label="val_acc" )
  plt.title (title+'Training and validation accuracy')
  plt.xlabel("Epoch #")
  plt.ylabel("Accuracy")
  fileToSaveAccuracyCurve=os.path.join("Results",title+"plot_acc.png")
  plt.savefig(fileToSaveAccuracyCurve)
  info=info+"[INFO] Accuracy curve saved to {}".format(fileToSaveAccuracyCurve)
  plt.legend(loc="upper left")
  plt.show()


  plt.figure()
  #------------------------------------------------
  # Plot training and validation loss per epoch
  #------------------------------------------------
  plt.plot  ( epochs,     loss ,label="train_loss")
  plt.plot  ( epochs, val_loss ,label="val_loss")
  plt.title (title+'Training and validation loss'   )
  plt.xlabel("Epoch #")
  plt.ylabel("Loss")
  fileToSaveLossCurve=os.path.join("Results",title+"plot_loss.png")
  info=info+"[INFO] Loss curve saved to {}".format(fileToSaveLossCurve)
  plt.savefig(os.path.join("Results","plot_loss.png"))
  plt.legend(loc="upper left")

  plt.show()
  return info



def plotAccuracyAndLossesonSameCurve(history,title=""):

    # construct a plot that plots and saves the training history

  info=""
    #-----------------------------------------------------------

    
  try:
    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

  except:  
    acc      = history.history[     'acc' ]
    val_acc  = history.history[ 'val_acc' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]




  epochs   = range(len(acc)) # Get number of epochs

  plt.style.use("ggplot")
  plt.figure()
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.plot(epochs, acc, label="train_acc")
  plt.plot(epochs, val_acc, label="val_acc")
  plt.title(title+"Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  fileToSaveLossAccCurve=os.path.join("Results",title+"plot_loss_accu.png")
  info=info+"[INFO] Loss curve saved to {}".format(fileToSaveLossAccCurve)
  plt.savefig(fileToSaveLossAccCurve)
  plt.show()
  return info

   



def drarwGridOfImages(dataSetDir,fileNameToSaveImage=None,channels=3):

  info=""


  #print(train_label1_fnames[:10])
  #print(train_label2_fnames[:10])
  imagePaths = sorted(list(paths.list_images(dataSetDir)))



  # Parameters for our graph; we'll output images in a 4x4 configuration
  nrows = 4
  ncols = 4

  pic_index = 0 # Index for iterating over images

  #display a batch of 4*4 pictures

  # Set up matplotlib fig, and size it to fit 4x4 pics
  fig = plt.gcf()
  fig.set_size_inches(ncols*4, nrows*4)

  pic_index+=8
  random.shuffle(imagePaths)
  imagePaths=imagePaths[0:16]





  for i, img_path in enumerate(imagePaths):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    if (channels==3):
      img = mpimg.imread(img_path)
    else:
      img=Image.open(img_path).convert('L')
    plt.imshow(img)
 

  if(fileNameToSaveImage != None):
    plt.savefig(fileNameToSaveImage)
  plt.show()
  return info

def drarwGridOfImagesFromImagesData(images,fileNameToSaveImage=None):

 
  info=""
  # Parameters for our graph; we'll output images in a 4x4 configuration
  nrows = 4
  ncols = 4

  pic_index = 0 # Index for iterating over images

  #display a batch of 4*4 pictures

  # Set up matplotlib fig, and size it to fit 4x4 pics
  fig = plt.gcf()
  fig.set_size_inches(ncols*4, nrows*4)



  for i in range(16):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = images[i]

    plt.imshow(img)

 
  if(fileNameToSaveImage != None):
    plt.savefig(fileNameToSaveImage)
  plt.show()
  return info

