
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use("Qt5Agg")
print("[INFO] matplotlib BACKEND IS {}".format(matplotlib.get_backend())) #[INFO] matplotlib BACKEND IS agg


def plotAccuracyAndLossesonSDifferentCurves(history):


    #Let's plot the training/validation accuracy and loss as collected during training:
  plt.style.use("ggplot")


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
  plt.savefig(os.path.join("Results","plot_acc.png"))
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
  plt.savefig(os.path.join("Results","plot_loss.png"))
  plt.legend(loc="upper left")


  plt.show()


def plotAccuracyAndLossesonSameCurve(history):

    # construct a plot that plots and saves the training history

 
    #-----------------------------------------------------------
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
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  plt.savefig(os.path.join("Results","plot_loss_accu.png"))
  plt.show()

   

def drarwGridOfImages(train_label1_dir,train_label2_dir):


  train_label1_fnames = os.listdir( train_label1_dir )
  train_label2_fnames = os.listdir( train_label2_dir )

  print(train_label1_fnames[:10])
  print(train_label2_fnames[:10])


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


