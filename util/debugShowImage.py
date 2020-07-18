
import os

def drarwGridOfImages(dataSetDir,fileNameToSaveImage="testgrid.png", display=True):


  info=""


  #print(train_label1_fnames[:10])
  #print(train_label2_fnames[:10])
  imagePaths = sorted(list(paths.list_images(dataSetDir)))

  firstImage=imagePaths[0]
  firstImage= imread(firstImage)
  print(firstImage.shape)
  channels=len(firstImage.shape)


  if (channels==2):
    channels=1


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
  print(imagePaths)
  print("channels={}".format(channels))



  for i, img_path in enumerate(imagePaths):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    if (channels==3):
      try:
        img = mpimg.imread(img_path)

        print("[MSG]  read image from file{}".format(img_path))

      except:  
        print("[Error]  Can not read image from file{}".format(img_path))
        exit()


    else:
      try:
        img=Image.open(img_path).convert('L')
        img = img.resize((48,48))

        #print(type(img))
        
      except:  
        print("[Error]  Can not read image from file{}".format(img_path))

    print("Attempting to show image {} from ".format(img_path))
    plt.imshow(img)  #draws an image on the current figure
    #cmap='gray', vmin=0, vmax=255
 

  if(fileNameToSaveImage != None):
    plt.savefig(fileNameToSaveImage)
    input("Image saved to {}  ".format(fileNameToSaveImage))


  if(display):  
    try:
      plt.show()   #displays the figure (and enters the mainloop of whatever gui backend you're using). You shouldn't call it until you've plotted things and want to see them displayed.
    except:
      print("ERROR")

  return info,channels



  if __name__ == "__main__":
  	dataSetDir=os.path.join("datasets","coronaVirus")
  	drarwGridOfImages(dataSetDir,fileNameToSaveImage=None, display=True)

