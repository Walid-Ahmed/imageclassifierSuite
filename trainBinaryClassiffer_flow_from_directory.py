


#usage python trainBinaryClassiffer_flow_from_directory.py
#The final layer will have only 1 neuron

import os
from imutils import paths
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from  util import  plotUtil
import pickle



if __name__ == '__main__':

    NNTitle="HoursedVsHumanModel"
    NNTitle="CatsvsDogsModel"
    numOfOutputs=1

    BS = 20
    numberOfEpochs=2
    #numberOfEpochs=2

    root_dir="datasets"

    NNTitle="net2"
    datasetDir='cats_and_dogs'
    input_shape=150,150    #width,height
    width,height=input_shape
    testDir="test_images_cats_and_dogs"
    labels=["cats","dogs"]




      
 



    if(NNTitle=="HoursedVsHumanModel"):
      datasetDir="horse-or-human"
      labels=["horses","humans"]  #todo read them from directory names
      input_shape=300,300    #width,height
      testDir="test_horses_or_Human"






    base_dir = os.path.join(root_dir,datasetDir)
    labels.sort()


       

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

    #print(train_label1_fnames[:10])
    #print(train_label2_fnames[:10])

    """Let's find out the total number of cat and dog images in the `train` and `validation` directories:"""
    totalImages = len(os.listdir(train_label1_dir ) )+ len(os.listdir(train_label2_dir ) )+len(os.listdir( validation_label1_dir ) )+len(os.listdir( validation_label2_dir ) )
    print('[INFO] Total images in dataset '+datasetDir+ 'images :', totalImages)

    print('[INFO] Total training '+labels[0]+ ' images :', len(os.listdir(train_label1_dir ) ))
    print('[INFO] Total training ' + labels[1]+ ' images :', len(os.listdir(train_label2_dir ) ))
    NUM_TRAIN_IMAGES= len(os.listdir(train_label1_dir ))+len(os.listdir(train_label2_dir ) )

    print('[INFO] Total validation '+labels[0]+ ' images :', len(os.listdir( validation_label1_dir ) ))
    print('[INFO] Total validation '+ labels[1]+ ' images :', len(os.listdir( validation_label2_dir ) ))
    NUM_TEST_IMAGES=len(os.listdir( validation_label1_dir ) )+len(os.listdir( validation_label2_dir ) )

    print('[INFO] Total  training images in dataset: {} '.format(NUM_TRAIN_IMAGES))
    print('[INFO] Total validation images in dataset  {}'.format( NUM_TEST_IMAGES))



    plotUtil.drarwGridOfImages(train_label1_dir,train_label2_dir)



    import tensorflow as tf




    model=modelsFactory.ModelCreator(numOfOutputs,width,height,NNTitle=NNTitle).model

    model.summary()


    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics = ['acc'])


    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # All images will be rescaled by 1./255.
    #train_datagen = ImageDataGenerator( rescale = 1.0/255. )


    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')


    test_datagen  = ImageDataGenerator( rescale = 1.0/255. )



    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=BS,
                                                        class_mode='binary',   #class_mode="categorical" will do one hot encoding
                                                        target_size=input_shape)     




    #Write   labels encoding to pickle file,thwy are sorted by default alphabetically
    labeles_dictionary = train_generator.class_indices
    print("[INFO] Class labels encoded  as follows {}".format(labeles_dictionary))  
    f_pickle=os.path.join("Results","labels.pkl")
    pickle.dump(labeles_dictionary, open(f_pickle, 'wb'))
    print("[INFO] Labels  are saved to pickle file {}  ".format(f_pickle))
    input("Press any key to continue")






    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                             batch_size=BS,
                                                             class_mode  = 'binary',
                                                             target_size = input_shape)






    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=NUM_TRAIN_IMAGES // BS,   ## 2000 images = batch_size * steps-----steps=images/batch_size
                                  epochs=numberOfEpochs,
                                  validation_steps=NUM_TEST_IMAGES // BS,
                                  verbose=2)

    """###Running the Model

    Let's now take a look at actually running a prediction using the model. This code will allow you to choose 1 or more files from your file system, it will then upload them, and run them through the model, giving an indication of whether the object is a dog or a cat.
    """

    fileNameToSaveModel="{}_{}_binaryClassifier.keras2".format(labels[0],labels[1])
    fileNameToSaveModel=os.path.join("Results",fileNameToSaveModel)
    model.save(fileNameToSaveModel)
    print("[INFO] Model saved  to file {}".format(fileNameToSaveModel))










    plotUtil.plotAccuracyAndLossesonSameCurve(history)
    plotUtil.plotAccuracyAndLossesonSDifferentCurves(history)


    modelFile=fileNameToSaveModel
    root_dir="TestImages"
      

    path_test=os.path.join(root_dir,testDir)


   #evaluate on a seperate yest dataset
    modelEvaluator=ModelEvaluator(modelFile,labels,input_shape,path_test)
    modelEvaluator.evaluate1()  #using sklearn & testGenerator
    modelEvaluator.evaluate2()  #without using sklearn & testGenerator
    modelEvaluator.evaluate3()  #using sklearn


