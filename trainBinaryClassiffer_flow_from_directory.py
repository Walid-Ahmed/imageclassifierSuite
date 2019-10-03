

#python trainBinaryClassiffer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 100  --width  150 --height  150 --testDir test_images_cats_and_dogs

#python trainBinaryClassiffer_flow_from_directory.py  --datasetDir horse-or-human --networkID net1  --EPOCHS 2  --width  300 --height  300 --testDir test_horses_or_Human

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
import argparse
from util import paths



if __name__ == '__main__':


    numOfOutputs=1

    BS = 20
    #numberOfEpochs=2

    root_dir="datasets"






      # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetDir", required=True, help="datasetDir")
    ap.add_argument("--testDir", required=True, help="testDir")
    ap.add_argument("--networkID", required=True, help="I.D. of the network")
    ap.add_argument("--EPOCHS", required=True, help="name of the network")
    ap.add_argument("--width", required=True, help="width of image")
    ap.add_argument("--height", required=True, help="height of image")


    args = vars(ap.parse_args())
    datasetDir=args["datasetDir"]
    networkID=args["networkID"]
    EPOCHS=int(args["EPOCHS"])
    width=int(args["width"])
    height=int(args["height"])
    testDir=args["testDir"]
    input_shape=width,height





      
 



    






    base_dir = os.path.join(root_dir,datasetDir)


       

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    labels=paths.get_immediate_subdirectories(train_dir)
    labels.sort()

    # Directory with our training cpictures



    NUM_TRAIN_IMAGES,NUM_TEST_IMAGES=paths.getTrainStatistics(datasetDir,train_dir,validation_dir)


    train_label1_dir = os.path.join(train_dir, labels[0])
    train_label2_dir = os.path.join(train_dir, labels[1])

    plotUtil.drarwGridOfImages(train_label1_dir,train_label2_dir)



    import tensorflow as tf




    model=modelsFactory.ModelCreator(numOfOutputs,width,height,NNTitle=networkID).model

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



    print("*************************************************************************************************************")      

    #Write   labels encoding to pickle file,they are sorted by default alphabetically
    labeles_dictionary = train_generator.class_indices
    print("[INFO] Class labels encoded  as follows {}".format(labeles_dictionary))  

    f_pickle=os.path.join("Results","labels.pkl")
    pickle.dump(labeles_dictionary, open(f_pickle, 'wb'))
    print("[INFO] Labels  are saved to pickle file {}  ".format(f_pickle))
    print("*************************************************************************************************************")      

    input("Press any key to start Training ")






    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                             batch_size=BS,
                                                             class_mode  = 'binary',
                                                             target_size = input_shape)






    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=NUM_TRAIN_IMAGES // BS,   ## 2000 images = batch_size * steps-----steps=images/batch_size
                                  epochs=EPOCHS,
                                  validation_steps=NUM_TEST_IMAGES // BS,
                                  verbose=2)

    """###Running the Model

    Let's now take a look at actually running a prediction using the model. This code will allow you to choose 1 or more files from your file system, it will then upload them, and run them through the model, giving an indication of whether the object is a dog or a cat.
    """

    fileNameToSaveModel="{}_{}_binaryClassifier.keras2".format(labels[0],labels[1])
    fileNameToSaveModel=os.path.join("Results",fileNameToSaveModel)
    model.save(fileNameToSaveModel)
    print("[INFO] Model saved  to file {}".format(fileNameToSaveModel))










    info1=plotUtil.plotAccuracyAndLossesonSameCurve(history)
    info2=plotUtil.plotAccuracyAndLossesonSDifferentCurves(history)
    print("*************************************************************************************************************")      
    print(info1)
    print(info2)
    print("*************************************************************************************************************")      


    modelFile=fileNameToSaveModel
    root_dir="TestImages"
      

    path_test=os.path.join(root_dir,testDir)


   #evaluate on a seperate yest dataset
    modelEvaluator=ModelEvaluator(modelFile,labels,input_shape,path_test)
    modelEvaluator.evaluate1()  #using sklearn & testGenerator
    modelEvaluator.evaluate2()  #without using sklearn & testGenerator
    modelEvaluator.evaluate3()  #using sklearn


