

#python trainBinaryClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 100  --width  150 --height  150 --testDir test_images_cats_and_dogs

#python trainBinaryClassifer_flow_from_directory.py  --datasetDir horse-or-human --networkID net1  --EPOCHS 2  --width  300 --height  300 --testDir test_horses_or_Human
#python trainBinaryClassifer_flow_from_directory.py  --datasetDir Food-5K --networkID net1  --EPOCHS 2  --width  300 --height  300 

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
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint




if __name__ == '__main__':


    numOfOutputs=1  # better not play with this as this suits better a binary classifier
    BS = 20
    root_dir="datasets"






    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetDir", required=True, help="path to dataset directory with train and validation images")
    ap.add_argument("--testDir", default=None, help="path to test directory with test images")
    ap.add_argument("--networkID", required=True, help="I.D. of the network")
    ap.add_argument("--EPOCHS", required=True, help="Number of maximum epochs to train")
    ap.add_argument("--width", required=True, help="width of image")
    ap.add_argument("--height", required=True, help="height of image")

    #read the arguments
    args = vars(ap.parse_args())
    datasetDir=args["datasetDir"]
    networkID=args["networkID"]
    EPOCHS=int(args["EPOCHS"])
    width=int(args["width"])
    height=int(args["height"])
    testDir=args["testDir"]
    input_shape=width,height



    #Always have training image folders in folder 'train' and validation images  folders in folder 'validation'. both  folders should be in  datasetDir in root_dir.  root_dir is always "datasets"
    base_dir = os.path.join(root_dir,datasetDir)       
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    #Read the labels as the name of folders, since this is a binary classifier it is expected to have a total of 2 folders. A folder for each class
    labels=paths.get_immediate_subdirectories(train_dir)
    print(train_dir)
    #sort labels alphabetically for consistency 
    labels.sort()
    print(labels)



    #get statistics about your dataset
    NUM_TRAIN_IMAGES,NUM_TEST_IMAGES=paths.getTrainStatistics(datasetDir,train_dir,validation_dir)

    #draw sample images for training and  validation datasets 
    train_label1_dir = os.path.join(train_dir, labels[0])
    train_label2_dir = os.path.join(train_dir, labels[1])


    fileToSaveSampleImage=os.path.join("Results","sample_"+datasetDir+".png")
    plotUtil.drarwGridOfImages(base_dir,fileToSaveSampleImage)

    folderNameToSaveBestModel="{}_Best_classifier".format(datasetDir)
    folderNameToSaveBestModel=os.path.join("Results",folderNameToSaveBestModel)

    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1 ,  patience=200)
    mc = ModelCheckpoint(folderNameToSaveBestModel, monitor='val_loss', mode='min', save_best_only=True)







    #build the model
    model=modelsFactory.ModelCreator(numOfOutputs,width,height,networkID=networkID).model
    #print model structure 
    model.summary()


    #compile model
    model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics = ['acc'])





    train_datagen = ImageDataGenerator(
          rescale=1./255,   #All images will be rescaled by 1./255
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




    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                             batch_size=BS,
                                                             class_mode  = 'binary',
                                                             target_size = input_shape)



    print("*************************************************************************************************************")      
    #Write   labels encoding to pickle file,they are sorted by default alphabetically
    labeles_dictionary = train_generator.class_indices
    print("[INFO] Class labels encoded  as follows {}".format(labeles_dictionary))  
    f_pickle=os.path.join("Results",datasetDir+"_labels.pkl")
    pickle.dump(labeles_dictionary, open(f_pickle, 'wb'))
    print("[INFO] Labels  are saved to pickle file {}  ".format(f_pickle))
    print("*************************************************************************************************************")      

    input("Press any key to start training ")



    #Actual training
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=NUM_TRAIN_IMAGES // BS,   ## 2000 images = batch_size * steps-----steps=images/batch_size
                                  epochs=EPOCHS,
                                  validation_steps=NUM_TEST_IMAGES // BS,
                                  verbose=2 ,callbacks=[es, mc])


    #save model
    fileNameToSaveModel="{}_{}_binaryClassifier.h5".format(labels[0],labels[1])
    fileNameToSaveModel=os.path.join("Results",fileNameToSaveModel)
    model.save(fileNameToSaveModel ,save_format='h5')


    # save the model to disk
    folderNameToSaveModel="{}_Classifier".format(datasetDir)
    folderNameToSaveModel=os.path.join("Results",folderNameToSaveModel)
    model.save(folderNameToSaveModel,save_format='tf') #model is saved in TF2 format (default)





    print("*************************************************************************************************************")      

    print("[INFO] Model saved  to folder {} in both .h5 and TF2 format".format(folderNameToSaveModel))
    print("[INFO] Best Model saved  to folder {}".format(folderNameToSaveBestModel))


    #plot and save training curves 
    info1=plotUtil.plotAccuracyAndLossesonSameCurve(history)
    info2=plotUtil.plotAccuracyAndLossesonSDifferentCurves(history)
    print("*************************************************************************************************************")      
    print(info1)
    print(info2)
    print("*************************************************************************************************************")      


    modelFile=fileNameToSaveModel
    root_dir="TestImages"
      

    if (testDir != None):
        path_test=os.path.join(root_dir,testDir)
       #evaluate on a seperate test dataset
        modelEvaluator=ModelEvaluator(modelFile,labels,input_shape,path_test)
        modelEvaluator.evaluate1()  #using sklearn & testGenerator



