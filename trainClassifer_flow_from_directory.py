

#python trainClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 10  --width  150 --height  150 --testDir test_images_cats_and_dogs --ResultsFolder  Results/r1_cats_dogs


#python trainClassifer_flow_from_directory.py  --datasetDir Cyclone_Wildfire_Flood_Earthquake_Database --networkID net2  --EPOCHS 20  --width  150 --height  150  --BS 32  --ResultsFolder  Results/r1_disaster

#python trainClassifer_flow_from_directory.py  --datasetDir horse-or-human --networkID net1  --EPOCHS 2  --width  300 --height  300 --testDir test_horses_or_Human


#The final layer will have only 1 neuron if we are dealing with 2 classes only (binary classiffier)

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
from tensorflow.keras.callbacks import TensorBoard
import shutil
from tensorflow.keras.utils import plot_model




def predictBinaryValue(probs,threshold=0.5):
    y_pred=[]
    for prob in probs:
        if (prob>threshold):
            y_pred.append(1)
        else:   
            y_pred.append(0)
    return np.asarray(y_pred)


if __name__ == '__main__':


    root_dir="datasets"






    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetDir", required=True, help="path to dataset directory with train and validation images")
    ap.add_argument("--testDir", default=None, help="path to test directory with test images")
    ap.add_argument("--networkID", required=True, help="I.D. of the network")
    ap.add_argument("--EPOCHS", required=False, type=int, default=25, help="Number of maximum epochs to train")
    ap.add_argument("--BS", required=False, default=16 , type=int, help="Batch size")
    ap.add_argument("--width", required=True, help="width of image")
    ap.add_argument("--height", required=True, help="height of image")
    ap.add_argument("--patience", required=False, default=50, type=int,help="Number of epochs to wait without accuracy imrovment")
    ap.add_argument("--ResultsFolder", required=False, default="Results",help="Folder to save Results")
    ap.add_argument("--lr", required=False, type=float, default=0.001,help="Initial Learning rate")



    #read the arguments
    args = vars(ap.parse_args())
    datasetDir=args["datasetDir"]
    networkID=args["networkID"]
    EPOCHS=args["EPOCHS"]
    width=int(args["width"])
    height=int(args["height"])
    testDir=args["testDir"]
    BS=args["BS"]
    patience=args["patience"]
    ResultsFolder=args["ResultsFolder"]
    learningRate=args["lr"]


    input_shape=width,height
    parameters="[INFO]  Data Set Directory".format(datasetDir) +"\n"
    parameters=parameters+"[INFO] networkID:".format(networkID) +"\n"
    parameters=parameters+"[INFO] EPOCHS:".format(EPOCHS) +"\n"
    parameters=parameters+"[INFO] BS".format(EPOCHS) +"\n"
    parameters=parameters+"[INFO] width".format(EPOCHS) +"\n"
    parameters=parameters+"[INFO] height".format(EPOCHS) +"\n"
    parameters=parameters+"[INFO] patience".format(EPOCHS) +"\n"
    parameters=parameters+"[INFO] ResultsFolder".format(EPOCHS) +"\n"
    parameters=parameters+"[INFO] lr".format(EPOCHS) +"\n"






    if os.path.exists(ResultsFolder):
        print("[Warning]  Folder aready exists, All files in folder will be deleted")
        input("[msg]  Press any key to continue")
        shutil.rmtree(ResultsFolder)
    os.mkdir(ResultsFolder)





    #Always have training image folders in folder 'train' and validation images  folders in folder 'validation'. both  folders should be in  datasetDir in root_dir.  root_dir is always "datasets"
    base_dir = os.path.join(root_dir,datasetDir)       
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    #Read the labels as the name of folders, since this is a binary classifier it is expected to have a total of 2 folders. A folder for each class
    labels=paths.get_immediate_subdirectories(train_dir)
    #sort labels alphabetically for consistency 
    labels.sort()
    print("[INFO] training labels are  {}".format(str(labels)))
    print("[INFO] Number of classes are  {}".format(len(labels)))


    numOfOutputs=len(labels)  

    if(numOfOutputs==2):  # binary classiffication problem
        numOfOutputs=1
        lossFun='binary_crossentropy'
        classMode='binary'
    else:
        lossFun='categorical_crossentropy'    
        classMode='categorical'  #class_mode="categorical" will do one hot encoding



    #get statistics about your dataset
    NUM_TRAIN_IMAGES,NUM_TEST_IMAGES=paths.getTrainStatistics(datasetDir,train_dir,validation_dir)

    #draw sample images for training and  validation datasets 
    fileToSaveSampleImage=os.path.join(ResultsFolder,"sample_"+datasetDir+".png")
    plotUtil.drarwGridOfImages(base_dir,fileToSaveSampleImage)


    folderNameToSaveBestModel="{}_Best_classifier".format(datasetDir)
    folderNameToSaveBestModel=os.path.join(ResultsFolder,folderNameToSaveBestModel)

    es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1 ,  patience=patience)
    mc = ModelCheckpoint(folderNameToSaveBestModel, monitor='val_acc', mode='max', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=ResultsFolder,profile_batch=0)








    #build the model
    model=modelsFactory.ModelCreator(numOfOutputs,width,height,networkID=networkID).model
    #print model structure 
    model.summary()
    #compile model
    model.compile(optimizer=RMSprop(lr=learningRate),loss=lossFun,metrics = ['acc'])
    print("[INFO] Model compiled sucessfully")


    fileToSaveModelPlot=os.path.join(ResultsFolder,'model.png')
    plot_model(model, to_file=fileToSaveModelPlot,show_shapes="True")
    print("[INFO] Model plot  saved to file  {} ".format(fileToSaveModelPlot))









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
                                                        class_mode=classMode,   #class_mode="categorical" will do one hot encoding
                                                        target_size=input_shape)     




    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                             batch_size=BS,
                                                             class_mode  = classMode,
                                                             target_size = input_shape)



    print("*************************************************************************************************************")      
    #Write   labels encoding to pickle file,they are sorted by default alphabetically
    labeles_dictionary = train_generator.class_indices
    print("[INFO] Class labels encoded  as follows {}".format(labeles_dictionary))  
    f_pickle=os.path.join(ResultsFolder,datasetDir+"_labels.pkl")
    pickle.dump(labeles_dictionary, open(f_pickle, 'wb'))
    print("[INFO] Labels  are saved to pickle file {}  ".format(f_pickle))
    print("*************************************************************************************************************")      

    print("[INFO] Training will start now ")




    #Actual training
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=NUM_TRAIN_IMAGES // BS,   ## 2000 images = batch_size * steps-----steps=images/batch_size
                                  epochs=EPOCHS,
                                  validation_steps=NUM_TEST_IMAGES // BS,
                                  verbose=2 ,
                                  callbacks=[es, mc,tensorboard_callback]
                                  )


    #save model
    fileNameToSaveModel="{}_{}_binaryClassifier.h5".format(labels[0],labels[1])
    fileNameToSaveModel=os.path.join(ResultsFolder,fileNameToSaveModel)
    model.save(fileNameToSaveModel ,save_format='h5')


    # save the model to disk
    folderNameToSaveModel="{}_Classifier".format(datasetDir)
    folderNameToSaveModel=os.path.join(ResultsFolder,folderNameToSaveModel)
    model.save(folderNameToSaveModel,save_format='tf') #model is saved in TF2 format (default)





    print("*************************************************************************************************************")      
    print("[INFO] Model saved  to folder {} in both .h5 and TF2 format".format(folderNameToSaveModel))
    print("[INFO] Best Model saved  to folder {}".format(folderNameToSaveBestModel))
    print("[INFO] Sample images from dataset saved to file  {} ".format(fileToSaveSampleImage))





    #plot and save training curves 
    info1=plotUtil.plotAccuracyAndLossesonSameCurve(history,ResultsFolder)
    info2=plotUtil.plotAccuracyAndLossesonSDifferentCurves(history,ResultsFolder)
    print("*************************************************************************************************************")      
    print(info1)
    print(info2)
    print("*************************************************************************************************************")      








    modelFile=fileNameToSaveModel
    root_dir="TestImages"
      

    if (testDir is None):
        path_test=os.path.join(root_dir,testDir)
    else:
        path_test=validation_dir
    
    modelEvaluator=ModelEvaluator(modelFile,labels,input_shape,ResultsFolder,path_test,datasetDir)
    modelEvaluator.evaluate1()  #using sklearn & testGenerator



