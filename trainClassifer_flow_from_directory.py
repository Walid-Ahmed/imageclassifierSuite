

#python trainClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 80  --width  150 --height  150  --ResultsFolder  Results/r1_cats_dogs --labelSmoothing 0.1 --augmentationLevel 1


#python trainClassifer_flow_from_directory.py  --datasetDir Cyclone_Wildfire_Flood_Earthquake_Database --networkID net2  --EPOCHS 20  --width  150 --height  150  --BS 32  --ResultsFolder  Results/r1_disaster


#FacialExpression
#python trainClassifer_flow_from_directory.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 20  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r1a_FacialExpression   --labelSmoothing 0.1   --opt Adam 
#python trainClassifer_flow_from_directory.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 20  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r1b_FacialExpression   --labelSmoothing 0.1   --opt Adam    --modelcheckpoint Results/r1a_FacialExpression/FacialExpression_Classifier.h5   --startepoch 20   --new_lr  0.0001



#python trainClassifer_flow_from_directory.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 20  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r2_FacialExpression  --modelcheckpoint Results/r1_FacialExpression/checkPoints/epoch_30.h5  --startepoch 30 

#python trainClassifer_flow_from_directory.py  --datasetDir horse-or-human --networkID net1  --EPOCHS 15  --width  300 --height  300  --augmentationLevel 1  --ResultsFolder  Results/r1_horse-or-human

#python trainClassifer_flow_from_directory.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 2  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r1_FacialExpression 


#python trainClassifer_flow_from_directory.py  --datasetDir SportsClassification  --networkID Resnet50  --EPOCHS 80  --width  224 --height  224  --ResultsFolder  Results/r1_SportsClassification --labelSmoothing 0.1     --opt Adam  

#python trainClassifer_flow_from_directory.py  --datasetDir coronaVirus  --networkID Resnet50  --EPOCHS 80  --width  224 --height  224  --ResultsFolder  Results/r1_coronaVirus  --labelSmoothing 0.1 --augmentationLevel 1

#python trainClassifer_flow_from_directory.py --datasetDir Cyclone_Wildfire_Flood_Earthquake_Database --networkID DPN --EPOCHS 20 --width 112 --height 112 --BS 32 --ResultsFolder Results/r1_FacialExpression --labelSmoothing 0.1


#python trainClassifer_flow_from_directory.py  --datasetDir coronaVirus  --networkID LenetModel  --EPOCHS 15  --width  224 --height  224  --ResultsFolder  Results/r1_coronaVirus  --labelSmoothing 0.1 --augmentationLevel 1  --useOneNeuronForBinaryClassification False
#python trainClassifer_flow_from_directory.py  --datasetDir coronaVirus  --networkID Resnet50  --EPOCHS 15  --width  224 --height  224  --ResultsFolder  Results/r2_coronaVirus  --labelSmoothing 0.1 --augmentationLevel 1
#python trainClassifer_flow_from_directory.py  --datasetDir coronaVirus  --networkID net1  --EPOCHS 15  --width  224 --height  224  --ResultsFolder  Results/r3_coronaVirus  --labelSmoothing 0.1 --augmentationLevel 1
#python trainClassifer_flow_from_directory.py  --datasetDir coronaVirus  --networkID net2  --EPOCHS 15  --width  224 --height  224  --ResultsFolder  Results/r4_coronaVirus  --labelSmoothing 0.1 --augmentationLevel 1


#python trainClassifer_flow_from_directory.py  --datasetDir Food-11  --networkID VGG16  --EPOCHS 15  --width  224 --height  224  --ResultsFolder  Results/food11  --labelSmoothing 0.0 --augmentationLevel 2

#python trainClassifer_flow_from_directory.py  --datasetDir coronaVirus  --networkID DPN  --EPOCHS 1  --width  224 --height  224  --ResultsFolder  Results/dpn  --labelSmoothing 0.0 --augmentationLevel 0


import os
from imutils import paths
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from modelsRepo import modelsFactory
from modelEvaluator import ModelEvaluator
from  util import     plotUtil
import pickle
import argparse
from util import paths
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import shutil
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
from callbacks  import  TrainingMonitor
from callbacks   import EpochCheckpoint
from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD



#aim to get Reproducible Results with Keras
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

print("[INFO] tf.Keras   version is {}".format( tf.keras.__version__)) 

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
    ap.add_argument("--networkID", required=True, help="I.D. of the network, it can be any of [net1,net2,net3,net4,net5,LenetModel,Resnet50,net3,MiniVGG,VGG16]")
    ap.add_argument("--EPOCHS", required=False, type=int, default=25, help="Number of maximum epochs to train")
    ap.add_argument("--BS", required=False, default=16 , type=int, help="Batch size")
    ap.add_argument("--width", required=True, help="width of image")
    ap.add_argument("--height", required=True, help="height of image")
    ap.add_argument("--patience", required=False, default=50, type=int,help="Number of epochs to wait without accuracy improvment")
    ap.add_argument("--ResultsFolder", required=False, default="Results",help="Folder to save Results")
    ap.add_argument("--lr", required=False, type=float, default=0.001,help="Initial Learning rate")
    ap.add_argument("--new_lr", required=False, type=float, default=1e-4,help="restarting Learning rate")
    ap.add_argument("--labelSmoothing", type=float, default=0, help="turn on label smoothing")
    ap.add_argument("--verbose", default="True",type=str,help="Print extra data")
    ap.add_argument("--modelcheckpoint", type=str, default=None ,help="path to *specific* model checkpoint to load")
    ap.add_argument("--startepoch", type=int, default=0, help="epoch to restart training at")
    ap.add_argument("--saveEpochRate", type=int, default=5, help="Frequency to save checkpoints")
    ap.add_argument("--opt", type=str, default="SGD", help="Type of optimizer")
    ap.add_argument("--augmentationLevel", type=int, default=0,help="turn on  Augmentation")
    ap.add_argument("--useOneNeuronForBinaryClassification", type=str, default="True",help="turn on  Augmentation")
    ap.add_argument("--display", type=str, default="True",help="turn on/off  display of plots")




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
    new_lr=args["new_lr"]
    labelSmoothing=args["labelSmoothing"]
    modelcheckpoint=args["modelcheckpoint"]    
    startepoch=args["startepoch"]
    saveEpochRate=args['saveEpochRate']
    opt=args['opt']
    augmentationLevel=args["augmentationLevel"]
    useOneNeuronForBinaryClassification=args["useOneNeuronForBinaryClassification"]
    display=args["display"]



        

    if(useOneNeuronForBinaryClassification=="True") :
        useOneNeuronForBinaryClassification=True
    else:
        useOneNeuronForBinaryClassification=False




    if(display=="True") :
        display=True
    else:
        display=False




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
    parameters=parameters+"[INFO] labelSmoothing".format(labelSmoothing) +"\n"








    if os.path.exists(ResultsFolder):
        print("[Warning]  Folder {} already exists, All files in folder will be deleted".format(ResultsFolder))
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
    print("[INFO] Training labels are  {}".format(str(labels)))
    print("[INFO] Number of classes are  {}".format(len(labels)))


    numOfOutputs=len(labels)  

    

    if(numOfOutputs==2) and useOneNeuronForBinaryClassification:  # binary classiffication problem  with 1 neuuron at last layer
        numOfOutputs=1
        lossFun=BinaryCrossentropy(label_smoothing=labelSmoothing)
        classMode='binary'
    else:
        lossFun = CategoricalCrossentropy(label_smoothing=labelSmoothing) 
        classMode='categorical'  #class_mode="categorical" will do one hot encoding
    



    #get statistics about your dataset
    NUM_TRAIN_IMAGES,NUM_TEST_IMAGES=paths.getTrainStatistics(datasetDir,train_dir,validation_dir)


    #draw sample images for training and  validation datasets 
    fileToSaveSampleImage=os.path.join(ResultsFolder,"sample_"+datasetDir+".png")

    #info,channels=plotUtil.drarwGridOfImages(base_dir,fileToSaveSampleImage,channels)
    info,channels=plotUtil.drarwGridOfImages(base_dir,fileToSaveSampleImage,display=display)
    print("[INFO] Number of input channels is {}".format(channels))




    folderNameToSaveBestModel="{}_Best_classifier".format(datasetDir)
    folderNameToSaveBestModel=os.path.join(ResultsFolder,folderNameToSaveBestModel)
    folderNameToSaveModelCheckPoints=os.path.join(ResultsFolder,"checkPoints")
    os.mkdir(folderNameToSaveModelCheckPoints)
    os.mkdir(folderNameToSaveBestModel)
    plotPath=os.path.join(ResultsFolder,"onlineLossAccPlot.png")
    jsonPath=os.path.join(ResultsFolder,"history.json")




    if(channels==1 and networkID not in ["Resnet50","VGG16"]):
        colorMode="grayscale"
    else:
        colorMode="rgb"
        channels=3


    if(networkID  in ["Resnet50","VGG16"]):   
        colorMode="rgb"
        channels=3




    if modelcheckpoint is None:
        #build the model
        model=modelsFactory.ModelCreator(numOfOutputs,width,height,networkID=networkID,channels=channels).model


        #setup optimizer
        if (opt=="RMSprop"):
            opt=RMSprop(learning_rate=learningRate, rho=0.9)
        elif(opt=="Adam"):
            opt=Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif(opt=="SGD"):
            opt=SGD(learning_rate=learningRate, momentum=0.0, nesterov=False)


        #compile model

        opt = Adam(lr=learningRate, decay=learningRate / EPOCHS)


        model.compile(optimizer=opt,loss=lossFun,metrics = ['accuracy'])


        
    else:   # otherwise, we're using a checkpoint model
    # load the checkpoint from disk
        print("[INFO] loading {}".format(modelcheckpoint))
        model = load_model(modelcheckpoint)
        # update the learning rate
        print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
        K.set_value(model.optimizer.lr, new_lr)
        print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
        print("[INFO] Model loaded  sucessfully from checkpoint")
        input("[MSG] Press any key to continue")

        #copy previous histort data of val and accuracy
        folderOfPerviousCheckPoint=os.path.dirname(modelcheckpoint)
        dst=ResultsFolder
        src=os.path.join(folderOfPerviousCheckPoint, "history.json" )
        shutil.copy(src,dst)




    model.summary()

    filenameToSaveModelSummary=os.path.join(ResultsFolder,networkID+"_modelSummary.txt")
    # Save summary to txt file
    with open(filenameToSaveModelSummary,'w') as fh:
     # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


    fileNameToSaveBestModel=os.path.join(folderNameToSaveBestModel,"best_classifier_"+datasetDir+".h5")

    earlyStopping = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0 ,  patience=patience , verbose=1)
    modelCheckpoint = ModelCheckpoint(fileNameToSaveBestModel, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)   #The format is inferred from the file extension you provide: if it is ".h5" or ".keras", the framework uses the Keras HDF5 format
    tensorboard_callback = TensorBoard(log_dir=ResultsFolder,profile_batch=0)
    epochCheckpoint=EpochCheckpoint(folderNameToSaveModelCheckPoints, every=saveEpochRate,startAt=startepoch)
    trainingMonitor=TrainingMonitor(plotPath,jsonPath=jsonPath,startAt=startepoch)



    fileToSaveModelPlot=os.path.join(ResultsFolder,'model.png')
    plot_model(model, to_file=fileToSaveModelPlot,show_shapes="True")
    print("[INFO] Model plot  saved to file  {} ".format(fileToSaveModelPlot))





#color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb"

    if(augmentationLevel==2):

        train_datagen = ImageDataGenerator(
              rescale=1./255,   #All images will be rescaled by 1./255
              rotation_range=40,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest')
   
    elif(augmentationLevel==1):
        train_datagen = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest",
        rescale=1./255, )

    else:    
        train_datagen = ImageDataGenerator(
                  rescale=1./255,   #All images will be rescaled by 1./255
                  )

 
        


    test_datagen  = ImageDataGenerator( rescale = 1.0/255. )



    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=BS,
                                                        class_mode=classMode,   #class_mode="categorical" will do one hot encoding
                                                        target_size=input_shape,color_mode=colorMode)     




    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                             batch_size=BS,
                                                             class_mode  = classMode,
                                                             target_size = input_shape,color_mode=colorMode)



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
                                  verbose=1 ,
                                  callbacks=[earlyStopping, modelCheckpoint,tensorboard_callback,trainingMonitor,epochCheckpoint]
                                  )


    
  

    #save model

    if(classMode=='binary'):
        fileNameToSaveModel="{}_{}_binaryClassifier.h5".format(labels[0],labels[1])
    else:
        fileNameToSaveModel="{}_Classifier.h5".format(datasetDir)


    fileNameToSaveModel=os.path.join(ResultsFolder,fileNameToSaveModel)
    model.save(fileNameToSaveModel ,save_format='h5')


    # save the model to disk
    folderNameToSaveModel="{}_Classifier".format(datasetDir)
    folderNameToSaveModel=os.path.join(ResultsFolder,folderNameToSaveModel)
    model.save(folderNameToSaveModel,save_format='tf') #model is saved in TF2 format (default)





    #plot and save training curves 
    title=datasetDir

    fileToSaveLossAccCurve=os.path.join(ResultsFolder,title+"plot_loss_accu.png")
    plotUtil.plotAccuracyAndLossesonSameCurve(history,title,fileToSaveLossAccCurve,display=display)

    fileToSaveAccuracyCurve=os.path.join(ResultsFolder,title+"plot_acc.png")
    fileToSaveLossCurve=os.path.join("Results",title+"plot_loss.png")
    plotUtil.plotAccuracyAndLossesonSDifferentCurves(history,title,fileToSaveAccuracyCurve,fileToSaveLossCurve,display=display)
   


    #copying history.json to model chechpoint folder
    dst=folderNameToSaveModelCheckPoints
    src=os.path.join(ResultsFolder, "history.json" )
    shutil.copy(src,dst)



    modelFile=fileNameToSaveModel
    root_dir="TestImages"
      

    if (testDir is not None):
        path_test=os.path.join(root_dir,testDir)
    else:
        path_test=validation_dir
    
    modelEvaluator=ModelEvaluator(modelFile,labels,input_shape,ResultsFolder,path_test,datasetDir,channels)
    modelEvaluator.evaluateGenerator()  #using sklearn & testGenerator



    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

    print("[INFO] Evaluation finished. Confusion matrix plot is now shown")
    print("*************************************************************************************************************")      






    print("[INFO] Loss and accuracy  curve saved to {}".format(fileToSaveLossAccCurve))
    print("[INFO] Loss curve saved to {}".format(fileToSaveLossCurve))
    print("[INFO] Accuracy  curve saved to {}".format(fileToSaveAccuracyCurve))
    print("[INFO] Best Model saved  to  {} as h5 file".format(fileNameToSaveBestModel))
    print("[INFO] Model check points saved to folder  {}  each  {} epochs ".format(folderNameToSaveModelCheckPoints,saveEpochRate))
    print("[INFO] Final model saved  to folder {} in both .h5  as {} and TF2 format".format(folderNameToSaveModel,fileNameToSaveModel))
    print("[INFO] Sample images from dataset saved to file  {} ".format(fileToSaveSampleImage))
    print("[INFO] History of loss and accuracy  saved to file  {} ".format(jsonPath))
    print("[INFO] Class labels encoded  as follows {}".format(labeles_dictionary))  
    print("[INFO] Model Summary  written to file {}".format(filenameToSaveModelSummary))  
    print("[INFO] Final  training accuracy {}".format(acc[EPOCHS-1]))
    print("[INFO] Final  val accuracy {}".format(val_acc[EPOCHS-1]))    
    print("[INFO] Final  training loss {}".format(loss[EPOCHS-1]))  
    print("[INFO] Final  val loss {}".format(val_loss[EPOCHS-1]))   
    print("[INFO] Model summary  written to file {}".format(filenameToSaveModelSummary))  



    print("*************************************************************************************************************")      







