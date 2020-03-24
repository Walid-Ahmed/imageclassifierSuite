# Image classifier Suite
 
 
 ## Introduction 

 Multiclass or multinomial classification is the problem of classifying instances into one of three or more classes. Classifying instances into one of two classes is called binary classification
 
This Repo can be used to train an image  classifier   on user defined dataset or on standard  datasets with different network structures. I tried  to include most of the nice features I learned in my deep journey for image classification. 

I owe a lot of what I learened to Adrian Rosebrock [blog](https://www.pyimagesearch.com/)

Training parmeters that can be set are:
- Batch size, 
- Type of optimizer, 
- Image input size, 
- Early stopping, 
- Data augmentation, 
- Label smoothing, 
- Learning rate schedule
 
 
The repo enables you also to stop training and restart  it again from a checkpoint withe a new learning rate. Checkpoints of model as .h5 saved each 5 epochs(You can change this value)



<p align="center">

 <img src="https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/animatedGif.gif">
</p>




# Install

Here is the list of libraries you need to install to execute the code:
- python v3.6
- Tesnsorflow. v2.0.0
- numpy v1.16.1
- sklearn v0.22
- matplotlib v3.0.2
- scikit-image
- jupyter
- pickle

## Available Datasets  For Binary classification

The repo comes loaded with following datasets (all in folder "datasets"):
 
 1. Santa/NoSanta: initially collected  by  [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/)
 2. Dogs/Cats
 3. Human/Horses
 4. Smile/noSmile dataset:originally from this [link](https://github.com/hromi/SMILEsmileD)  
 5. Food5K: a [Kaggle](https://www.kaggle.com/binhminhs10/food5k)
dataset containing 2,500 food and 2,500 non-food images, originally from this [link](https://www.kaggle.com/binhminhs10/food5k/download))
 6. NIH malaria dataset:
 The dataset consists of 27,588 images belonging to two separate classes: Parasitized/ Uninfected.
The number of images per class i 13,794 images per each. The dataset is  originally from this [link](https://lhncbc.nlm.nih.gov/publication/pub9932))
7. CoronaVirus (pos_covid vs ned_covid) the dataset is  originally from this [link](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/))


## Available Datasets  For  Multiclass or multinomial classification

8. FacialExpression
9. spatial_envelope_256x256_static_8outdoorcategories
10. Cyclone_Wildfire_Flood_Earthquake_Database
11. SportsClassification:22 types of sports in a total of 14,405 images , originally from this [link](https://github.com/anubhavmaity/Sports-Type-Classifier)). The type of sports are Swimming
,Badminton,Wrestling,Olympic Shooting,Cricket,Football,Tennis,Hockey,Ice Hockey,Kabaddi,WWE,Gymnasium,Weight lifting,Volleyball,Table tennis,Baseball,Formula 1,Moto GP,Chess,Boxing,FencingBasketbal]


The following tree structure represents the current datasets structure in repo 

 ![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/treeStructure.png)


## Available network structures 

A very importnat file is modelsRepo.modelsFactory.py this file includes the definition of more than 5 deep neural networks,  each given a special id that  you can pass when you start training. Plots of all models are saved to folder modelsPlots  

These  networks include:

 - Resnet50   
 - Lenet
 - VGG16  
 - miniVGG as defined  in this [link](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning)
 - DPN as defined  in this [link](https://github.com/titu1994/Keras-DualPathNetworks)
 - net2 as defined in this [link](https://www.coursera.org/lecture/convolutional-neural-networks-tensorflow/training-with-  the-cats-vs-dogs-dataset-jV6tw)
 - net3 ad defined in this [link](https://ermlab.com/en/blog/nlp/cifar-10-classification-using-keras-tutorial/)
 - net4 as defined in this [link]( https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py)
 - net5 as defined in this [link](https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/)
 
Beside these networks, the modelsFactory defines some usefull neural networks that can be used  in classification.
More and more networks will be added soon, however you can still define your own, add to this file and start training with it!

Yoi can run the folllowing command to explore the available models
```
python modelsRepo/modelsFactory.py
```

| Network I.D. | Number of Parmeters |Size of Input(H,W,C)|
|--------------|---------------------|------------|
| net1         | 1704610             |(300,300,3)|
| net2         | 9495074             |(150,150,3)|
| net3         | 2395434             |(32,32,3)|
| net4         | 150148              |(32,32,3)|
| net5         | 309290              |(32,32,3)|
| LenetModel   | 1631080             |(32,32,1)|
| Resnet50     | 25149800            |(224,224,3)|
| MiniVGG      | 18945829            |(96,96,3)|
| VGG16        | 14812520            |(224,224,3)|
| DPN          | 37769632            |(224,224,3)|




 ![Network parameters](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/numOfParametersPerNetwork.png)

 ![Model Plots](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/modelsPlots.gif)

## Training and Testing Scripts 


For binary classifier, you can choose if the last layer will have a single neuron and in this case  probabilty more than 0.5 means that the output is the second  class when they are sorted aphabitically. For example  predicting  the class from "cats" and "dogs" labels, the probabilty of more than 0.5  means a prediction of "dog".  
For mutlti class classisifier,the last layer will have a number of neurons as the number of outputs, The activation  function in  last layer will be changed from Sigmoid to Softmax accordingly.

When the training starts, it will show sample of images and print  statistics about the dataset.  The training  script will stop automatically if the  validation accuracy is not improving after a patience number of epochs(default 50).
You do not need to worry about whether your are training with color images or gray scale images as the number of channels is detected automatically.

```
optional arguments:
  -h, --help            show this help message and exit
  --datasetDir DATASETDIR
                        path to dataset directory with train and validation
                        images
  --testDir TESTDIR     path to test directory with test images
  --networkID NETWORKID
                        I.D. of the network, it can be any of [net1,net2,net3,
                        net4,net5,LenetModel,Resnet50,net3,MiniVGG,VGG16]
  --EPOCHS EPOCHS       Number of maximum epochs to train
  --BS BS               Batch size
  --width WIDTH         width of image
  --height HEIGHT       height of image
  --patience PATIENCE   Number of epochs to wait without accuracy improvment
  --ResultsFolder RESULTSFOLDER
                        Folder to save Results
  --lr LR               Initial Learning rate
  --new_lr NEW_LR       restarting Learning rate
  --labelSmoothing LABELSMOOTHING
                        turn on label smoothing
  --verbose VERBOSE     Print extra data
  --modelcheckpoint MODELCHECKPOINT
                        path to *specific* model checkpoint to load
  --startepoch STARTEPOCH
                        epoch to restart training at
  --saveEpochRate SAVEEPOCHRATE
                        Frequency to save checkpoints
  --opt OPT             Type of optimizer
  --augmentationLevel AUGMENTATIONLEVEL
                        can be either  0 or 1 or 2
  --useOneNeuronForBinaryClassification USEONENEURONFORBINARYCLASSIFICATION
                        turn on Augmentation
  --display DISPLAY     turn on/off display of plots
  
  ```

After finishing training, the following files are automatically saved to a  "Results" folder you pass as argument when you start training.


 1. Loss and accuracy curves
 2. The model as a .h5 file and .pb file in a folder with same name as dataset 
 3. The best model (highest accuracy) during training as .h5 file
 4. The labels in dictionary stored as a pickle file
 5. Confusion matrix as image
 6. Precision Recal curve (for binary classification only )
 7. F1-score vs Threshould (for binary classification only)
 8. History of accuracy and loss for training and valisation as a json file (history.json)
 9. A plot for acc and accuracy that is being updated each epoch (onlineLossAccPlot.png), this will also take into account any training done before when training starts from a previous checkpoint.
 10. Checkpoints of model as .h5 saved each 5 epochs(You can change this value) 
 


 
Also when training starts, it will show a thumbnail image  for sample images from training dataset  Also training and validation losses and accuray curves are  plotted to tensorboard, you can view them during training by running the command 
```

tensorboard --logdir Results
```
and then browsing the following url http://localhost:6008/






Here are the basic scripts for training and testing


 [I-Train a CIFAR10/MNIST dataset](#TRAINcIFAR)  </br>
 
 [II-Train an image classifier using flow from directory](#binaryimageclassifierusingflowfromdirectory)</br>
 
 [III-Train a binary/multiclass image classifier using flow from data](#multiclassimageclassifier) </br>
 
 [IV-Train a multiclass image classifier using satandard dataset](#Trainmulticlassimageclassifierusingsatandarddataset)</br>
 
 [V-Predict using a Model](#TestBinaryModels)


<h2 id="TRAINcIFAR">I-Train on CIFAR10/MNIST dataset</h2>

You can start  testing your environment by training  a model for the   CIFAR10 dataset  just by excuting the command
```
python trainCIFAR10.py 
```
This code will download the CIFAR10 dataset(if needed) and start training using a deep convloution neural network. When it finishes training, results will be shown. You can also  run  the code in your browser with the command   "ipython notebook trainCIFAR10.ipynb".
 
 You can also train the mnist dataset  with any of the following: 
```
 ipython notebook  notebooks/mnist-CNN.ipynb
 ```

 
``` 
 ipython notebook  notebooks/mnist_without_CNN.ipynb
```


<h2 id="binaryimageclassifierusingflowfromdirectory">II-Train a binary/multiclass(multinomial)  image classifier using flow from directory</h2>


The 
trainClassifer_flow_from_director.py  trains a neural network with final layer of one neuron  in case of  binary classification(2 classes)  otherwise the number of neurons in last layer will be equal to number of classes in case we are training a multiclass image classifier.

The data should have been **splitted** earlier in the folder to train and eval, for example the cat_and_dogs dataset is arranged as in following figure:

![structure for cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/cats_and_dogs_dir.png)

To start  training  using this file on "cats and dogs" dataset you can run the follwing command:
```

python trainClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 25  --width  150 --height  150  --ResultsFolder  Results/r1_cats_dogs --labelSmoothing 0.1
```

To start  training  using this file on "Facial Expression" dataset you can run the follwing command:

```
python trainClassifer_flow_from_directory.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 80  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r1_FacialExpression 
```
To train your dataset using this script, it is is super easy, just add the folder of your images to the folder "datasets".
Your folder of images  should have two sub folders "train" and "eval". In each of the "train" and "eval" folder, you should have 2 subfolders, each labeled with the name of the class. 
if your datas is not splitted already, you can use  util/split_dataset.py   to split it
 



![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/treeStructure.png)
 


<h1 id="multiclassimageclassifier">III-Train a binary/multiclass image classifier using flow from data</h1>


The script   trainClassifier_flow_from_data.py   can be used to train a binary classifier or a multi class classifier. 

You can run it as follows

```

python trainClassifier_flow_from_data.py    --EPOCHS 15   --width 28 --height 28  --datasetDir Santa --networkID LenetModel --verbose False --ResultsFolder  Results/r2_santa --augmentationLevel 2 --useOneNeuronForBinaryClassification True   --opt Adam
```


```

python trainClassifier_flow_from_data.py    --EPOCHS 15   --width 28 --height 28  --datasetDir Santa --networkID LenetModel --verbose False --ResultsFolder  Results/r2_santa --augmentationLevel 2 --useOneNeuronForBinaryClassification False   --opt Adam
```


```

python trainClassifier_flow_from_data.py  --datasetDir FacialExpression --networkID net2  --EPOCHS 25  --width  48 --height  48  --BS 32  --ResultsFolder  Results/r29_FacialExpression   --augmentationLevel 1 --opt Adam
```


```

python trainClassifier_flow_from_data.py  --datasetDir Cyclone_Wildfire_Flood_Earthquake_Database --networkID Resnet50  --EPOCHS 25  --width  224 --height  224  --BS 32  --ResultsFolder  Results/r22_Cyclone_Wildfire_Flood_Earthquake_Database  --augmentationLevel 1 --opt Adam
```


You do not have to enter your labels or to split your data into train/eval, all what you have to do is to arrange your images so that each class in a folder with its label and all theses folders within a single folder as the following, the name of this single folder is what you should pass as argument when training. The folder should be in folder datasetes.
 ![Sample Arrangment of dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/classArrang.png)





<h2 id="Trainmulticlassimageclassifierusingsatandarddataset">IV-Train a multiclass image classifier using satandard dataset </h2>


The  python script trainStandardDatasetMulticlass.py trains a multiclass neural network using one of the  standard datasets that are built in Keras(but beware if you are behind proxy as you might have problems downloading data!).

Some of these standard datasets are:
 1. MNIST
 2. Fashion_mnist
 3. CIFAR10
 4. CIFAR100

All what you have to do is to pass the i.d. of the dataset, together with the i.d. of the neural network you want to use in training. The network is build to have the last layer with the correct number of neurons to fit  the dataset.

Some of the sample commands you can run are:

    python trainStandardDatasetMulticlass.py --dataset MNIST  --networkID  LenetModel --EPOCHS 20
    python trainStandardDatasetMulticlass.py  --dataset fashion_mnist --networkID MiniVGG --EPOCHS 25  
    python trainStandardDatasetMulticlass.py  --dataset CIFAR10 --networkID net5  --EPOCHS 25  
    python trainStandardDatasetMulticlass.py  --dataset CIFAR100 --networkID MiniVGG  --EPOCHS 25 









# VI-Train on large dataset using Keras fit_generator
Not all the datasets that you will be using for training will be small to fit your memory. For example the the sport classification dataset(included in this repo ) has 14,405   images and  resized to only 224px*224px. If you try to load this  whole dataset to memory in any list like structure , you will most likely face memory issues.

```
python trainClassifier_flow_from_large_data.py    --EPOCHS 25   --width 224 --height 224 --channels 3  --datasetDir SportsClassification --networkID Resnet50 --BS 16  --verbose True
```


# VII-Others
## Use K-NN to build a classifier cat vs dog
jupyter notebook catsVsDog_imageClassification_K_Nearest_Neighbourhood.ipynb





<h2 id="TestBinaryModels">V-Predict using a Model </h2>


You can test the binary classifier mode lwith one neuron at last layer  using the script in **test_network_binary.py**. 
for multiclass classifier , you can use the script **test_network_multiClassifier.py**

The results will be displayed as an image with the predicted label typed on it. It will also be   saved with the name as file precceded with "prediction_"  in "Results" folder

```
python test_network_binary.py --model Results/r2_santa/Santa_Classifier.h5  --image TestImages/test_images_Santa_and_noSanta/santa_01.png  --width  28 --height  28 --labelPKL Results/r2_santa/Santa_labels.pkl 
```
```
python test_network_binary.py --model Results/r2_santa/Santa_Classifier.h5  --image TestImages/test_images_Santa_and_noSanta/night_sky.png  --width  28 --height  28 --labelPKL Results/r2_santa/Santa_labels.pkl 
```


Note the pkl file is the one created for you by trainClassifier_flow_from_data.py. It contains a dictionary like this {'not_santa': 0, 'santa': 1}

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/result_cat.png)


```
python test_network_multiClassifier.py  --model  Results/r1_disaster/Cyclone_Wildfire_Flood_Earthquake_Database_Classifier.h5  --image TestImages/test_Cyclone_Wildfire_Flood_Earthquake/earthquake_175.jpg --labelPKL    Results/r1_disaster/Cyclone_Wildfire_Flood_Earthquake_Database_labels.pkl    --width 150 --height 150
```
![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/prediction_earthquake_175.jpg)

# IX-Utilities

The script split_dataset.py  can be used to split image files of the  dataset to training and validation based on split percentage you choose, for example you can run it as follows
```
python util/split_dataset.py   --dataset Cyclone_Wildfire_Flood_Earthquake_Database  --TRAIN_SPLIT 0.7
```


# Credits
The animated gif was made using  Animated GIF Maker available at https://ezgif.com/maker

Tables in readme  file created withhelp from https://www.tablesgenerator.com/markdown_tables#
