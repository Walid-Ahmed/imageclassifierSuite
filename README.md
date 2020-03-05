# image classifier Suite

 

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/result_dog.png)


## Introduction 

This Repo can be used to train standard  datasets  or user defined dataset with different network structures. I tried  to include most of the nice features I learned in my deep journey for image classification. Training parmeters that can be set are  batch size, type of optimizer, image input size, early stopping, data augmentation, label smoothing, learning rate schedule,..... 


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

## Available Datasets 

The repo comes loaded with following datasets (all in folder "datasets"):
 1. Santa/NoSanta: initially collected  by  Adrian Rosebrock
 2. Dogs/Cats
 3. Human/Horses
 4. SportsClassification:22 types of sports in a total of 14,405 images , originally from this [link](https://github.com/anubhavmaity/Sports-Type-Classifier)). The type of sports are Swimming
,Badminton,Wrestling,Olympic Shooting,Cricket,Football,Tennis,Hockey,Ice Hockey,Kabaddi,WWE,Gymnasium,Weight lifting,Volleyball,Table tennis,Baseball,Formula 1,Moto GP,Chess,Boxing,FencingBasketbal]
 5. Smile/noSmile datset:originally from this [link](https://github.com/hromi/SMILEsmileD)  
 6. Food5K: a [Kaggle](https://www.kaggle.com/binhminhs10/food5k)
dataset containing 2,500 food and 2,500 non-food images, originally from this [link](https://www.kaggle.com/binhminhs10/food5k/download))
 7. NIH malaria dataset:
 The dataset consists of 27,588 images belonging to two separate classes: Parasitized/ Uninfected.
The number of images per class i 13,794 images per each. The dataset is  originally from this [link](https://lhncbc.nlm.nih.gov/publication/pub9932))

The following tree structure represents the current datasets structure in repo 

 ![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/treeStructure.png)


## Available network structures 

A very importnat file is modelsRepo.modelsFactory.py this file includes the definition of more than 5 deep neural networks  each given a special id that can you pass when you start training. These  networks include 

 - Resnet50   
 - Lenet
 - VGG16  
 - miniVGG as defined  in this [link](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning)
 
Beside these networks, the modelsFactory define some usefull neural networks that can be used  in classification.
More and more networks will be added soon, however you can still define your own, add to this file and start training with it!







## Training and Testing Scripts 

When the training starts, it will show sample of images and print  statistics about the dataset.  The training  script will stop automatically if the  validation accuracy is not improving after a patience number of epochs(default 50).

After finishing training, the following files are automatically saved to a  "Results" folder you pss as argument when you start training.


 1. Loss and accuracy curves
 2. The model as a .h5 file and .pb file in a folder with same name as dataset 
 3. The best model (highest accuracy) during training
 4. The labels in dictionary stored as a pickle file
 5. Confusion matrix as image
 


 
 When training starts, it will show a thumbnail image  for sample images from training dataset like the following one:

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/sample_CIFAR10.png)

 A sample confusion matrix  image  is saved as the following![Sample Arrangment of dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/MNIST_ConfusionMatrix.png)

Also training and validation losses and accuray curves are  plotted to tensorboard, you can view them during training by running the command 
```

tensorboard --logdir Results
```

 [I-Train a CIFAR10](#TRAINcIFAR)  </br>
 [II-Train a binary image classifier using flow from directory](#binaryimageclassifierusingflowfromdirectory)</br>
 [III--Train a binary/multiclass image classifier](#multiclassimageclassifier) </br>
 [IV-Train a multiclass image classifier using satandard dataset](#Trainmulticlassimageclassifierusingsatandarddataset)</br>
 [V-Test a Binary Model](#TestBinaryModels)


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


<h2 id="binaryimageclassifierusingflowfromdirectory">II-Train an image classifier using flow from directory</h2>


The 
trainBinaryClassifer_flow_from_directory trains a neural network with final layer of one neuron that is suitable for binary classification.

The data should have been splitted from the start to train and eval, for example the cat_and_dogs dataset is arranged as in following figure:

![structure for cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/cats_and_dogs_dir.png)

To start  training  using this file on "cats and dogs " dataset you can run the follwing command:
```

python trainClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 10  --width  150 --height  150 --testDir test_images_cats_and_dogs --ResultsFolder  Results/r1_cats_dogs
```







To train your dataset, it is is super easy, just add the folder of your images to the folder "datasets".
Your folder of images  should have two sub folders "train" and "eval".In each of the "train" and "eval" folder, you should have 2 subfolders, each labeled with the name of the class. 
 
A probabilty more than 0.5 means that the output is the second  class when they are sorted aphabitically. For example  predicting  the class from "cats" and "dogs" labels, the probabilty of more than 0.5  means a prediction of "dog".



![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/treeStructure.png)

<h1 id="multiclassimageclassifier">III--Train a binary/multiclass image classifier</h3>


The script   trainClassifier_flow_from_data.py   can be usd to train a binary classifier or a multi classifier. 


For binary classification with 2 classes only, a probabilty more than 0.5 means that the output is the second  class when they are sorted alphabetically. For example  predicting  the class from "cats" and "dogs" labels, the probabilty of more than 0.5  means a prediction of "dog".

To train your dataset, it is is super easy, just add the folder of your images to the folder "datasets".
Your folder of images  should have two sub folders "train" and "eval". In each of the "train" and "eval" folder, you should have  subfolders, each labeled with the name of the class. 
 


<h1 id="multiclassimageclassifier">III-Train a binary/multiclass image classifier</h1>


The script   trainClassifier_flow_from_data.py   can be used to train a binary classifier or a multi classifier. 

You can run it as follows


    python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 28 --height 28 --datasetDir Santa --networkID LenetModel



In case of binary classifier, the last layer will have only one neuron, otherwise  the last laye will have a number of neurons as the number of outputs, The activation  function in  last layer will be changed from Sigmoid to Softmax accordingly

You do not have to enter your labels or to split your data into train/eval, all what you have to do is to arrange your images so that each class in a folder with its label and all theses folders within a single folder as the following, the name of this single folder is what you should pass as argument when training. The folder should be in folder datasetes.
 ![Sample Arrangment of dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/classArrang.png)





## IV-Train a multiclass image classifier using satandard dataset . 


The file python trainStandardDatasetMulticlass.py trains a multiclass neural network using a standard datasets that are built in in Keras(but beware if you are behind proxy as u might have problems downloadind data!).


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




![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/sample_fashion_mnist.png)

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/sample_CIFAR10.png)

 
 A sample confusion matrix  image saved is as the following![Sample Arrangment of dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/MNIST_ConfusionMatrix.png)








# VI-Train on large dataset using Keras fit_generator
Not all the datasets that you will be using for training will be small to fit your memory. For example the the sport classification dataset(included in this repo ) has 14,405   images and  resized to only 224px*224px. If you try to load this  whole dataset to memory in any list like structure , you will most likely face memory issues.

```
python trainClassifier_flow_from_large_data.py    --EPOCHS 25   --width 224 --height 224 --channels 3  --datasetDir SportsClassification --networkID Resnet50 --BS 16  --verbose True
```


# VII-Others
## Use K-NN to build a classifier cat vs dog
jupyter notebook catsVsDog_imageClassification_K_Nearest_Neighbourhood.ipynb


# VIII-Train with your own dataset

To train your dataset, it is is super easy, just add the folder of your images to the folder "datasets".
Your folder of images  should have two sub folders "train" and "eval". In each of the "train" and "eval" folder, you should have  subfolders, each labeled with the name of the class. 



<h2 id="TestBinaryModels">V-Test Binary Models</h2>


You can test the binary model using the script in test_network_binary.py. The results will be displayed as an image with the predicted label typed on it. It will also be   saved with the name as file precceded with "prediction_"  in Results folder

```
python test_network_binary.py --model Results/cats_dogs_binaryClassifier.keras2 --image TestImages/test_images_cats_and_dogs/cats/cat_44.jpeg  --width  150 --height  150 --labelPKL Results/cats_and_dogs_labels.pkl 
```


Note the pkl file is the one created for you by trainClassifier_flow_from_data.py. It contains a dictionary like this  {'cats': 0, 'dogs': 1}

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/result_cat.png)

# IX-Utilities

The script split_dataset.py  can be used to split image files ofteh dataset to training and validation based on split percentage you choose, for example you can run it as follows
```
python split_dataset.py   --dataset Cyclone_Wildfire_Flood_Earthquake_Database  --TRAIN_SPLIT 0.7
```
