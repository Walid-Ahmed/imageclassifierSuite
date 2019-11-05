# image classifier Suite


This Repo can be used to train standard keras dataset  or used defined dataset with different network structures. I tried  to include most of the nice features I learned in my deep journey for image classification.

The repo comes loaded with following datasets (all in folder "datasets"):
 1. Santa/NoSanta     (initially collected  by  Adrian Rosebrock) 
 2. Dogs/Cats
 3. Human/Horses
 4. SportsClassification(22 types of sports in a total of 14,405 images , originally from this [link](https://github.com/anubhavmaity/Sports-Type-Classifier))
 5. Smile/noSmile datset(originally from this [link](https://github.com/hromi/SMILEsmileD))   
 6. Food5K (a [Kaggle (https://www.kaggle.com/binhminhs10/food5k)
) dataset containing 2500 food and 2500 non-food images, originally from this [link](https://www.kaggle.com/binhminhs10/food5k/download))
 7. NIH malaria dataset(originally from this [link](https://lhncbc.nlm.nih.gov/publication/pub9932))

The following tree structure represents the current datasets structure in repo 

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/treeStructure.png)

A very importnat file is modelsRepo.modelsFactory.py this file includes the definition of more than 5 deep neural networks  each given a special id that can you pass when you start training. These  networks include 

 - Resnet50   
 - Lenet
 - VGG16  
 - miniVGG as defined  in this [link](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning)
 
Beside these networks, the modelsFactory define some usefull neural networks that can be used  in classification.
More and more networks will be added soon, however you can still define your own, add to this file and start training with it!

# I-Train a CIFAR10

You can start  testing your environment by running python trainCIFAR10.py. 
This code will download the CIFAR10 dataset and start training using a deep convloution neural network. When it finishes training, results will be shown. You can also  run  the code in your browser with the command   "ipython notebook trainCIFAR10.ipynb"
 

# II-Train a binary image classiffier using flow from directory

The 
trainBinaryClassifer_flow_from_directory trains a neural network with final layer of one neuron that is suitable for binary classification.

To start  training  using this file on "cats and dogs " dataset you can run the follwing command:

python trainBinaryClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 100  --width  150 --height  150 --testDir test_images_cats_and_dogs

To start  training  using this file on "cats and dogs" dataset you can run the follwing command:

python trainBinaryClassifer_flow_from_directory.py  --datasetDir horse-or-human --networkID net1  --EPOCHS 2  --width  300 --height  300 --testDir test_horses_or_Human

When the training starts it will show sample of images and print some statistics about the dataset. after finishing, the following files are automatically saved to the "Results" folder

 1. Loss curve
 2. Accuracy curve
 3. Loss and accuracy curves
 4. The model as a .keras2 file
 5. The labels in dictionary stored as a pickle file


To train your dataset, it is is super easy, just add the folder of your images to the folder "datasets".
Your folder of images  should have two sub folders "train" and "eval".In each of the "train" and "eval" folder, you should have 2 subfolders, each labeled with the name of the class. 
 
A probabilty more than 0.5 means that the output is the second  class when they are sorted aphabitically. For example  predicting  the class from "cats" and "dogs" labels, the probabilty of more than 0.5  means a prediction of "dog".



![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/plot_loss_accu.png)

# III--Train a binary/multiclass image classiffier 

The file   trainClassifier_flow_from_data.py   can be usd to train a binary classifier or a multi classifier. 

You can run it as follows


    python trainClassifier_flow_from_data.py    --EPOCHS 25   --width 28 --height 28 --datasetDir Santa --networkID LenetModel



In case of binary classifier, the last layer will have only one neuron, otherwise  the last laye will have a number of neurons as the number of outputs, The activation  function in  last layer will be changed from Sigmoid to Softmax accordingly

You do not have to enter your labels or to split your data into train/eval, all what you have to do is to arrange your images so that each class in a folder with its label and all theses folders within a single folder as the following, the name of this single folder is what you should pass as argument when training. The folder should be in folder datasetes.
 ![Sample Arrangment of dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/classArrang.png)




The following files are automatically saved to the "Results" folder

 1. Loss curve
 2. Accuracy curve
 3. Loss and accuracy curves
 4. The model as a .keras2 file
 5. The labels in dictionary stored as a pickle file
 6. Confusion matrix as an image
 
 A sample confusion matrix  image saved is as the following![Sample Arrangment of dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/MNIST_ConfusionMatrix.png)

# IV-Train a multiclass image classiffier using satandard dataset


The file python trainStandardDatasetMulticlass.py trains a multiclass neural network using a standard datasets that are built in in Keras(but beware if you are behind proxy as u might have problems downloadind data!).

Some of these standard dataset are:
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

when training starts, it will show a thumbnail image  for sample images from training dataset like the following ones:

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/sample_MNIST.png)


![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/sample_fashion_mnist.png)

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/sample_CIFAR10.png)

Each training will save the follwoing files in the "Results" folder

 1. Thumbnail image including sample images from the training dtataset
 2. The trained mode as a .keras2 file
 



# V-Test Binary Models

python test_network_binary.py --model Results/cats_dogs_binaryClassifier.keras2 --image TestImages/test_images_cats_and_dogs/cats/cat_44.jpeg  --width  150 --height  150 --labelPKL Results/cats_and_dogs_labels.pkl 

python test_network_binary.py --model Results/cats_dogs_binaryClassifier.keras2 --image TestImages/test_images_cats_and_dogs/dogs/dog_23.jpeg --labelPKL Results/cats_and_dogs_labels.pkl --width  150 --height  150

Note the pkl file is the one created for you by trainClassifier_flow_from_data.py. It contains a dictionary like this  {'cats': 0, 'dogs': 1}

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/result_cat.png)

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/result_dog.png)

# VI-Train on large ataset using Keras fit_generator
Not all the datasets that you will be using for training will be small to fit your memory. For example the the sport classification dataset( curated by Anubhav Maity) has xxx  images and only resized to 224px*224px. If you try to load this  whole dataset to memory in any list like structure , you will most likely face memory issues.

python trainClassifier_flow_from_large_data.py    --EPOCHS 25   --width 224 --height 224 --channels 3  --datasetDir SportsClassification --networkID Resnet50 --BS 16  --verbose True

# VII-Others
## Use K-NN to build a classifier cat vs dog
jupyter notebook catsVsDog_imageClassification_K_Nearest_Neighbourhood.ipynb



