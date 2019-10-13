# image classifier Suite


This Repo can be used to train standard keras dataset  or used defined dataser with different network structures. I tried  to include most of the nice features I learned in my deep journey for image classification.

The repo comes loaded with following datasets (all in folder "datasets"):
 1. Santa/NoSanta     (initially collected  by  Adrian Rosebrock) 
 2. Dogs/Cats
 3. Human/Horses
 4. SportsClassification (originally from this [link])(https://github.com/anubhavmaity/Sports-Type-Classifier)

The following tree structure represents the current datasets structure in repo 

![Sample curve output from training cats vs dogs dataset](https://github.com/Walid-Ahmed/imageclassifierSuite/blob/master/sampleImages/treeStructure.png)

A very importnat file is modelsRepo.modelsFactory.py this file includes the definition of more than 5 deep neural networks  each given a special id that can you pass when you start training. These  networks include 

 - Resnet50   
 - Lenet
 - VGG16  
 - miniVGG as defined  in this [link](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning)
 
Besides these networks, the modelsFactory define some usefull neural networks that can be used  in classification.
More and more networks will be added soon, however you can still define your own, add to this file and start training with it!

# I-Train a CIFAR10

you can start by testing your environment by running python trainCIFAR10.py. This code will download the CIFAR10 dataset and start training using a deep convloution neural network. when it finishes training, results will be shown. You can also  run  the code in your browser with the command   "ipython notebook trainCIFAR10.ipynb"
 

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


# III-Train a multiclass image classiffier using satandard dataset


The file python trainStandardDatasetMulticlass.py trains a multiclass neural network using a standard datasets that are builtin in Keras.

Some of these standard dataset are:
 1. MNIST
 2. Fashion_mnist
 3. CIFAR10
 4. CIFAR100

All what you have to do is to pass the i.d. of the dataset, together with the i.d. of the neural network you want to use in training. The network is build to have the last layer with the correct number of neurons to fit  the dataset.

Some of the sample commands you can run are:

python trainStandardDatasetMulticlass.py --dataset MNIST  --networkID  LenetModel --EPOCHS 20 .  
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
 

# III-Train a multiclass image classiffier using flow_from_data


...TODO documentaiom 

# III-Test Model
python test_network.py --model Results/not_santa_santa_binaryClassifier.keras2  --image TestImages/test_images_Santa_and_noSanta/santa_01.png --labelPKL Results/Santa_labels.pkl

...TODO documentaiom 

