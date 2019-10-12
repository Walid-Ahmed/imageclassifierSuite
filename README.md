# image classifier Suite


This Repo can be used to train standard keras dataset  or used defined dataser with different network structures. I tried  to include most of the nice features I learned in my deep journey for image classification.

The repo comes loaded with following datasets (all in folder datasets):
1-Santa/NoSanta     (initially collected  by  Adrian Rosebrock) 
2-Dogs/Cats
3-Human/Horses

A very importnat file is modelsRepo.modelsFactory.py this file includes the definition of more than 5 deep neural networks  each given a special id that can you pass when you start training. This newworks include 

 - Resnet50   
 - Lenet
 - VGG16  
 - miniVGG as defined  in this [link](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning)

**I-Train a binary image classiffier using flow from directory **

The 
trainBinaryClassifer_flow_from_directory builds a neural network with final layer of one neuron

To start  training  using this file on "cats and dogs " dataset you can run the follwing command:

python trainBinaryClassifer_flow_from_directory.py  --datasetDir cats_and_dogs --networkID net2  --EPOCHS 100  --width  150 --height  150 --testDir test_images_cats_and_dogs

To start  training  using this file on "cats and dogs" dataset you can run the follwing command:

python trainBinaryClassifer_flow_from_directory.py  --datasetDir horse-or-human --networkID net1  --EPOCHS 2  --width  300 --height  300 --testDir test_horses_or_Human

When the training starts it will show sample of images and print some statistics about the dataser. after finishing, loss and accuracy curves  together with the model are saved in the results folder 

**II-Train a binary image classiffier using flow from directory **


python trainStandardDatasetLeNet.py --dataset MNIST  --networkID  LenetModel --EPOCHS 20 .  <br />
python trainStandardDatasetLeNet.py  --dataset fashion_mnist --networkID MiniVGG --EPOCHS 25  <br />
python trainStandardDatasetLeNet.py  --dataset CIFAR10 --networkID net5  --EPOCHS 25    #val_acc: 0.8553  <br />
python trainStandardDatasetLeNet.py  --dataset CIFAR100 --networkID MiniVGG  --EPOCHS 25  #val_acc: 0.5397  <br />


The Repo already include these dataset




The prediction model have only one neuron at last layer. A probabilty more than 0.5 means that the output is the first class(Santa/Dogs/Human) otherwise it is the second class (No Santa/Cats/Horse)


python train_BinaryClassiffer_flow_from_data.py
python trainBinaryClassiffer_flow_from_directory.py.   


python test_network.py --model Results/not_santa_santa_binaryClassifier.keras2  --image TestImages/test_images_Santa_and_noSanta/santa_01.png --labelPKL Results/Santa_labels.pkl
