# image classifier Suite

This Repo can be used to train standatd keras dataset with different network structures

python trainStandardDatasetLeNet.py --dataset MNIST  --networkID  LenetModel --EPOCHS 20 .  <br />
python trainStandardDatasetLeNet.py  --dataset fashion_mnist --networkID MiniVGG --EPOCHS 25  <br />
python trainStandardDatasetLeNet.py  --dataset CIFAR10 --networkID net5  --EPOCHS 25    #val_acc: 0.8553  <br />
python trainStandardDatasetLeNet.py  --dataset CIFAR100 --networkID MiniVGG  --EPOCHS 25  #val_acc: 0.5397  <br />


The Repo already include these dataset

1-Santa/NoSanta     (initially collected  by  Adrian Rosebrock) 
2-Dogs/Cats
3-Human/Horses


The prediction model have only one neuron at last layer. A probabilty more than 0.5 means that the output is the first class(Santa/Dogs/Human) otherwise it is the second class (No Santa/Cats/Horse)


python train_BinaryClassiffer_flow_from_data.py
python trainBinaryClassiffer_flow_from_directory.py.   


python test_network.py --model Results/not_santa_santa_binaryClassifier.keras2  --image TestImages/test_images_Santa_and_noSanta/santa_01.png --labelPKL Results/Santa_labels.pkl
