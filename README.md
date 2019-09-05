# image classifier Suite

The Repo already include these dataset

1- Santa/NoSanta     (initially collected  by  Adrian Rosebrock)   &nbsp;
2-Dogs/Cats &nbsp;
3-Human/Horses &nbsp;


The prediction model have only one neuron at last layer. A probabilty more than 0.5 means that the output is the first class(Santa/Dogs/Human) otherwise it is the second class (No Santa/Cats/Horse)


python train_BinaryClassiffer_flow_from_data.py
python trainBinaryClassiffer_flow_from_directory.py.   


python test_network.py --model Results/not_santa_santa_binaryClassifier.keras2  --image TestImages/test_images_Santa_and_noSanta/santa_01.png --labelPKL Results/Santa_labels.pkl
