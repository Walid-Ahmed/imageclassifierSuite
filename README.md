# image classifier Suite

The Repo already include these dataset

1- Santa/NoSanta     (initially collected  by  Adrian Rosebrock) 
2-Dogs/Cats
3-Human/Horses
4-Sport classiffication   # dataset was curated by Anubhav Maity by downloading photos from Google Images

The prediction model have only one neuron at last layer. A probabilty more than 0.5 means that the output is the first class(Santa/Dogs/Human) otherwise it is the second class (No Santa/Cats/Horse)


python train_BinaryClassiffer_flow_from_data.py. # train binary image classiffier  from data NOT distributed to train and eval
python trainBinaryClassiffer_flow_from_directory.py # train binary image classiffier  from data distributed to train and eval



python trainClassiffer_flow_from_data.py  # train image classiffier  # train image classiffier  from data distributed to train and eval
