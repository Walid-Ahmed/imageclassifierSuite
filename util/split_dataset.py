# USAGE
# python split_dataset.py   --dataset Cyclone_Wildfire_Flood_Earthquake_Database  --TRAIN_SPLIT 0.7


# import the necessary packages
from imutils import paths
import random
import shutil
import os
import argparse


def createFolders(dataset):

	pathtoDataset=os.path.join("..","datasets",dataset)


	# derive the training, validation, and testing directories



	TRAIN_PATH = os.path.join(pathtoDataset, "train")
	VAL_PATH = os.path.join(pathtoDataset, "validation")


	#Remove  folder if exists
	shutil.rmtree(TRAIN_PATH, ignore_errors=True)
	shutil.rmtree(VAL_PATH, ignore_errors=True)









	labels = os.listdir(pathtoDataset)

	if  (".DS_Store" in labels):
		labels.remove(".DS_Store")

	if ("train" in labels):
		labels.remove("train")

	if ("validation" in labels):
		labels.remove("validation")




	print("****************")

	#print(labels)
	print("****************")

	for label in labels:
		#print(label)
		pathToCreateTrain=os.path.join(TRAIN_PATH,label)
		pathToCreateVal=os.path.join(VAL_PATH,label)

		if not os.path.exists(pathToCreateTrain):
			os.makedirs(pathToCreateTrain)
			print(pathToCreateTrain)

		if not os.path.exists(pathToCreateVal):
			os.makedirs(pathToCreateVal)	
			print(pathToCreateVal)	





def split(pathtoDataset,label,TRAIN_PATH,VAL_PATH):

	# grab the paths to all input images in the original input directory
	# and shuffle them
	
	print(pathtoDataset)
	imagePaths = list(paths.list_images(pathtoDataset))
	random.seed(42)
	random.shuffle(imagePaths)

	#print(imagePaths)  #datasets/Cyclone_Wildfire_Flood_Earthquake_Database/Flood/614.jpg



	# compute the training and testing split
	i = int(len(imagePaths) * TRAIN_SPLIT)
	trainPaths = imagePaths[:i]
	testPaths = imagePaths[i:]


	#create training split
	dst=os.path.join(TRAIN_PATH,label)

	for src in trainPaths:
		


		shutil.copy(src, dst)

		print("[INFO] file {} copied to {}".format(src,dst))
		
	
	#create Val split

	dst=os.path.join(VAL_PATH,label)

	for src in testPaths:
		


		shutil.copy(src, dst)

		print("[INFO' file {} copied to {}".format(src,dst))




	


if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()

	ap.add_argument("--dataset",  required=True,help="name of dataset")
	ap.add_argument("--TRAIN_SPLIT", type=float, default=0.7, required=True,help="percentatge of data to use in ytaining")

	args = vars(ap.parse_args())
	dataset=args['dataset']
	TRAIN_SPLIT=args['TRAIN_SPLIT']

	createFolders(dataset)



	pathtoDataset=os.path.join("..","datasets",dataset)

	if not os.path.exists(pathtoDataset):
		print(("[Error] directory {} does not exist".format(pathtoDataset)))
		exit()

	#createFolders(dataset)
	TRAIN_PATH = os.path.join(pathtoDataset, "train")
	VAL_PATH = os.path.join(pathtoDataset, "validation")

	

	labels = os.listdir(pathtoDataset)
	if  (".DS_Store" in labels):
		labels.remove(".DS_Store")

	if ("train" in labels):
		labels.remove("train")

	if ("validation" in labels):
		labels.remove("validation")




	print("****************")

	#print(labels)
	print("****************")
	

	for label in labels:
		pathToLabelImages=os.path.join("..","datasets",dataset,label)
		split(pathToLabelImages,label,TRAIN_PATH,VAL_PATH)


