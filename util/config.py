# import the necessary packages
import os
import shutil

target="HH_NHH"
foldersToCreate=["models","graphs","montage"]
for folderToCreate in foldersToCreate:
	shutil.rmtree(folderToCreate, ignore_errors=True, onerror=None)
	if not os.path.exists(folderToCreate):
        		os.makedirs(folderToCreate)
# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-1
BS= 32
#print(BS)
#netName="net11"

# initialize the path to the *original* input directory of images
#ORIG_INPUT_DATASET = "malaria/cell_images"
ORIG_INPUT_DATASET ="/Users/walidahmed/desktop/BackupDataSets/HardHat"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
#BASE_PATH = "malaria"
BASE_PATH = "/Users/walidahmed/desktop/BackupDataSets"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
print("Training path is {0}".format(TRAIN_PATH))
print("Validation path is {0}".format(VAL_PATH))
print("Testing path is {0}".format(TEST_PATH))

# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1