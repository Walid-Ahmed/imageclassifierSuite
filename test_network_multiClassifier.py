# USAGE


#python test_network_multiClassifier.py  --model Results/soccer2/soccer_Classifier.h5   --image TestImages/test_Cyclone_Wildfire_Flood_Earthquake/earthquake_175.jpg --labelPKL    Results/soccer2/soccer_labels.pkl    --width 32 --height 32

# import the necessary packages

import tensorflow  as tf

from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import imutils
import cv2
import pickle






# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-lbpkl", "--labelPKL", required=True,
	help="path to label list as pickle file")
ap.add_argument("--width",  required=True,help="image width")
ap.add_argument("--height",  required=True,help="image height")

args = vars(ap.parse_args())

width=int(args["width"])
height=int(args["height"])



tmpLabels=dict()
labels = pickle.loads(open(args["labelPKL"], "rb").read())
# build the label to be key is th outout index and value is the class
for key,value in labels.items():   #{'cats': 0, 'dogs': 1}
	tmpLabels[value]= key

labels=tmpLabels
print(labels)

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (width,height))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network. from {}".format(args["model"]))
#model = tf.keras.models.load_model(args["model"])
model = tf.keras.models.load_model(args["model"])


print("[INFO] Model loaded succesfully from {}".format(args["model"]))

# classify the input image


probs= model.predict(image) #probabilty
print(probs) #[[4.580059e-14 1.000000e+00]]
y_pred=probs.argmax(axis=1)
y_pred=y_pred[0]
print(y_pred)
prob=probs[0][y_pred]

# build the label
label = labels[y_pred]



label = "{}: {:.2f}%".format(label, prob * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)