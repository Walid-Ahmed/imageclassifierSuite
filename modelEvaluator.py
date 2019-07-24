from sklearn.metrics import classification_report
from keras.preprocessing import image
import keras
import os
from imutils import paths
import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator



BS=32

class   ModelEvaluator:


	def __init__(self, model,mode=1):
		self.model=tf.keras.models.load_model(model)

		self.testFilesFullPathList=[]
		root_dir="datasets"
		datasetDir='cats_and_dogs'
		testDir="test_images_cats_and_dogs"
		self.input_shape=150,150
		base_dir = os.path.join(root_dir,datasetDir)
		self.labels=["cats","dogs"]

		self.labels.sort()
		self.path_test=os.path.join(root_dir,testDir)
		self.totalTest = len(list(paths.list_images(self.path_test)))
		print('[INFO] Total images in test  dataset '+self.path_test+ 'images :', self.totalTest)


		for root, dirs, files in os.walk(self.path_test):
		   for name in files:
		      print(os.path.join(root, name))
		      self.testFilesFullPathList.append(os.path.join(root, name))


	def evaluate3(self):
		print("[INFO] Evaluating  Classiffication Report 3")
		y_pred=[]
		y_true=[]
		input_shape=150,150    #width,height



		#sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False)

		for imgPath in self.testFilesFullPathList:

		  if (".DS_Store") in imgPath:
		    continue
		  #print(imgPath)
		  folderName=(imgPath).split("/")[-2:-1][0]
		  if(self.labels[0] in folderName  ):
		    y_true.append(0)
		  elif (self.labels[1] in folderName ):  
		    y_true.append(1)



		  img=keras.preprocessing.image.load_img(imgPath, target_size=input_shape)
		  
		  x=keras.preprocessing.image.img_to_array(img)
		  x=x/255   #rescale image
		  x=np.expand_dims(x, axis=0)
		  image = np.vstack([x])

		  classes = self.model.predict(image)
		  #print(type(classes))     <class 'numpy.ndarray'>
		  
		  #print(classes)

		  #print(classes[0])
		  
		  if classes[0]>0.5:     #1  is a labels[1]
		    y_pred.append(1)
		  else:
		    y_pred.append(0)
	  



		print(classification_report(y_true, y_pred,target_names=self.labels))

	def evaluate2(self):
		print("[INFO] Evaluating  Classiffication Report 2")
		labels=self.labels

		TP_LABEL1=0   #labels[0]
		FP_LABEL1=0
		TP_LABEL2=0   #labels[1]
		FP_LABEL2=0

		TN_LABEL1=0   #labels[0]
		FN_LABEL1=0
		TN_LABEL2=0   #labels[1]
		FN_LABEL2=0

		for imgPath in self.testFilesFullPathList:
		 


		  if (".DS_Store") in imgPath:
		    continue
		  #print(imgPath)
		  folderName=(imgPath).split("/")[-2:-1][0]
		  img=keras.preprocessing.image.load_img(imgPath, target_size=self.input_shape)
		  
		  x=keras.preprocessing.image.img_to_array(img)
		  x=x/255   #rescale image
		  x=np.expand_dims(x, axis=0)
		  image = np.vstack([x])

		  classes = self.model.predict(image)
		  #print(type(classes))     <class 'numpy.ndarray'>
		  
		  #print(classes)

		  #print(classes[0])
		  
		  if classes[0]>0.5:     #1  is a labels[1]
		    #print(imgPath  + " belongs to {}".format(labels[1]))
		   
		    if(labels[1] in folderName  ):
		      TP_LABEL2=TP_LABEL2+1
		      TN_LABEL1=TN_LABEL1+1
		      #print("True Prediction")

		    else:
		      FP_LABEL2=FP_LABEL2+1  
		      FN_LABEL1=FN_LABEL1+1
		      #print("False Prediction")
		    
		  else:
		    #print(imgPath + " belongs to  {}".format(labels[0]))
		    if(labels[0] in folderName):
		      TP_LABEL1=TP_LABEL1+1
		      TN_LABEL2=TN_LABEL2+1
		      #print("True Prediction")
		    else:
		      FP_LABEL1=FP_LABEL1+1  
		      FN_LABEL2=FN_LABEL2+1
		      #print("False Prediction")

		  '''
		  datasets/test_images_cats_and_dogs/cats/cat.585.jpg belongs to dogs
		  Class cats  TP=40,FP=0,TN=1951,FN=0
		  Class dogs  TP=1951,FP=0,TN=40,FN=0
		  '''

		  #print("Class {}  TP={},FP={},TN={},FN={}".format(labels[0],TP_LABEL1,FP_LABEL1,TN_LABEL1,FN_LABEL1))
		  #print("Class {}  TP={},FP={},TN={},FN={}".format(labels[1],TP_LABEL2,FP_LABEL2,TN_LABEL2,FN_LABEL2))
		  #input("Press any key")




		accuracy_LABEL1=(TP_LABEL1+TN_LABEL1)/(TP_LABEL1+FP_LABEL1+TN_LABEL1+FN_LABEL1)
		precision_LABEL1=TP_LABEL1/(TP_LABEL1+FP_LABEL1)
		recall_LABEL1=TP_LABEL1/(TP_LABEL1+FN_LABEL1)
		accuracy_LABEL2=(TP_LABEL2+TN_LABEL2)/(TP_LABEL2+FP_LABEL2+TN_LABEL2+FN_LABEL2)
		precision_LABEL2=TP_LABEL2/(TP_LABEL2+FP_LABEL2)
		recall_LABEL2=TP_LABEL2/(TP_LABEL2+FN_LABEL2)


		print("Class {}  TP={},FP={},TN={},FN={}".format(labels[0],TP_LABEL1,FP_LABEL1,TN_LABEL1,FN_LABEL1))
		print("Class {}  TP={},FP={},TN={},FN={}".format(labels[1],TP_LABEL2,FP_LABEL2,TN_LABEL2,FN_LABEL2))




		#print("Class {}  Accuracy={:.2f},Precision={:.2f},Recall={:.2f}".format(labels[0],accuracy_LABEL1,precision_LABEL1,recall_LABEL1))





		print("Class {}  Accuracy={:.2f},Precision={:.2f},Recall={:.2f}".format(labels[1],accuracy_LABEL2,precision_LABEL2,recall_LABEL2))

		print("*************************************************************************************************************")

	def evaluate1(self):


		test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
	# initialize the testing generator
		test_generator = test_datagen.flow_from_directory(self.path_test,class_mode="binary",target_size=self.input_shape,batch_size=BS)
			# reset the testing generator and then use our trained model to
		# make predictions on the data
		print("[INFO] Evaluating  Classiffication Report 1")

		#test_generator.reset()
		predIdxs = self.model.predict_generator(test_generator,steps=(self.totalTest // BS) + 1)

		predictedLabels=[]



		for  predIdx in predIdxs:

		  if predIdx>0.5:     #1  is a labels[1]
		      #print(" belongs to {}".format(labels[1]))
		      predictedLabels.append(1)
		      
		  else:
		      #print( " belongs to  {}".format(labels[0]))
		      predictedLabels.append(0)

		print(classification_report(test_generator.classes, predictedLabels,target_names=test_generator.class_indices.keys()))
		print("*************************************************************************************************************")      

if __name__ == '__main__':
	modelFile="/Users/walidahmed/Google Drive/code/imageclassifierSuite/results/cats_dogs_binaryClassifier.keras2"



	modelEvaluator=ModelEvaluator(modelFile)
	modelEvaluator.evaluate1()
	modelEvaluator.evaluate2()
	modelEvaluator.evaluate3()



