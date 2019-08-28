from sklearn.metrics import classification_report
from keras.preprocessing import image
import keras
import os
from imutils import paths
import numpy as np
from tensorflow import keras
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pickle


BS=32

class   ModelEvaluator:


	def probToOneOrZero(x,threshold):
		if x>=threshold:
			return 1
		else:
			return 0	


	def __init__(self, model,labels,input_shape,path_test,mode=1):
		self.model=tf.keras.models.load_model(model)

		self.testFilesFullPathList=[]
		self.input_shape=input_shape
		self.labels=labels

		self.labels.sort()
		self.path_test=path_test
		self.totalTest = len(list(paths.list_images(self.path_test)))
		print('[INFO] Total images in test  dataset '+self.path_test+ 'images :', self.totalTest)


		for root, dirs, files in os.walk(self.path_test):
		   for name in files:
		      print(os.path.join(root, name))
		      self.testFilesFullPathList.append(os.path.join(root, name))


	def PrecisionRecall(self):
		print("[INFO] Evaluating  Precision-Recall curve")
		y_pred=[]
		y_true=[]
		probs=[]



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



		  img=keras.preprocessing.image.load_img(imgPath, target_size=self.input_shape)
		  
		  x=keras.preprocessing.image.img_to_array(img)
		  x=x/255   #rescale image
		  x=np.expand_dims(x, axis=0)
		  image = np.vstack([x])

		  classes = self.model.predict(image)
		  probs.append(classes[0])
		  #print(type(classes))     <class 'numpy.ndarray'>
		  
		  #print(classes)

		  #print(classes[0])
		  
		  if classes[0]>0.5:     #1  is a labels[1]
		    y_pred.append(1)
		  else:
		    y_pred.append(0)


		precision, recall, thresholds = precision_recall_curve(y_true, probs) #y_score    probabilities between 0 and 1

		#print(precision)
		#print(recall)



		# calculate 
		F1 = f1_score(y_true, y_pred)  # vcalculated for a certian threshold used to calculate  y_pred
		print("[INFO] F1 score at threshold 0.5=" +str(F1) )

	
		F1=[]
		for i in range(len(thresholds)):
			F1.append( 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
			#print("Perciosion = {0} and Recall={1} at Threshold={2} and F1-Score={3}".format(precision[i], recall[i],thresholds[i],F1[i]) )
		F1_Max=max(F1)
		index=F1.index(F1_Max)
		print("The Highest F1_Score is {0}  with Precision {1} and recall {2} at Threshold {3}".format(F1_Max,precision[index] , recall[index],thresholds[index]))
			

		average_precision = average_precision_score(y_true, probs)


		#predictions = [numpy.round(x) for x in predictions]    #0 or 1 

		precision_value=precision_score(y_true, y_pred, average='macro')  

		print(precision_value)
		print("[INFO] precision_value at threshold 0.5=" +str(precision_value) )


		plt.step(recall, precision, color='b', alpha=0.2, where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Class Precision-Recall curve for class {0}'.format(self.labels[1] +" vs " +  self.labels[0]))
		fileName="Precision_Recall_curve_"+self.labels[1]+".png"
		plt.savefig(fileName)
		plt.show()

		          

		plt.plot(thresholds, F1, 'ro')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.05])
		plt.xlabel('Threshold')
		plt.ylabel('F1-Score')
		plt.title('F1-Score Vs Threshold for  class {0}'.format(self.labels[1] +" vs " +  self.labels[0]))
		fileName="F1_Vs Threshold_"+self.labels[1] +" vs " +  self.labels[0]+".png"
		plt.savefig(fileName)
		plt.show()



 	

	def evaluate3(self):
		print("[INFO] Evaluating  Classiffication Report 3")
		y_pred=[]
		y_true=[]



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
		  folderName=(imgPath).split(os.sep)[-2:-1][0]
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




		print("Class {}  Accuracy={:.2f},Precision={:.2f},Recall={:.2f}".format(labels[0],accuracy_LABEL1,precision_LABEL1,recall_LABEL1))





		print("Class {}  Accuracy={:.2f},Precision={:.2f},Recall={:.2f}".format(labels[1],accuracy_LABEL2,precision_LABEL2,recall_LABEL2))

		print("*************************************************************************************************************")

	def evaluate1(self):


		test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
	# initialize the testing generator
		test_generator = test_datagen.flow_from_directory(self.path_test,
			class_mode="binary",
			target_size=self.input_shape,
			color_mode="rgb",
			shuffle = False,
			batch_size=1)

			# reset the testing generator and then use our trained model to
		# make predictions on the data
		print("[INFO] Evaluating  Classiffication Report 1")

		test_generator.reset()
		predIdxs = self.model.predict_generator(test_generator,steps=self.totalTest) 

		predictedLabels=[]



		for  predIdx in predIdxs:


		  if predIdx[0]>0.5:     #1  is a labels[1]
		      #print(" belongs to {}".format(labels[1]))
		      predictedLabels.append(1)
		      
		  else:
		      #print( " belongs to  {}".format(labels[0]))
		      predictedLabels.append(0)

		print(classification_report(test_generator.classes, predictedLabels,target_names=test_generator.class_indices.keys()))
		print("*************************************************************************************************************")      




	def ROC_Calculate(self):
		print("[INFO] Evaluating  ROC")
		y_pred=[]
		y_true=[]
		probs=[]



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



		  img=keras.preprocessing.image.load_img(imgPath, target_size=self.input_shape)
		  
		  x=keras.preprocessing.image.img_to_array(img)
		  x=x/255   #rescale image
		  x=np.expand_dims(x, axis=0)
		  image = np.vstack([x])

		  classes = self.model.predict(image)
		  probs.append(classes[0])
		  #print(type(classes))     <class 'numpy.ndarray'>
		  
		  #print(classes)

		  #print(classes[0])
		  
		  if classes[0]>0.5:     #1  is a labels[1]
		    y_pred.append(1)
		  else:
		    y_pred.append(0)
		# calculate roc curve
		fpr, tpr, thresholds = roc_curve(y_true, probs)
		# calculate AUC
		roc_auc = roc_auc_score(y_true, probs)
		print('AUC: %.3f' % roc_auc) 
		#plot_roc_curve(y_true, probs)


		plt.title('Receiver Operating Characteristic')
		plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
		plt.legend(loc='lower right')
		plt.plot([0,1],[0,1],'r--')
		plt.xlim([-0.1,1.2])
		plt.ylim([-0.1,1.2])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()


if __name__ == '__main__':

	modelFile="results/cats_dogs_binaryClassifier.keras2"
	labels=["cats","dogs"]
	root_dir="TestImages"
	testDir="test_images_cats_and_dogs"
	input_shape=150,150    #width,height


	path_test=os.path.join(root_dir,testDir)


	modelEvaluator=ModelEvaluator(modelFile,labels,input_shape,path_test)
	modelEvaluator.PrecisionRecall()

	modelEvaluator.ROC_Calculate()

	modelEvaluator.evaluate1()
	modelEvaluator.evaluate2()
	modelEvaluator.evaluate3()



