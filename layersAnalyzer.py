import tensorflow as tf
import os
import keras





def getKernelSizes(loaded_model):
	convLayersNames=[]
	convLayerDict=dict()
	info=""
	fiieNameToSaveConvAnalysis=os.path.join("Results","convAnalysis.txt")
	f= open(fiieNameToSaveConvAnalysis,"w+")



	for layer in  loaded_model.layers:
		if isinstance(layer, tf.keras.layers.Conv2D)  or  isinstance(layer, keras.layers.Conv2D):
			layerName=layer.name
			convLayersNames.append(layerName)
			layerConfig=layer.get_config()
			print("[INFO] working with layer {} of type {}".format(layerName,type(layer)))
			for k, v in layerConfig.items():
				if (k=="kernel_size"):
					print("[INFO] layer {}  has  {}   equal  {}".format(layerName,k,v))
					info=info+"[INFO] layer {}  has  {}   equal  {}".format(layerName,k,v)+"\n"
					convLayerDict[layerName]=v


	numOfConv2dLayers=len(convLayersNames)
	#print(convLayerDict)

	#for k, v in convLayerDict.items():
		#print(k,v)

    
	print("[INFO] Number of conv layers {}".format(numOfConv2dLayers))
	f.write("***************   Conv Layer Analysis  *********************************"+"\n")
	f.write("[INFO] Number of conv layers {}".format(numOfConv2dLayers))
	f.write(info)
	f.flush()
	f.close()
	print("[INFO] Conv layers  analysis saved to {}".format(fiieNameToSaveConvAnalysis))


if __name__ == "__main__":

	modelFile=os.path.join("Results","dpn","neg_covid_pos_covid_binaryClassifier.h5")	
	print("[INFO] loading model from file  {}..............".format(modelFile))
	loaded_model = tf.keras.models.load_model(modelFile)
	print("[INFO] Model loaded")
	getKernelSizes(loaded_model)