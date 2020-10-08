###Feature Normalization###
import numpy as np
import os,sys

class feature_normalize:

	def apply_normalization(self,tot,parFea_usefeature,parFea_featurenum,feature_cell):
		#if tot == 1:
			#mean_cell = np.zeros((parFea_featurenum, ), dtype=np.object)
                mean_cell = np.zeros((parFea_featurenum, ), dtype=np.object)


		for f in range(parFea_featurenum):
			if parFea_usefeature[f] == 1:
				X = feature_cell[f] #X: feature vectors of feature type f. Size: [datanum, feature dimension]
				#print(X.shape)
			
			if tot == 1: #training data
                            print("Got in training normalization!!!")
			    meanX = (X.mean(axis=0)).reshape(1,X.shape[1]) #meanX -- mean vector of features
			    mean_cell[f]= meanX
                            #print(X.shape,meanX)

			if tot == 2: #test data
			    meanX = mean_cell[f]
                        #print('mean x',meanX.shape)
			#tmp = F2local - np.tile(np.transpose(meanVec),(F2local.shape[0], 1))
                        #Y = ( X - repmat(meanX, size(X,1), 1))
			Y = X - np.tile(meanX,(X.shape[0],1))#Mean removal
			#print(meanX.shape,feature_cell[f].shape,Y.shape)
			for dnum in range(X.shape[0]):
			    Y[dnum,:] = Y[dnum, :]/np.linalg.norm(Y[dnum, :], 2) #L2 norm normalization

			   
			feature_cell[f] = Y
			
		return feature_cell,meanX
	
	def apply_normalization2(self,tot,parFea_featurenum,feature_cell):
                mean_cell = np.zeros((parFea_featurenum, ), dtype=np.object)
		#if tot == 1:
			#mean_cell = np.zeros((parFea_featurenum, ), dtype=np.object)
		for f in range(parFea_featurenum):
			if parFea_usefeature[f] == 1 and f<=4 and f>=1:
				X = feature_cell[f]
			if tot == 1: #training data
			    meanX = (X.mean(axis=0)).reshape(1,X.shape[1]) #meanX -- mean vector of features
			    mean_cell[f]= meanX

			if tot == 2: #test data
			    meanX = mean_cell[f]
			#tmp = F2local - np.tile(np.transpose(meanVec),(F2local.shape[0], 1))
			Y = X - np.tile(meanX,(X.shape[0],1))#Mean removal
			for dnum in range(X.shape[0]):
				Y[dnum,:] = Y[dnum, :]/np.linalg.norm(Y[dnum, :], 2) #L2 norm normalization

			feature_cell[f] = Y
			
			if parFea_usefeature[f] == 1:
				Y = feature_cell[f] #X: feature vectors of feature type f. Size: [datanum, feature dimension]
				for dnum in range(Y.shape[0]):
					Y[dnum,] = Y[dnum, :]/np.linalg.norm(Y[dnum, :], 2) #L2 norm normalization
			feature_cell[f] = Y
                        
		return feature_cell,meanX

	def apply_normalization_onGoG(self,tot,parFea_featurenum,feature_cell):
		#if tot == 1:
		#	mean_cell = np.zeros((parFea_featurenum, ), dtype=np.object)
                mean_cell = np.zeros((parFea_featurenum, ), dtype=np.object)
		for f in range(parFea_featurenum):
			if parFea_usefeature[f] == 1 and f<=4 and f>=1:
				X = feature_cell[f]
			if tot == 1: #training data
			    meanX = (X.mean(axis=0)).reshape(1,X.shape[1]) #meanX -- mean vector of features
			    mean_cell[f]= meanX

			if tot == 2: #test data
			    meanX = mean_cell[f]
			#tmp = F2local - np.tile(np.transpose(meanVec),(F2local.shape[0], 1))
			Y = X - np.tile(meanX,(X.shape[0],1))#Mean removal
			for dnum in range(X.shape[0]):
			    Y[dnum,:] = Y[dnum, :]/np.linalg.norm(Y[dnum, :], 2) #L2 norm normalization
			feature_cell[f] = Y

		return feature_cell,meanX

