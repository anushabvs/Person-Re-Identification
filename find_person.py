import numpy as np
import cv2
import Dataloader as dn
import os,sys
import time
import scipy.io
import argparse
from PIL import Image
from set_pixelfeature import set_pixelfeatures
from GOG_C1_copy import GOG_C1
import Feature_normalization
import KLDA as k
from scipy.spatial import distance
import utils as ut
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Find Person ')

parser.add_argument('--normalization', default = 1,type = int,help=' 1--normalization, 2 -- normalization2, 3 -- normalization_on_GOG')
#parser.set_defaults(database = 0)
def RBF_kernel_test(X,Y):
    a = X**2
    #print('a',a.shape)
    norm1 = np.sum(a,axis =1,keepdims = True)
    norm2 = np.sum(a,axis =1, keepdims = True)
    dist = np.tile(norm1, (1, X.shape[0]))  + np.tile(np.transpose(norm2), (X.shape[0], 1)) - 2*np.matmul(X,np.transpose(X))
    mu = np.sqrt(dist.flatten(1).mean(axis=0)/2)
    K_train = np.exp(-0.5/(mu**2)* dist)
    norm_1 = np.sum(X**2,axis =1,keepdims = True)
    norm_2 = np.sum(Y**2,axis =1, keepdims = True)
    dist_1 = np.tile(norm_1, (1, Y.shape[0])) + np.tile(np.transpose(norm_2), (X.shape[0], 1)) - 2*np.matmul(X,np.transpose(Y))
    K_test = np.exp(-0.5/(mu**2)* dist_1)
    return K_test, mu
 

def RBF_kernel(X, Y, mu):
    norm1 = np.sum(X**2,axis =1,keepdims = True)
    norm2 = np.sum(Y**2,axis =1, keepdims = True)
    dist = np.tile(norm1, (1, Y.shape[0])) + np.tile(np.transpose(norm2), (X.shape[0], 1)) - 2*np.matmul(X,np.transpose(Y))
    K_train = np.exp(-0.5/(mu**2)* dist)
    return K_train

def MahDist(M,Xg,Xp):
    if (Xg == Xp).all() == True:
        print("True")
        D = np.matmul(np.matmul(Xg,M),np.transpose(Xg))
        u = D.diagonal()
        dist = u* np.transpose(u) - 2*D
        
    else:
        
        #print("Not equal")
        u = np.sum((np.matmul(Xg,M)*Xg), axis =1)
        v = np.sum((np.matmul(Xp,M)*Xp), axis =1)
        u = np.expand_dims(u, axis=1)
        v = np.expand_dims(v, axis=1)
        dist = u*v.T- 2*np.dot(np.dot(Xg,M),Xp.T)
        #print("Params",dist[:,p].shape,np.sort(dist[:,p]), np.argsort(dist[:,p]))
    return dist

def interClassStatistics3(projY_b,projY_a):
    interSampleDist = MahDist(np.eye(projY_b.shape[1]), projY_b, projY_a)
    return interSampleDist
    

def main():
    global args
    args = parser.parse_args()
    path = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018/'
    img = cv2.imread(path+'/DatasetIITB/cropped_s4c1_test/228b/228_0.jpg',1)
    
    a,Data,parFea =  dn.dataloader()
    alpha = np.load('alpha_1.npy')
    print('*** load all extracted features ***')
    feature_cell_all = np.zeros((parFea.featurenum, ), dtype=np.object)
    traininds_set = Data['traininds_set'][0] + 1
    trainlabels_set = Data['trainlabels_set'][0]+ 1
    testinds_set = Data['testinds_set'][0] + 1
    testlabels_set = Data['testlabels_set'][0] + 1
    traincamIDs_set = Data['traincamIDs_set'][0]
    testcamIDs_set = Data['testcamIDs_set'][0]
    normalize = 1
    s = 0
    #Training data
    print('Training Data!!!!')
    tot = 1;
    #extract feature cells for particular training/test division
    databasename = 'DatasetIITB'
    for f in range(parFea.featurenum-1):
        if parFea.usefeature[f] == 1:
            print('feature = %d [ %s ] \n', f, parFea.featureConf[f].name)
        
        print('feature_all', parFea.featureConf[f].name);
        name = '/home/user/Desktop/Person_ReID_python/' + databasename + '_'+parFea.featureConf[f].name+'.npy'
        print('%s \n', name)
        temp = np.load(name)
        feature_cell_all[f] =temp

    feature_cell = ut.extract_feature_cell_from_all(tot,s,traininds_set,testinds_set,feature_cell_all,parFea.featurenum,parFea.usefeature)

    d = Feature_normalization.feature_normalize()
    if(args.normalization==1):
        features,meanX = d.apply_normalization(tot,parFea.usefeature,parFea.featurenum,feature_cell) #feature normalization
    elif (args.normalization==2):
        features,meanX = d.apply_normalization_onGoG(tot,parFea.usefeature,parFea.featurenum,feature_cell)
    elif(args.normalization==3):
        features,meanX = d.apply_normalization2(tot,parFea.usefeature,parFea.featurenum,feature_cell)
    else:
        print('Undefined normalization setting \n');
        assert(False)
    
    feature = ut.conc_feature_cell(parFea.featurenum,parFea.usefeature,features)
    #print(feature.shape)
    #print("Final feature", feature.shape)
    #train NFST metric learning
    camIDs = traincamIDs_set[s]
    prob_1 = np.where(camIDs==1)
    p_1 = camIDs[prob_1].shape[0]
    probX = feature[0:p_1, :]  
    galX = feature[p_1:camIDs.shape[0], :]  
    #print("Train",(probX == galX))
    labelsX = trainlabels_set[s]
    probXLabels = labelsX[camIDs == 1]
    galXLabels = labelsX[camIDs == 2]
    print("Completed loading training parameters!!!")
    ###Testing Data####
    tot = 2
    #extract feature cells for particular training/test division
    databasename = 'DatasetIITB'
    #feature_cell_all = np.zeros((parFea.featurenum, ), dtype=np.object)
    feature_cell = ut.extract_feature_cell_from_all(tot,s,traininds_set,testinds_set,feature_cell_all,parFea.featurenum,parFea.usefeature)
    #print("Test_features",feature_cell[0])          
    d = Feature_normalization.feature_normalize()
    #extract_feature_cell_from_all;  % load training data 
    if(args.normalization==1):
        features,meanX = d.apply_normalization(tot,parFea.usefeature,parFea.featurenum,feature_cell) #feature normalization
    elif (args.normalization==2):
        features,meanX = d.apply_normalization_onGoG(tot,parFea.usefeature,parFea.featurenum,feature_cell)
    elif(args.normalization==3):
        features,meanX = d.apply_normalization2(tot,parFea.usefeature,parFea.featurenum,feature_cell)
    else:
        print('Undefined normalization setting \n');
        assert(False)

    feature = ut.conc_feature_cell(parFea.featurenum,parFea.usefeature,features)
    
    #train NFST metric learning
    camIDs = testcamIDs_set[s]
    prob_1 = np.where(camIDs==1)
    p_2 = camIDs[prob_1].shape[0]
    probY = feature[0:p_2, :]  
    galY = feature[p_2:camIDs.shape[0], :]
    labelsY = testlabels_set[s]
    probYLabels = labelsY[camIDs == 1]
    galYLabels = labelsY[camIDs == 2]  
    H = 128
    W = 48
    
    if img.shape != (H,W,3):
        img = cv2.resize(img,(W,H))
     
    feat = GOG_C1(img)
    
    feat = (feat - np.transpose(meanX))/np.linalg.norm(feat, 2)
    print('Query feat extracted')

    minimum = 9999
    minIndex = 0

    for i in range(probY.shape[0]):
       #print("True",i)
       #D  = np.sqrt(np.sum((probY[i, :] - np.transpose(feat))**2))
       D = np.linalg.norm(probY[i, :]-np.transpose(feat))
       #print(D,minimum)
       if D < minimum:
           #print("True",i)
           minimum = D
           minIndex = i
      
    p = minIndex
    identity = p
    X = np.vstack((galX,probX))
    Y_a = probY
    Y_b = galY
    K_a, mu = RBF_kernel_test(X, Y_a) ##def
    K_b = RBF_kernel(X, Y_b, mu) ##def
    projY_a = np.dot(np.transpose(K_a),alpha)
    projY_b = np.dot(np.transpose(K_b),alpha)
    
    interSampleDistTst = interClassStatistics3(projY_b,projY_a)
    #print(interSampleDistTst.shape)
    score = interSampleDistTst[:,p]
    #print((score[:,p]).shape)
    sortscore = np.sort(score)
    ind = np.argsort(score)
    #print(sortscore,ind)
    print('Matching of %d is %d %d %d %d %d\n', p, ind[0], ind[1], ind[2], ind[3], ind[4])
    
    path1 = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018'+'/DatasetIITB/'+'cropped_s4c2_test/'+ str(ind[0]+ 206)+ 'b/'+str(ind[0]+ 206)+ '_0.jpg'
    path2 = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018'+'/DatasetIITB/'+'cropped_s4c2_test/'+ str(ind[1]+ 206)+ 'b/'+str(ind[1]+ 206)+ '_0.jpg'
    path3 = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018'+'/DatasetIITB/'+'cropped_s4c2_test/'+ str(ind[2]+ 206)+ 'b/'+str(ind[2]+ 206)+ '_0.jpg'
    path4 = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018'+'/DatasetIITB/'+'cropped_s4c2_test/'+ str(ind[3]+ 206)+ 'b/'+str(ind[3]+ 206)+ '_0.jpg'
    path5 = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018'+'/DatasetIITB/'+'cropped_s4c2_test/'+ str(ind[4]+ 206)+ 'b/'+str(ind[4]+ 206)+ '_0.jpg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img3 = cv2.imread(path3)
    img4 = cv2.imread(path4)
    img5 = cv2.imread(path5)
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 8))
    
    axes[0].set_title('Original')
    axes[0].imshow(img)
    axes[1].set_title('Match_1')
    axes[1].imshow(img1)
    axes[2].set_title('Match_2')
    axes[2].imshow(img2)
    axes[3].set_title('Match_3')
    axes[3].imshow(img3)
    axes[4].set_title('Match_4')
    axes[4].imshow(img4)
    axes[5].set_title('Match_5')
    axes[5].imshow(img5)
    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    main()
