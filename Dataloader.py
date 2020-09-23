import numpy as np
import argparse
import os,sys
import time
import scipy.io
import cv2
from PIL import Image
from set_pixelfeature import set_pixelfeatures
from GOG_C1_copy import GOG_C1

parser = argparse.ArgumentParser(description='Person ReID')

parser.add_argument('--database', default = 0,type = int,help='Select a number from 0-7 to select a database')
parser.add_argument('--feature_settings', default = 1,type = int,help=' 1-- GOG_RGB, 2 -- GOG_Fusion')
parser.set_defaults(database = 0)

def set_default_parameter(lf_type):
	class param:
        	epsilon0 = 0.001 #reguralization paramter of covariance
                p = 2 #patch extraction interval
                k = 5 #patch size (k x k pixels) 
                ifweight = 1 #patch weight  0 -- not use,  1 -- use.  
                G = 7 #number of horizontal strips
                class lfparam:
                	num_element,lf_name,usebase= set_pixelfeatures(lf_type)
                d = lfparam.num_element #dimension of pixel features
                
		m = (d*d + 3*d )/2 + 1 #dimension of patch Gaussian vector
		dimlocal = (m*m + 3*m)/2 + 1 #dimension of region Gaussian vector
		dimension = int(G*dimlocal) #dimension of feature vector
		name = 'GOG' + lfparam.lf_name
                print(dimension)
	return param


def main():
    global args
    args = parser.parse_args()
    setnum = 1
    datadirname_root = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018/' #indicate path that contains the path to KLDA_iitb
    featuredirname_root = '/home/user/Downloads/KLDA_iitb_ICVGIP_2018-20200912T102424Z-001/KLDA_iitb_ICVGIP_2018/KLDA_iitb_ICVGIP_2018/Features/' # indicate root path to save features
    print("Database selected is:",args.database )

    if args.database == 0:
        print("Entered!!!")
        databasename = 'DatasetIITB';
        datadirname = datadirname_root + databasename + '/'
        featuredirname = featuredirname_root
        DBfile = datadirname_root + '/DB/DatasetIITB.mat'
        print(DBfile)
        numperson_train = 206
        numperson_probe = 81
        numperson_garalley = 81

    if args.database == 1:
        databasename = 'VIPeR';
        datadirname = datadirname_root + databasename + '/'
        featuredirname = featuredirname_root
        DBfile = datadirname_root + '/DB/VIPeR.mat'
        
        numperson_train = 316
        numperson_probe = 316
        numperson_garalley = 316
    if args.database == 2:
        databasename = 'CUHK01'
        datadirname = datadirname_root+ 'CUHK01/campus/'
        featuredirname = featuredirname_root
        DBfile = datadirname_root + '/DB/CUHK01M1.mat'
        
        #person number is same as [25] (Paisitkriangkrai et. al 2015)
        numperson_train = 486
        numperson_probe = 485
        numperson_garalley = 485

    if args.database == 3:
        databasename = 'CUHK01'  # CUHK01(M=2) multishot
        datadirname = datadirname_root+ 'CUHK01/campus/'
        featuredirname = featuredirname_root
        DBfile = './DB/CUHK01M2.mat'
        
        #person number is same as [26] (Liao et. al 2015) 
        numperson_train = 485
        numperson_probe = 486
        numperson_garalley = 486
    if args.database == 4:
        databasename = 'PRID450s'
        datadirname = datadirname_root+ '/prid_450s/'
        featuredirname = featuredirname_root
        DBfile =datadirname_root +'/DB/PRID450s.mat'
        
        numperson_train = 225
        numperson_garalley = 225
        numperson_probe = 225
    if args.database == 5:
        databasename = 'GRID'
        datadirname = datadirname_root+databasename+'/'
        featuredirname = featuredirname_root;
        DBfile = datadirname_root + '/DB/GRID.mat'
          
        numperson_train = 125
        numperson_probe = 125
        numperson_garalley = 900
    if args.database == 6:
        databasename = 'CUHK03labeled'
        datadirname = datadirname_root+'CUHK03/labeled/'
        featuredirname = featuredirname_root
        DBfile = datadirname_root +'/DB/CUHK03labeled.mat'
       
        numperson_train = 1260
        numperson_garalley = 100
        numperson_probe = 100
    if args.database == 7:
        databasename = 'CUHK03detected'
        datadirname = datadirname_root+'CUHK03/detected/'
        featuredirname = featuredirname_root
        DBfile = datadirname_root +'/DB/CUHK03detected.mat'
        
        numperson_train = 1260
        numperson_garalley = 100
        numperson_probe = 100
    elif args.database not in range(8):
        print("Database is not defined!!!")
    
    #print(DBfile)
    Data = scipy.io.loadmat(DBfile)  
    #print("All image names are: ",Data['allimagenames'].shape)
    all_image_names = Data['allimagenames']
    allimagenums = int(Data['allimagenames'].shape[0])
    #image size for resize
    H = 128
    W = 48
    class parFea:
    	featurenum = 4
        featureConf = np.zeros((featurenum, ), dtype=np.object)
        usefeature = np.zeros((featurenum,1))
 
    if args.feature_settings == 1:
        parFea.usefeature[0] = 1 #GOG_RGB
        parFea.usefeature[1] = 0 #GOG_Lab
        parFea.usefeature[2] = 0 #GOG_HSV
        parFea.usefeature[3] = 0 #GOG_nRnG
    elif args.feature_settings == 2:
        parFea.usefeature[0] = 1 #GOG_RGB
        parFea.usefeature[1] = 1 #GOG_Lab
        parFea.usefeature[2] = 1 #GOG_HSV
        parFea.usefeature[3] = 0 #GOG_nRnG
    else:
   	print("Feature Setting is not defined!!!")
    print("Feature setting is ", args.feature_settings)
    for f in range(parFea.featurenum):
    	parFea.featureConf[f] = set_default_parameter(f)
        #print(parFea.featureConf[f].name)
    ##########Feature Extraction########################

    print('*** low level feature extraction *** ')
    print('database = ', databasename)
    #Extracting features for all images
    print('Extract feature for all images \n')

    for f in range(parFea.featurenum):
    	if parFea.usefeature[f] == 1:
        	param = parFea.featureConf[f]
                print(param.name)
                feature_all = np.zeros((allimagenums, param.dimension))
                print(feature_all.shape)
                t_0 = time.clock()
                for imgind in range(allimagenums):
                        print("Processing image ",imgind)
                	if (imgind%100 == 1):
                        	print('imging = ',imgind,allimagenums)
                        path = str(all_image_names[imgind])
                        path_final = os.path.join(datadirname,path[10:(len(path)-18)])
                        print(path_final)
                        #print(path[9:(len(path)-17)])
                        X = cv2.imread(path_final,1)
                        if X.shape != (H,W,3):
                                #print("Resizing")
                        	X = cv2.resize(X,(W,H))
                        #print("Final_shape", X.shape,Y.shape)
			feature_all[imgind,:] = GOG_C1(X)
		#print(feature_all)
		t_1 = time.clock()
		mean_time = t_1/allimagenums
		print("Mean feature extraction time is {} seconds per image".format(round(mean_time,3)))
		#print("Feature all", param.name)
		path_name = featuredirname + databasename+'_'+  param.name # change it to csv file or numpy array npy format
                name = databasename+'_'+  param.name
                np.save(name,feature_all)
                np.savetxt(name, feature_all, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
  
if __name__ == '__main__':
    main()
