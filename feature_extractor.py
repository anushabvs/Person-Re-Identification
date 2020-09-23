'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This file is part of the Code Repository for Person Re-Id on IIT-B Dataset
%% Contributors to the Code :-
%% T.M Feroz Ali, Kalpesh Patel, Saurabh Chavan, Royston Rodrigues

%% Please Cite this work if you find it useful:
%% Multiple Kernel Fisher Discriminant Metric Learning for Person Re-identification
%% T M Feroz Ali, Kalpesh K Patel, Rajbabu Velmurugan, Subhasis Chaudhuri
%% ACM Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP), Hyderabad, India, December 2018.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     function [ feature_vec] = GOG_C1( X )
% Gaussian of Gaussian (GOG) descriptor
% 
% Input: 
%  <X>: input RGB image. Size: [h, w, 3]
%%%%%%%%%%%%%  param %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% feature_vec = zeros(7567,1);
'''
import numpy as np
import math
import scipy.ndimage
import cv2
#from rgb2lab_C import rgb2lab
from scipy.linalg import logm, expm

def halfvec(a11):
    m,n = len(a11), len(a11[0])
    l = (m*(m+1))/2
    logPcells = np.zeros((l,1))
    k=0
    for i in range(m):
        for j in range(i,n):
            logPcells[k,0] = a11[i,j]
            if i!=j:
                logPcells[k,0]=math.sqrt(2)*logPcells[k,0]
            k+=1
    return logPcells


def vec2mat(vec,matcol):
    mat = np.zeros((matcol,matcol))
    vec = np.reshape(vec,[vec.shape[0],1])
    n = matcol
    k=0
    j=0
    while j!= len(vec):
        for i in range(k,n): 
            mat[i,k]=vec[j,0]
            mat[k,i]=vec[j,0]
            j=j+1
        k=k+1
    return mat

def create_corr_MT(F):
    m,n,l = F.shape
    ch = (l*(l+1))/2
    b = np.zeros((m,n,ch))
    k = 0
    for i in range(l):
        for j in range (i,l):
            b[:,:,k]=F[:,:,j]*F[:,:,i]
	    k +=1
    return b

def create_IH_MT(F):
    h,w,c = F.shape
    IH = np.zeros((h+1,w+1,c))
    for i in range(h+1):
        for j in range(w+1):
            if j==0 or i ==0:
                IH[i,j,:] = 0
            else:
                IH[i,j,:]=IH[i-1,j,:]+IH[i,j-1,:]-IH[i-1,j-1,:]+F[i-1,j-1,:]
    return IH

def get_gradmap(X,binnum):
    hx = np.zeros((3,1))
    hx[0] = 1
    hx[2] = -1
    hy = -hx.T
    grad_x = scipy.ndimage.correlate(X, hx, mode='constant') ##See the outputs if it is working or change it
    grad_y =  scipy.ndimage.correlate(X, hy, mode='constant')
    tan_2 = np.arctan2(grad_y,grad_x)
    ori = (tan_2 + np.pi)*(180/np.pi) #gradient orientations
    mag = np.sqrt(grad_x**2 + grad_y**2 ) #gradient magnitude
    binwidth = 360/binnum
    IND = np.array(np.floor(ori/binwidth))
    ref1 = IND*binwidth
    ref2 = (IND + 1)*binwidth
    dif1 = ori - ref1
    dif2 = ref2 - ori
    weight1 = dif2/(dif1 + dif2)
    weight2 = dif1/(dif1 + dif2)
    h,w = X.shape
    qori = np.zeros((h, w, binnum)) 
    IND = np.where(IND == binnum, 0, IND)  
    IND1 = IND + 1
    IND2 = IND + 2
    IND2 = np.where(IND2 == binnum+1, 1, IND2)
    for y in range(h):
        for x in range(w):
            qori[y,x,int(IND1[y,x])-1] = weight1[y,x]
            qori[y,x,int(IND2[y,x])-1] = weight2[y,x]
    return (qori,ori,mag)
 



###########################################################################################
def get_pixel_features(X,param):
	(h,w,_) = X.shape
        F = np.zeros((h,w,param.d))
        dimstart = 0  
	if param.selection[0] == 1: #y
        	PY = np.zeros((X.shape[0], X.shape[1]))
		for tmpy in range(X.shape[0]):
			PY[tmpy,:] = tmpy
		PY = np.divide(PY,X.shape[0])
		F[:,:,dimstart] = PY
		dimstart+=1
	
	if param.selection[1] == 1:#M_theta
		print("True,section 1 = 1")
        	binhog = 4
                img_ycbcr = cv2.cvtColor(X, cv2.COLOR_BGR2YCR_CB)
		img = np.double(img_ycbcr[:,:,0]) - 16
                img_converted = np.divide(img,235)
                ####Gradmap###
                (qori,ori,mag) = get_gradmap(img_converted ,binhog)
                mag = np.reshape(mag,[mag.shape[0],mag.shape[1],1])
                g = np.repeat(mag[:, :,], binhog, axis=2)
                Yq = np.multiply(qori,g)
                F[:,:, dimstart:(dimstart + binhog)] = Yq
                dimstart+=binhog
	
	if param.selection[2] == 1:#RGB
		F[:,:,dimstart:dimstart+3] = X/255
                dimstart +=3
	'''
	if param.selection[3] == 1:#LAB this was commented in matlab
        	img_lab = rgb2lab(X)
		img_lab[:,:,0] = img_lab[:,:,0]/100
		img_lab[:,:,1] = (img_lab[:,:,1] + 50)/100
		img_lab[:,:,2] = (img_lab[:,:,2] + 50)/100
		F[:,:, dimstart: dimstart+1] = img_lab
		dimstart = dimstart + 3
	'''
        if param.selection[4] == 1: #HSV
		F[:,:,dimstart:dimstart+2] = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
		dimstart = dimstart + 3
	if param.selection[5] == 1:  #nRnG
		sumVal = np.maximum(X.sum(axis=2), 1)
	    	F[:,:,dimstart] = X[:, :, 0]/sumVal
	    	F[:,:,dimstart + 1] = X[:, :, 1]/sumVal
	    	dimstart = dimstart + 2
	return F

def GOG_C1(X):
    s = [1,1,1,0,0,0]
    d = s[0]+(4*s[1])+(3*s[2])+(3*s[3])+(3*s[4])+(2*s[5])
    m = ((d**2)+(3*d)+2)/2
    G = 7
    dimlocal = ((m**2)+(3*m)+2)/2
    class param:
            selection = [1,1,1,0,0,0]  #Select Feature\n  [Y HOG RGB LAB HSV nRnG]
            epsilon0 = 0.001 #reguralization paramter of covariance
            p = 2 #patch extraction interval
            k = 5 #patch size (k x k pixels) 
            ifweight = 1 #patch weight  0 -- not use,  1 -- use.  
            G = 7 #number of horizontal strips
            d = s[0]+(4*s[1])+(3*s[2])+(3*s[3])+(3*s[4])+(2*s[5])
            m = ((d**2)+(3*d)+2)/2
            dimlocal = ((m**2)+(3*m)+2)/2
            dimension = dimlocal*G
    if param.G == 7:
    # 7X1 horizontal strips
	    print("True!!")
            class parGrid:
                    gheight = X.shape[0]/4
                    gwidth = X.shape[1]
                    ystep = gheight/2
                    xstep = X.shape[1]
    else:
            print('This region configuration is not defined !!!\n')
    #####Pixel Feature Extraction#######
    print('Started pixel feature extraction!!!\n')
    F = get_pixel_features(X, param) 
    FCorr = create_corr_MT(F)
    IH_Corr = create_IH_MT(FCorr)
    IH_F = create_IH_MT(F)
    halffsize = (param.k - 1)/2
    a_in1 = np.arange(1,F.shape[1],param.p)
    a_in2 = np.arange(1,F.shape[0],param.p)
    [cols,rows] = np.meshgrid(a_in1,a_in2) 
    cols = cols.flatten()
    rows = rows.flatten()
    cols = np.reshape(cols,[cols.shape[0],1])
    rows = np.reshape(rows,[rows.shape[0],1])
    xlefts = np.maximum((cols + 1 - halffsize), 2)
    xrights =np.minimum((cols + 1 + halffsize),IH_Corr.shape[1])
    yups = np.maximum((rows + 1 - halffsize), 2)
    ydowns =np.minimum((rows + 1  + halffsize), IH_Corr.shape[0])
    IH_Corr2 = np.reshape(IH_Corr, [IH_Corr.shape[0]*IH_Corr.shape[1], IH_Corr.shape[2]])
    IH_F2 = np.reshape(IH_F, [IH_F.shape[0]*IH_F.shape[1], IH_F.shape[2]])
    points1 = (xrights -1)*IH_Corr.shape[0] + ydowns-1
    points2 = (xlefts-2)*IH_Corr.shape[0] + yups-2
    points3 = (xrights-1)*IH_Corr.shape[0] + yups-2
    points4 = (xlefts-2)*IH_Corr.shape[0] + ydowns-1
    points1 = np.reshape(points1,[points1.shape[0],1])
    points1 = np.reshape(points2,[points2.shape[0],1])
    points1 = np.reshape(points3,[points3.shape[0],1])
    points1 = np.reshape(points4,[points4.shape[0],1])
    sumFCorrs = IH_Corr2[points1, :] + IH_Corr2[points2, :] - IH_Corr2[points3, :] - IH_Corr2[points4, :]  #get values from IH
    sumFs = IH_F2[points1, :] + IH_F2[points2, :] - IH_F2[points3, :] - IH_F2[points4, :] #get values from IH
    sumpixels = (xrights - xlefts + 1)*(ydowns -yups + 1)
    logPcells = np.zeros((np.size(cols), 45))
    for i in range(np.size(cols)):
        sumFCorr = np.transpose(sumFCorrs[i,:])
        sumF = np.transpose(sumFs[i,:])
        sumpixel = sumpixels[i]
        S_mat = vec2mat( sumFCorr, param.d ) ###vec2mat def
        S = ( S_mat - np.matmul(sumF,np.transpose(sumF))/sumpixel)/(sumpixel -1 ) #covariance matrix by integral image 
        S = S + param.epsilon0*np.maximum(S.trace(offset=0), 0.01)*np.eye(S.shape[0]) #regularizaiton 
        meanVec = sumF/sumpixel #mean vector 
	meanVec = meanVec.reshape(meanVec.shape[0],1)
        some_val = np.linalg.det(S)**((-1/(S.shape[0] + 1)))
        a_1 = some_val *(S+np.matmul(meanVec,np.transpose(meanVec)))
        arr_1 = np.append(np.transpose(meanVec),1)
        Pcells = np.column_stack((a_1, meanVec))
        Pcells =  np.vstack((Pcells, arr_1))
        a11  = logm(Pcells)
        logPcells[i,:]  = np.transpose(halfvec(a11))###halfvec def
    F2 = np.reshape(logPcells, [F.shape[0]/param.p, F.shape[1]/param.p,logPcells.shape[1]])

    ####################################################################        

    #####Region Gaussians##########
    ###Setup patch weight########## 
    print("Everything above is fine. Entering patch gaussians")
    H0,W0 = F2.shape[0],F2.shape[1]
    if param.ifweight == 0:
        weightmap = np.ones((H0, W0))

    if param.ifweight == 1:
        weightmap = np.zeros((H0, W0))
        sigma = W0/4
        mu = W0/2
        for x in range(W0):
            weightmap[:, x] = np.exp(-((x-mu)*(x-mu)/(2*sigma*sigma)))/(sigma*math.sqrt(2*np.pi))                          
    gheight2 = parGrid.gheight/param.p
    gwidth2 = parGrid.gwidth/param.p
    ystep2 = parGrid.ystep/param.p
    xstep2 = parGrid.xstep/param.p
    in1 = np.arange(0,F2.shape[1]-gwidth2+1,xstep2)
    in2 = np.arange(0,F2.shape[0]-gheight2+1,ystep2)
    [cols, rows] = np.meshgrid(in1, in2)
    cols = cols.flatten()
    rows = rows.flatten()
    cols = np.reshape(cols,[cols.shape[0],1])
    rows = np.reshape(rows,[rows.shape[0],1])
    xlefts = cols
    xrights = cols + gwidth2 -1
    yups = rows
    ydowns = rows + gheight2 - 1
    logPcells1 = np.zeros((np.size(cols), 1081))
    for i in range(np.size(cols)):
        weightlocal = weightmap[yups[i][0]:ydowns[i][0] +1, xlefts[i][0]:xrights[i][0]+1]
        F2localorg =  F2[yups[i][0]:ydowns[i][0] +1 ,xlefts[i][0]:xrights[i][0]+1]    
     
        weightlocal = np.reshape(weightlocal,[weightlocal.shape[0]*weightlocal.shape[1],1])
        F2local = np.reshape( F2localorg,[F2localorg.shape[0]*F2localorg.shape[1], F2localorg.shape[2]])

        wF2local = np.tile(weightlocal,(1,F2local.shape[1]))*F2local
        meanVec = np.transpose(wF2local.sum(axis=0))/weightlocal.sum(axis=0) #mean vector
        tmp = F2local - np.tile(np.transpose(meanVec),(F2local.shape[0], 1))
        tmp1 =np.reshape(tmp, [F2localorg.shape[0],F2localorg.shape[1],tmp.shape[1]])
        FCorr2 = create_corr_MT(tmp1)
        mul_1 = FCorr2.shape[0]*FCorr2.shape[1]
        FCorr2 = np.reshape(FCorr2, [mul_1,FCorr2.shape[2]])
        wFCorr2 = np.tile(weightlocal,(1, FCorr2.shape[1]))*FCorr2
        fcov =  wFCorr2.sum(axis=0)/weightlocal.sum(axis=0)
        S = vec2mat(np.transpose(fcov), F2.shape[2]) #covariance matrix 
        S = S + param.epsilon0*np.maximum(S.trace(offset=0), 0.01)*np.eye(S.shape[0]) #regularizaiton 
        some_val = np.linalg.det(S)**((-1/(S.shape[0] + 1)))
        a_1 = some_val *(S+np.matmul(meanVec,np.transpose(meanVec)))
        arr_1 = np.append(np.transpose(meanVec),1)
        Pcells = np.column_stack((a_1, meanVec))
        Pcells =  np.vstack((Pcells, arr_1))
        a11= logm(Pcells)
        logPcells1[i,:]= np.transpose(halfvec(a11))

    # logPcells = cell2mat_my( cellfun( @(x) halfvec(logm_my(x))', Pcells, 'un', 0) ); % apply log Euclidean and half-vectorization
    feature_vec = np.reshape(logPcells1,[1,(logPcells1.shape[0]*logPcells1.shape[1])]) #concatenate feature vector of each grid
    #feature_vec = GOG_C11(logPcells1); % extract GOG
    return feature_vec	

