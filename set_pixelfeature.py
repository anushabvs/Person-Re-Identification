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

function [ num_element, lf_name, usebase ] = set_pixelfeatures( lf_type )
%  set_pixel_features
%  Set the pixel features as combinations of pre-defined base features. 
%  
%  Input: 
%           lf_type (0--yMthetaRGB, 1--yMthetaLab, 2--yMthetaHSV, 3--yMthetanRnG)
%  Output: 
%           num_element   dimension of pixel features 
%           lf_name       name of pixel features. 
%           usebase       indicator that shows which base pixel features are used

% definition of base pixel feature components ( see get_pixelfeatures.m for details)
'''
import numpy as np
def set_pixelfeatures(lf_type):
	baselfname = np.zeros((6, 1), dtype=np.object)
	baselfdim = np.zeros((6, 1))
	usebase = np.zeros((6, 1))

	baselfname[0] = 'y'
	baselfdim[0] = 1

	baselfname[1] = 'Mtheta'
	baselfdim[1] = 4

	baselfname[2]= 'RGB'
	baselfdim[2] = 3

	baselfname[3] = 'LAB'
	baselfdim[3] = 3

	baselfname[4] = 'HSV'
	baselfdim[4] = 3

	baselfname[5] = 'rg'
	baselfdim[5] = 2

	#set the pixel features as combinations of pre-defined base features. 
	if lf_type == 0:
		lf_name = 'yMthetaRGB';
		usebase[0] = 1 #y
		usebase[1] = 1 #Mtheta
		usebase[2] = 1 #RGB
	if lf_type == 1:
		lf_name = 'yMthetaLab';
		usebase[0] = 1 #y
		usebase[1] = 1 #Mtheta
		usebase[3] = 1 #LAB
	if lf_type == 2:
		lf_name = 'yMthetaHSV';
		usebase[0] = 1 #y
		usebase[1] = 1 #Mtheta
		usebase[4] = 1 #HSV
	if lf_type == 3:
		lf_name = 'yMthetanRnG';
		usebase[0] = 1 #y
		usebase[1] = 1 #Mtheta
		usebase[5] = 1#nRnG
	elif lf_type not in range(4):
		print("lf_type = {} is not defined".lf_type)

	num_element = 0
	for i in range(len(baselfname)):
	    if usebase[i] == 1:
		num_element = num_element + baselfdim[i]

	return num_element,lf_name,usebase

#lf_name = strcat( lf_name );
#end

