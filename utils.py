import numpy as np

def extract_feature_cell_from_all(tot,s,traininds_set,testinds_set,feature_cell_all,parFea_featurenum,parFea_usefeature):
    feature_cell = np.zeros((parFea_featurenum, ), dtype=np.object)
    if tot == 1:
        numimages_train = np.size(traininds_set[s])
	print('Number of training images are:',numimages_train)
	for f in range(parFea_featurenum):
            if parFea_usefeature[f] == 1:
                feature_cell[f] = np.zeros(( numimages_train, feature_cell_all[f].shape[1]))
		for ind in range (numimages_train):
                    feature_cell[f][ind,:] = feature_cell_all[f][traininds_set[s][ind, :]]

    elif tot == 2:
        #testinds_set[s] = map(str, testinds_set[s])
        numimages_test = np.size(testinds_set[s])
        print('Number of testing images are:',numimages_test)
        for f in range(parFea_featurenum):
            if parFea_usefeature[f] == 1:
                feature_cell[f] = np.zeros(( numimages_test, feature_cell_all[f].shape[1]))
                #print(feature_cell[f].shape,len(testinds_set[s]))
                for ind in range(numimages_test):
                    testinds_set[s] = testinds_set[s] -1
                    #print(feature_cell_all[f].shape,testinds_set[s][ind, :],feature_cell_all[f][testinds_set[s][ind, :]])
                    feature_cell[f][ind,:] = feature_cell_all[f][testinds_set[s][ind, :]]
                    

    return feature_cell


def conc_feature_cell(parFea_featurenum,parFea_usefeature,features):
    isfirst = 1
    for f in range(parFea_featurenum):
        if parFea_usefeature[f] == 1:
            if isfirst == 1:
                feature = features[f]
                #print(feature.shape)
                isfirst = 0
            else:
                feature2 = features[f]  
                feature = np.hstack((feature,feature2))
    return feature

