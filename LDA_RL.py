'''
Baseline for transfer learning
Using CSP+LDA

'''

import sys
import numpy as np
import scipy as sp
from scipy import signal

import sklearn.discriminant_analysis as sklda
from sklearn.model_selection import KFold

import scipy.io as sio
import bcitools

from random import shuffle
import math


# ----------------------------------------------------------------------------------------------------------------------
# Load all data from matlab format matrix
def get_data_from_mat(data_W):
    data_Gr = list()
    data_Br = list()
    data_Gl = list()
    data_Bl = list()
    for il in range(len(data_W)): #each frequency band
        data = data_W[il]
        data_con = data['prepData']

        data_Gr.append(data_con[0, 0,]['G_r'])
        data_Gl.append(data_con[0, 0,]['G_l'])
        data_Br.append(data_con[0, 0,]['B_r'])
        data_Bl.append(data_con[0, 0,]['B_l'])

    data_Gr = np.squeeze(np.asarray(data_Gr)) #(11,341)
    data_Gl = np.squeeze(np.asarray(data_Gl))
    data_Bl = np.squeeze(np.asarray(data_Bl))
    data_Br = np.squeeze(np.asarray(data_Br))
    #print(data_Gr.shape)

    # balance classes
    [data_Gr, data_Gl] = bcitools.balance_classes(data_Gr, data_Gl)
    [data_Br, data_Bl] = bcitools.balance_classes(data_Br, data_Bl)
    [data_Gr, data_Bl] = bcitools.balance_classes(data_Gr, data_Bl)
    [data_Br, data_Gl] = bcitools.balance_classes(data_Br, data_Gl)

    # print(np.shape(data_Gl))

    data_Gr = data_Gr.transpose() #(189,11)
    data_Br = data_Br.transpose()
    data_Gl = data_Gl.transpose()
    data_Bl = data_Bl.transpose()
    data_Right = np.asarray(np.concatenate((data_Gr, data_Br), axis=0))
    data_Left = np.asarray(np.concatenate((data_Gl, data_Bl), axis=0))
    return [data_Right,data_Left]

# ----------------------------------------------------------------------------------------------------------------------
# Load the dataset for each subject and each frequency band
SUBS_NAM = ['S_BP-130516-1','S_BP-220416-2','S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2']
freqsRange = np.array(
    [[1, 3], [2, 5], [4, 7],[6, 10], [7, 12], [10, 15], [12, 19], [18, 25], [19, 30], [25, 35], [30, 40]])

# store data into whole matrix
# data: (subjectnum,Left or Right,experiment_time,trial,time)
data_all = list()
for sub_idx in range(len(SUBS_NAM)):
    data_tmp = list()
    for freqs_idx in range(len(freqsRange)):
        namm = '/media/gin/hacker/UCSD_Summer_Research/data/output/' + SUBS_NAM[sub_idx] + \
               'freqs' + str(freqsRange[freqs_idx][0]) + '_' + str(freqsRange[freqs_idx][1]) + '_shams.mat'
        
        data_tmp.append(sio.loadmat(namm))
    tmp = np.asarray(get_data_from_mat(data_tmp))
    
    data_all.append(tmp)


num_fold = len(data_all)
numF = 3
kf = KFold(n_splits=num_fold)
classRate = np.zeros(num_fold)
count = 0;
#data_all = np.asarray(data_all)
for train_idx, test_idx in kf.split(np.ones(num_fold)):
    #print(train_idx,test_idx)
    #print(len(data_all[train_idx]))
    data_train = [data_all[i] for i in train_idx]
    data_test = [data_all[i] for i in test_idx]
    data_train = np.concatenate(data_train,axis = 1)
    data_test = np.concatenate(data_test,axis = 1)
    print(data_train.shape)
    print(data_test.shape)
    tempTrain = np.shape(data_train) #(314,11)
    train_data_Right = data_train[0]
    train_data_Left = data_train[1]
    test_data_Right = data_test[0]
    test_data_Left = data_test[1]

    train_logP_L = np.zeros([tempTrain[1], 2*numF, 11])
    train_logP_R = np.zeros([tempTrain[1], 2*numF, 11])

    tempTest = np.shape(data_test)

    test_logP_L = np.zeros([tempTest[1], 2*numF, 11])
    test_logP_R = np.zeros([tempTest[1], 2*numF, 11])
    #print(train_data_Right[:,0][0].shape)
    
    for il in range(len(freqsRange)):
        v, a = bcitools.calculate_csp(train_data_Right[:, il], train_data_Left[:, il])# train_data_Right[:, il], train_data_Left[:, il])

        # extract the features
        train_filt_R = bcitools.apply_csp(train_data_Right[:, il],v, numF)
        train_logP_R[:, :, il] = np.squeeze(bcitools.log_power(train_filt_R))
        train_filt_L = bcitools.apply_csp(train_data_Left[:, il], v,numF)
        train_logP_L[:, :, il] = np.squeeze(bcitools.log_power(train_filt_L))

        # test the classifier on the test data
        test_filt_R = bcitools.apply_csp(test_data_Right[:, il], v, numF)
        test_logP_R[:, :, il] = np.squeeze(bcitools.log_power(test_filt_R))
        test_filt_L = bcitools.apply_csp(test_data_Left[:, il], v, numF)
        test_logP_L[:, :, il] = np.squeeze(bcitools.log_power(test_filt_L))

    train_logP_L = np.reshape(train_logP_L, (tempTrain[1], 22*numF))
    train_logP_R = np.reshape(train_logP_R, (tempTrain[1], 22*numF))

    test_logP_L = np.reshape(test_logP_L, (tempTest[1], 22*numF))
    test_logP_R = np.reshape(test_logP_R, (tempTest[1], 22*numF))

    # ---------------------------------------------------------------------------------------------------------------
    # train an LDA on RL
    temp = np.shape(train_logP_L)

    clf_RL = sklda.LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    y1 = np.ones(temp[0])
    y2 = 2 * np.ones(temp[0])
    y = np.concatenate((y1, y2))

    X = np.concatenate((train_logP_R, train_logP_L), axis=0)

    clf_RL.fit(X, y)


    temp = np.shape(test_logP_R)
    vect1 = clf_RL.predict(test_logP_R)
    vect2 = clf_RL.predict(test_logP_L)


    tempp = np.sum(vect1 == 1) + np.sum(vect2 == 2)

    # since classes are balanced, calculate the overall accuracy
    classRate[count] = tempp / float(2 * temp[0])

    print(classRate[count])
    count = count + 1 
    

