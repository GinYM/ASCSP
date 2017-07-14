'''
Baseline for transfer learning
Using CSP+LDA
'''


import bcitools

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
# Load all data and create matrix (subjectsnum,freq,GBLR)
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


def rl_combined(data_W, num_fold, numF):
    # figure out what is the minimum of all to balance classes
    #data = data_W[0]
    #data_con = data['prepData']
    #num = np.min([len(data_con[0, 0,]['G_r'])])
    #print(num)
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

    #print(np.shape(data_Gl))
    #print(data_Gr[0,0][0,0]) (59,326)


    # make the indices for KFold cross validation
    kf = KFold(n_splits=num_fold, shuffle=True)

    # this is to save the classification rate for each fold
    classRate = np.zeros(num_fold)
    count = 0

    M = np.shape(data_Gr)
    print(M[0])

    for train_idx, test_idx in kf.split(np.ones(M[0])):

        train_data_GR = data_Gr[train_idx, :]
        train_data_GL = data_Gl[train_idx, :]
        train_data_BR = data_Br[train_idx, :]
        train_data_BL = data_Bl[train_idx, :]

        # make the left and right classes - training set and validation sets
        train_data_Right = np.asarray(np.concatenate((train_data_GR, train_data_BR), axis=0))
        train_data_Left = np.asarray(np.concatenate((train_data_GL, train_data_BL), axis=0))


        test_data_GR = data_Gr[test_idx, :]
        test_data_GL = data_Gl[test_idx, :]
        test_data_BR = data_Br[test_idx, :]
        test_data_BL = data_Bl[test_idx, :]

        # make the left and right for test
        test_data_Right = np.asarray(np.concatenate((test_data_GR, test_data_BR), axis=0))
        test_data_Left = np.asarray(np.concatenate((test_data_GL, test_data_BL), axis=0))

        # find the CSP filters for the training set in each frequency band separately
        tempTrain = np.shape(train_data_Right) #(314,11)


        train_logP_L = np.zeros([tempTrain[0], 2*numF, 11])
        train_logP_R = np.zeros([tempTrain[0], 2*numF, 11])

        tempTest = np.shape(test_data_Right)

        test_logP_L = np.zeros([tempTest[0], 2*numF, 11])
        test_logP_R = np.zeros([tempTest[0], 2*numF, 11])

        for il in range(len(freqsRange)):
            v, a = bcitools.calculate_csp(train_data_Right[:, il], train_data_Left[:, il])

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


        train_logP_L = np.reshape(train_logP_L, (tempTrain[0], 22*numF))
        train_logP_R = np.reshape(train_logP_R, (tempTrain[0], 22*numF))

        test_logP_L = np.reshape(test_logP_L, (tempTest[0], 22*numF))
        test_logP_R = np.reshape(test_logP_R, (tempTest[0], 22*numF))

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
    return classRate


# ----------------------------------------------------------------------------------------------------------------------
# Load the dataset for each subject and each frequency band
#SUBS_NAM = ['S_BP-220416-2', 'S_BP-240416-1', 'S_BP-240416-2', 'S_BP-240416-3', 'S_BP-270416-2', 'S_BP-130516-1','S_BP-141216', 'S_BP-191216', 'S_BP-010117', 'S_BP-011217']
SUBS_NAM = ['S_BP-130516-1','S_BP-220416-2','S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2']


freqsRange = np.array(
    [[1, 3], [2, 5], [4, 7],[6, 10], [7, 12], [10, 15], [12, 19], [18, 25], [19, 30], [25, 35], [30, 40]])


data_all = list()
for sub_idx in range(len(SUBS_NAM)):
    data_tmp = list()
    for freqs_idx in range(len(freqsRange)):
        namm = '/media/gin/hacker/UCSD_Summer_Research/data/' + SUBS_NAM[iii] + \
               'freqs' + str(freqsRange[kkk][0]) + '_' + str(freqsRange[kkk][1]) + '_shams.mat'
        
        data_tmp.append(sio.loadmat(namm))
    data_all.append(get_data_from_mat(data_tmp))


# make a text file to write in the results

num_CV = 3
#num_fold = 5
numF = 3
num_fold = 6

#ccacspWin = 10 # this is the number fo samples to be shifted in ccacsp



for iii in range(len(SUBS_NAM)):
    print(iii)

    namSave = 'results/'+SUBS_NAM[iii] + 'RL.mat'
    exonCell = np.zeros((num_CV,), dtype=np.object)

    data = list()
    for kkk in range(len(freqsRange)):  # number of frequencies
        print(kkk)

        #namm = '/home/mmousavi/Matlab/sham_feedback_paradigm/prepd-data-U/' + SUBS_NAM[iii] + \
        #       'freqs' + str(freqsRange[kkk][0]) + '_' + str(freqsRange[kkk][1]) + '_shams_FP.mat'
        namm = '/media/gin/hacker/UCSD_Summer_Research/data/' + SUBS_NAM[iii] + \
               'freqs' + str(freqsRange[kkk][0]) + '_' + str(freqsRange[kkk][1]) + '_shams.mat'
        
        data.append(sio.loadmat(namm))

    for ijj in range(num_CV):
        classRate = rl_combined(data, num_fold, numF)
        print(classRate)
        exonCell[ijj] = classRate

    sio.savemat(namSave, mdict={'rate': exonCell})


