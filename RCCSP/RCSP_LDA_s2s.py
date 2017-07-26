'''
RCSP subject to subject without composite
Test id: 5
main difference: calculate_rccsp_norm

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
from CSP_util import apply_rccsp,calculate_rccsp,positive_transform,calculate_rccsp_norm


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
SUBS_NAM = ['S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216']
#SUBS_NAM = ['S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216']
freqsRange = np.array(
    [[1, 3], [2, 5], [4, 7],[6, 10], [7, 12], [10, 15], [12, 19], [18, 25], [19, 30], [25, 35], [30, 40]])

# store data into whole matrix
# data: (subjectnum,Left or Right,experiment_time,trial,time)
data_all = list()
subj_each_len = np.zeros(len(SUBS_NAM))
for sub_idx in range(len(SUBS_NAM)):
    print('Loading Subj: %s' %(sub_idx))
    data_tmp = list()
    for freqs_idx in range(len(freqsRange)):
        namm = '/media/gin/hacker/UCSD_Summer_Research/data/output_new/' + SUBS_NAM[sub_idx] + \
               'freqs' + str(freqsRange[freqs_idx][0]) + '_' + str(freqsRange[freqs_idx][1]) + '_shams_FP.mat'
        
        data_tmp.append(sio.loadmat(namm))
    tmp = np.asarray(get_data_from_mat(data_tmp))
    subj_each_len[sub_idx] = tmp[0].shape[0]
    data_all.append(tmp)

print(subj_each_len)
subj_each_idx_start = np.zeros(len(subj_each_len))

total_len = 0
for i in range(1,len(subj_each_len)):
    total_len += subj_each_len[i-1]
    subj_each_idx_start[i] = total_len

num_fold = len(data_all)
numF = 3
kf = KFold(n_splits=num_fold)
classRate = np.zeros(num_fold)
count = 0;
#data_all = np.asarray(data_all)
f = open('result/result_LDA_subj_to_subj_RCSP','w')
n_chan = 47

general_matrix = np.load('/media/gin/hacker/UCSD_Summer_Research/data/CSP/CSP_covariance_matrix.npy')

data_total = [data_all[i] for i in range(len(SUBS_NAM))]
data_total = np.concatenate(data_total,axis = 1) #[0] right [1] left
test_idx = int(len(SUBS_NAM)-1)
N_t = 0
C_s = np.zeros([len(SUBS_NAM)-1,11,n_chan,n_chan])
C_t = np.zeros([11,n_chan,n_chan])
C_all = np.zeros([11,n_chan,n_chan])
F_c = np.zeros([len(SUBS_NAM)-1,11])
#C_t = general_matrix[int(len(SUBS_NAM)-1)]


for freq_idx in range(len(freqsRange)):
    C_t[freq_idx] = (general_matrix[int(len(SUBS_NAM)-1)][freq_idx][0]+general_matrix[int(len(SUBS_NAM)-1)][freq_idx][1])/2
    for i in range(len(SUBS_NAM)-1):
        C_s[i][freq_idx] = (general_matrix[i][freq_idx][0]+general_matrix[i][freq_idx][1])/2
        F_c[i][freq_idx] = np.sqrt(np.trace(np.cov(C_s[i][freq_idx]-C_t[freq_idx])))
        C_all[freq_idx] += positive_transform(C_s[i][freq_idx],C_t[freq_idx])/F_c[i][freq_idx]

N_t = np.zeros(11)
for freq_idx in range(len(freqsRange)):
    for i in range(len(SUBS_NAM)-1):
        N_t[freq_idx] += 1/F_c[i][freq_idx]

for freq_idx in range(len(freqsRange)):
    C_all[freq_idx]/=N_t[freq_idx]


for train_idx in range(len(SUBS_NAM)-1):#  train_idx, test_idx in kf.split(np.ones(num_fold)):
    print('Test subject: %s' %(train_idx))
    
    train_size = int(subj_each_len[train_idx])

    train_logP_L = np.zeros([train_size, 2*numF, 11])
    train_logP_R = np.zeros([train_size, 2*numF, 11])

    test_size = subj_each_len[test_idx]
    test_size = int(test_size)
    test_logP_L = np.zeros([test_size, 2*numF, 11])
    test_logP_R = np.zeros([test_size, 2*numF, 11])
    
    for freq_idx in range(len(freqsRange)):
        #avg_filter = np.zeros([2,n_chan,n_chan])
        source = np.zeros([n_chan,n_chan])
        target = np.zeros([n_chan,n_chan])
        source = (general_matrix[train_idx][freq_idx][0]+general_matrix[train_idx][freq_idx][1])/2
        target = (general_matrix[test_idx][freq_idx][0]+general_matrix[test_idx][freq_idx][1])/2
        G1 = source
        G2 = target
        start_idx = int(subj_each_idx_start[train_idx])
        end_idx = int(subj_each_idx_start[train_idx]+subj_each_len[train_idx])
        v = calculate_rccsp_norm(data_total[0][start_idx:end_idx, freq_idx], data_total[1][start_idx:end_idx, freq_idx],C_all[freq_idx],0.5)
        train_filt_R = apply_rccsp(data_total[0][start_idx:end_idx, freq_idx],v, numF)
        train_logP_R[:, :, freq_idx] = np.squeeze(bcitools.log_power(train_filt_R))
        train_filt_L = apply_rccsp(data_total[1][start_idx:end_idx, freq_idx], v,numF)
        train_logP_L[:, :, freq_idx] = np.squeeze(bcitools.log_power(train_filt_L))

        #create testing filter
        test_start = int(subj_each_idx_start[test_idx])
        test_end = int(test_start + subj_each_len[test_idx])
        test_filt_R = apply_rccsp(data_total[0][test_start:test_end, freq_idx], v, numF)
        test_logP_R[:, :, freq_idx] = np.squeeze(bcitools.log_power(test_filt_R))
        test_filt_L = apply_rccsp(data_total[1][test_start:test_end, freq_idx], v, numF)
        test_logP_L[:, :, freq_idx] = np.squeeze(bcitools.log_power(test_filt_L))

    train_logP_L = np.reshape(train_logP_L, (train_size, 22*numF))
    train_logP_R = np.reshape(train_logP_R, (train_size, 22*numF))
    test_logP_L = np.reshape(test_logP_L, (test_size, 22*numF))
    test_logP_R = np.reshape(test_logP_R, (test_size, 22*numF))

    

    # ---------------------------------------------------------------------------------------------------------------
    # train an LDA on CSP
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

    #print(vect1)
    tempp = np.sum(vect1 == 1) + np.sum(vect2 == 2)

    # since classes are balanced, calculate the overall accuracy
    classRate[count] = tempp / float(2 * temp[0])

    print(classRate[count])
    f.write('Subject %s: %f\n' % (count+1,classRate[count]))
    count = count + 1 
    

