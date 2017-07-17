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
import pickle

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


SUBS_NAM = ['S_BP-130516-1','S_BP-220416-2','S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2']
freqsRange = np.array(
    [[1, 3], [2, 5], [4, 7],[6, 10], [7, 12], [10, 15], [12, 19], [18, 25], [19, 30], [25, 35], [30, 40]])

# store data into whole matrix
# data: (subjectnum,Left or Right,experiment_time,trial,time)
'''
data_all = list()
for sub_idx in range(len(SUBS_NAM)):
    data_tmp = list()
    for freqs_idx in range(len(freqsRange)):
        namm = '/media/gin/hacker/UCSD_Summer_Research/data/output/' + SUBS_NAM[sub_idx] + \
               'freqs' + str(freqsRange[freqs_idx][0]) + '_' + str(freqsRange[freqs_idx][1]) + '_shams.mat'
        
        data_tmp.append(sio.loadmat(namm))
    tmp = np.asarray(get_data_from_mat(data_tmp))
    
    data_all.append(tmp)

with open("cropData.txt","wb") as fp:
	pickle.dump(data_all,fp)
'''

with open("../data/cropData.txt","rb") as fp:
	b = pickle.load(fp)
data = list()
subj_num = list()
for sub_idx in range(len(b)):
	print(sub_idx)
	for LR in range(len(b[sub_idx])):
		print(len(b[sub_idx][LR]))
		for num in range(len(b[sub_idx][LR])):
			data_tmp = list()
			for freqs_idx in range(len(b[sub_idx][LR][num])):
				for electrode in range(len(b[sub_idx][LR][num][freqs_idx])):
					time_win_idx = range(0,len(b[sub_idx][LR][num][freqs_idx][electrode]),40)
					for t_idx in range(len(time_win_idx)-1) :
						data_tmp.append(sum(b[sub_idx][LR][num][freqs_idx][electrode][time_win_idx[t_idx]:time_win_idx[t_idx+1]])/40)
			data_tmp.append(LR)
			data.append(data_tmp)
			subj_num.append(sub_idx)
print("Saving data!")
data = np.asarray(data)
np.save('../data/data_dalstm.npy',data)
subj_num = np.asarray(subj_num)
np.save('../data/subj_num.npy',subj_num)

'''
with open("./data_dalstm.txt","wb") as fp:
	pickle.dump(data,fp)

with open("../data/data_dalstm_subj.txt","wb") as fp:
	pickle.dump(subj_num,fp)
'''
