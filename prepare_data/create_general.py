import scipy.io as sio
import numpy as np
#create general covriant matrix for two class
SUBS_NAM = ['S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216']

data_name_in_mat = ['G_r','B_r','G_l','B_l']
freqsRange = np.array(
	[[1, 3], [2, 5], [4, 7],[6, 10], [7, 12], [10, 15], [12, 19], [18, 25], [19, 30], [25, 35], [30, 40]])
channel = 47
class_num = 2
general_matrix = np.zeros([len(SUBS_NAM),freqsRange.shape[0],class_num,channel,channel])
for subj_idx in range(len(SUBS_NAM)):
	print('Processing %s\n' % (subj_idx))
	for freq_idx in range(freqsRange.shape[0]):
		#print(freq_idx)
		namm = '/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' + SUBS_NAM[subj_idx] + 'freqs' + str(freqsRange[freq_idx][0]) + '_' + str(freqsRange[freq_idx][1]) + '_shams_FP.mat'
		#type data[0,0,]['G_r'][0][0:341][:47][:401]
		#print('Here! %s\n' %(freq_idx))
		data = sio.loadmat(namm)['prepData']
		#print(data)
		#print('Here aaaaa  ! %s\n' %(freq_idx))
		for name_idx in range(0,len(data_name_in_mat),2):
			#print('Here! %s\n' %(freq_idx))
			#print('Name id:%s' %(name_idx)) 
			cx = np.zeros([channel,channel])
			count_cx = 0
			for name_idx_1 in range(name_idx,name_idx+2):
				data_name = data_name_in_mat[name_idx_1]
				#print(data_name)
				data_each = data[0,0,][data_name]
				#n_chans = data_each[0][0].shape[0]

				#count_cx += data_each.shape[1]

				#cx = np.zeros(n_chans,n_chans)
				for trial in range(0,162):#data_each.shape[1]):
					count_cx = count_cx+1
					#print(trial)
					x = data_each[0][trial]
					#print(x.shape)
					x = np.asarray(x)
					cx += np.cov(x)/np.trace(np.cov(x))
					#print((subj_idx,name_idx))
					#if(freq_idx == 0 and subj_idx == 0 and name_idx == 0):
					#	print(cx)
					#print(x)
			print(count_cx)
			cx = np.divide(cx,count_cx)
			#print(cx)
			#print(cx)
				
			#print('%s %s %s' % (subj_idx,freq_idx,name_idx) )
			general_matrix[subj_idx][freq_idx][name_idx/2][:][:]=cx
			#print(general_matrix[0][0][0])
np.save('../../data/CSP/CSP_covariance_matrix_new.npy',general_matrix)






