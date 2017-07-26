import numpy as np
import scipy as sp
import random as rd
import scipy.io as sio
from bcitools import balance_classes,log_power
from math import log
import numpy.linalg as nl

def KL_distance(Cs,Ct):
    Ct_inv = nl.inv(Ct)
    result = (log(nl.det(Ct)/nl.det(Cs))+np.trace(np.dot(Ct_inv,Cs))-len(Cs))/2
    return result

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
    [data_Gr, data_Gl] = balance_classes(data_Gr, data_Gl)
    [data_Br, data_Bl] = balance_classes(data_Br, data_Bl)
    [data_Gr, data_Bl] = balance_classes(data_Gr, data_Bl)
    [data_Br, data_Gl] = balance_classes(data_Br, data_Gl)

    # print(np.shape(data_Gl))

    data_Gr = data_Gr.transpose() #(189,11)
    data_Br = data_Br.transpose()
    data_Gl = data_Gl.transpose()
    data_Bl = data_Bl.transpose()

    data_Gr = data_Gr[:162]
    data_Br = data_Br[:162]
    data_Gl = data_Gl[:162]
    data_Bl = data_Bl[:162]

    data_Right = np.asarray(np.concatenate((data_Gr, data_Br), axis=0))
    data_Left = np.asarray(np.concatenate((data_Gl, data_Bl), axis=0))
    return [data_Right,data_Left]

def positive_transform(c1,c2):
    c3 = c1-c2
    d,v = sp.linalg.eigh(c3)
    c4 = np.dot(np.dot(v,np.diag(abs(d))),v.transpose())
    return c4


def find_nearest_pd(m):
    d,v = sp.linalg.eigh(m)
    for i in range(len(d)):
        if d[i]<0:
            d[i]=0
    m_new = np.dot(np.dot(v,np.diag(d)),v.transpose())
    return m_new


def calculate_rccsp_norm(x1,x2,Cs_Ct,alpha):
    #x1,x2 input matrix of two class
    #G1, G2 general matrix for class 1 2
    #alpha,beta parameter
    temp = np.shape(x1[0])
    n_chans = temp[0]
    c1 = np.zeros([n_chans,n_chans])
    c2 = np.zeros([n_chans,n_chans])
    num_trials = np.shape(x1)[0]
    for trial_idx in range(num_trials):
        c1 = c1+np.cov(x1[trial_idx])/np.trace(np.cov(x1[trial_idx]))
        c2 = c2+np.cov(x2[trial_idx])/np.trace(np.cov(x2[trial_idx]))
        #print(c1)

    c1 = np.divide(c1,num_trials)
    #print(num_trials)
    c2 = np.divide(c2,num_trials)
    

    d1,v1 = sp.linalg.eig(c1,c2+alpha*Cs_Ct)
    d2,v2 = sp.linalg.eig(c2,c1+alpha*Cs_Ct)
    
    indx1 = np.argsort(d1)
    indx2 = np.argsort(d2)

    indx1 = indx1[::-1]
    indx2 = indx2[::-1]

    v1 = v1.take(indx1, axis=1)
    v2 = v2.take(indx2, axis=1)

    return [v1,v2]

def calculate_rccsp_st(x1,x2,G1,G2,Cs_Ct,alpha,beta,source_num):
    #x1,x2 input matrix of two class
    #G1, G2 general matrix for class 1 2
    #alpha,beta parameter
    temp = np.shape(x1[0])
    n_chans = temp[0]
    c1 = np.zeros([n_chans,n_chans])
    c2 = np.zeros([n_chans,n_chans])
    num_trials = np.shape(x1)[0]
    for trial_idx in range(num_trials):
        c1 = c1+np.cov(x1[trial_idx])/np.trace(np.cov(x1[trial_idx]))
        c2 = c2+np.cov(x2[trial_idx])/np.trace(np.cov(x2[trial_idx]))
        #print(c1)

    c1 = np.divide(c1,num_trials)
    #print(num_trials)
    c2 = np.divide(c2,num_trials)
    if beta != 1:
        c1_ = (1-beta)/source_num*c1 + beta*G1
        c2_ = (1-beta)/source_num*c2 + beta*G2
    else:
        c1_ = c1
        c2_ = c2

    d1,v1 = sp.linalg.eig(c1_,c2_+alpha*Cs_Ct)
    d2,v2 = sp.linalg.eig(c2_,c1_+alpha*Cs_Ct)
    indx1 = np.argsort(d1)
    indx2 = np.argsort(d2)

    indx1 = indx1[::-1]
    indx2 = indx2[::-1]

    v1 = v1.take(indx1, axis=1)
    v2 = v2.take(indx2, axis=1)

    return [v1,v2]


def calculate_rccsp(x1,x2,G1,G2,source,target,alpha,beta,source_num):
    #x1,x2 input matrix of two class
    #G1, G2 general matrix for class 1 2
    #alpha,beta parameter
    temp = np.shape(x1[0])
    n_chans = temp[0]
    c1 = np.zeros([n_chans,n_chans])
    c2 = np.zeros([n_chans,n_chans])
    num_trials = np.shape(x1)[0]
    for trial_idx in range(num_trials):
        c1 = c1+np.cov(x1[trial_idx])/np.trace(np.cov(x1[trial_idx]))
        c2 = c2+np.cov(x2[trial_idx])/np.trace(np.cov(x2[trial_idx]))
        #print(c1)

    c1 = np.divide(c1,num_trials)
    #print(num_trials)
    c2 = np.divide(c2,num_trials)
    if beta != 1:
        c1_ = (1-beta)/source_num*c1 + beta*G1
        c2_ = (1-beta)/source_num*c2 + beta*G2
    else:
        c1_ = c1
        c2_ = c2

    #solve generalized eigenvalue problem
    
    #print(c1)
    #d,v = sp.linalg.eigh(c1)
    #print(d)
    #d,v = sp.linalg.eigh(c1_)
    #print(d)
    #d,v = sp.linalg.eigh(G1)
    #c1_ = find_nearest_pd(c1_)
    #c2_ = find_nearest_pd(c2_)
    #t1 = find_nearest_pd(c1_)
    #t2 = find_nearest_pd(c2_+alpha*positive_transform(source,target))
    #t3 = find_nearest_pd(c2_)
    #t4 = find_nearest_pd(c1_+alpha*positive_transform(source,target))

    
    #sio.savemat('test.mat',{'c1':c1_,'c2':c2_+alpha*positive_transform(source,target)})
    d1,v1 = sp.linalg.eig(c1_,c2_+alpha*positive_transform(source,target))
    d2,v2 = sp.linalg.eig(c2_,c1_+alpha*positive_transform(source,target))
    
    #d1,v1 = sp.linalg.eig(t1,t2)
    #d2,v2 = sp.linalg.eig(t3,t4)

    #d1,v1 = sp.linalg.eigh(c1,c2)
    #d2,v2 = sp.linalg.eigh(c2,c1)

    #select the highest value
    indx1 = np.argsort(d1)
    indx2 = np.argsort(d2)

    indx1 = indx1[::-1]
    indx2 = indx2[::-1]

    v1 = v1.take(indx1, axis=1)
    v2 = v2.take(indx2, axis=1)

    return [v1,v2]


def apply_rccsp(X, filt, col_num):

    temp = np.shape(filt[0])
    #print(temp)
    columns = np.concatenate((np.arange(0,col_num),np.arange(temp[1]-col_num,temp[1])))
    idx = np.arange(0,col_num)
    filt = np.asarray(filt)
    np.save('test',filt)
    #print(idx)
    #print(filt[0])
    filt[0] = np.asarray(filt[0])
    filt[1] = np.asarray(filt[1])

    f = np.concatenate([filt[0][:,:col_num],filt[1][:,:col_num]],axis=1)
    f = np.transpose(f)

    temp = np.shape(X)
    num_trials = temp[0]

    #dat = np.zeros(np.shape(X), dtype = object)
    dat = list()
    for ik in range(num_trials):
        temp = np.mat(f) * np.mat(X[ik])
        dat.append(temp)

    return dat

def CSP_filter(data_all,test_idx,general_matrix):
    freqsRange = np.array(
    [[1, 3], [2, 5], [4, 7],[6, 10], [7, 12], [10, 15], [12, 19], [18, 25], [19, 30], [25, 35], [30, 40]])
    n_chan = 47
    numF = 3
    data_total = [data_all[i] for i in range(len(data_all))]
    data_total = np.concatenate(data_total,axis = 1)
    subj_each_len = np.zeros(len(data_all))
    for i in range(len(data_all)):
        subj_each_len[i] = len(data_all[i][0])
    subj_each_idx_start = np.zeros(len(subj_each_len))
    total_len = 0
    for i in range(1,len(subj_each_len)):
        total_len += subj_each_len[i-1]
        subj_each_idx_start[i] = total_len
    
    print('Test subject: %s' %(test_idx))

    train_size = 0
    train_idx = np.concatenate([range(test_idx),range(test_idx+1,len(data_all))])
    for i in train_idx:
        train_size += subj_each_len[int(i)]
    train_size = int(train_size)

    train_logP_L = np.zeros([train_size, 2*numF, 11])
    train_logP_R = np.zeros([train_size, 2*numF, 11])

    #tempTest = np.shape(data_test)
    test_size = subj_each_len[test_idx]
    test_size = int(test_size)
    test_logP_L = np.zeros([test_size, 2*numF, 11])
    test_logP_R = np.zeros([test_size, 2*numF, 11])
    #print(train_data_Right[:,0][0].shape)
    

    

    for freq_idx in range(len(freqsRange)):
        avg_filter = np.zeros([2,n_chan,n_chan])
        for tidx_ in range(len(train_idx)):#  train_idx:
            tidx = int(train_idx[tidx_])
            source = np.zeros([n_chan,n_chan])
            target = np.zeros([n_chan,n_chan])
            #for itr in train_idx:
            source = (general_matrix[tidx][freq_idx][0]+general_matrix[tidx][freq_idx][1])/2
            target = (general_matrix[test_idx][freq_idx][0]+general_matrix[test_idx][freq_idx][1])/2
            G1 = np.zeros([n_chan,n_chan])
            G2 = np.zeros([n_chan,n_chan])
            for itr in train_idx:
                G1 += general_matrix[int(itr)][freq_idx][0]
                G2 += general_matrix[int(itr)][freq_idx][1]
            G1/=len(train_idx)
            G2/=len(train_idx)
            source = (G1+G2)/2
            start_idx = int(subj_each_idx_start[tidx])
            end_idx = int(subj_each_idx_start[tidx]+subj_each_len[tidx])
            #print('start idx: %s' %(start_idx))
            #print('end idx: %s' % (end_idx))
            v = calculate_rccsp(data_total[0][start_idx:end_idx, freq_idx], data_total[1][start_idx:end_idx, freq_idx],G1,G2,source,target,0.5,0.5,len(data_all)-1)
            avg_filter += v
            #extract train features
            train_filt_R = apply_rccsp(data_total[0][start_idx:end_idx, freq_idx],v, numF)
            if tidx > test_idx:
                start_idx-=int(subj_each_len[int(test_idx)])
                end_idx -= int(subj_each_len[int(test_idx)])
            train_logP_R[start_idx:end_idx, :, freq_idx] = np.squeeze(log_power(train_filt_R))
            train_filt_L = apply_rccsp(data_total[1][start_idx:end_idx, freq_idx], v,numF)
            train_logP_L[start_idx:end_idx, :, freq_idx] = np.squeeze(log_power(train_filt_L))
            #extract test features
        test_start = int(subj_each_idx_start[test_idx])
        test_end = int(test_start + subj_each_len[test_idx])
        #print('Test start: %s '%(test_start))
        #print('Test end: %s'%(test_end))
        avg_filter /= len(train_idx)
        test_filt_R = apply_rccsp(data_total[0][test_start:test_end, freq_idx], avg_filter, numF)
        test_logP_R[:, :, freq_idx] = np.squeeze(log_power(test_filt_R))
        test_filt_L = apply_rccsp(data_total[1][test_start:test_end, freq_idx], avg_filter, numF)
        test_logP_L[:, :, freq_idx] = np.squeeze(log_power(test_filt_L))

    train_logP_L = np.reshape(train_logP_L, (train_size, 22*numF))
    train_logP_R = np.reshape(train_logP_R, (train_size, 22*numF))

    test_logP_L = np.reshape(test_logP_L, (test_size, 22*numF))
    test_logP_R = np.reshape(test_logP_R, (test_size, 22*numF))
    X_train_start = 0
    X_train_end = 0
    X_train = list()
    y_train = list()
    X_val = list()
    y_val = list()
    X_test = list()
    y_test = list()

    val_size = int(test_size*0.1)

    for i in train_idx:

        X_train_start = int(subj_each_idx_start[int(i)])
        if(i>test_idx):
            X_train_start -=int(subj_each_len[int(test_idx)])

        X_train_end = X_train_start+subj_each_len[int(i)]
        X_train_end = int(X_train_end)
        X_train += train_logP_R[X_train_start:X_train_end- val_size ].tolist()
        X_train += train_logP_L[X_train_start:X_train_end  -val_size ].tolist()
        y_train += [0]*(X_train_end- X_train_start -val_size)
        y_train += [1]*(X_train_end- X_train_start - val_size )

        X_val += train_logP_R[X_train_end-val_size:X_train_end].tolist()
        X_val += train_logP_L[X_train_end-val_size:X_train_end].tolist()
        y_val += [0]*(val_size)
        y_val += [1]*(val_size)




    X_test += test_logP_R[:-val_size].tolist()
    X_test += test_logP_L[:-val_size].tolist()
    y_test += [0]*(test_size-val_size) 
    y_test += [1]*(test_size-val_size)

    #X_val += test_logP_R[-val_size:].tolist()
    #X_val += test_logP_L[-val_size:].tolist()
    #y_val += [0]*val_size
    #y_val += [1]*val_size

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    y_train = y_train.astype("int32", casting='unsafe')
    y_val = y_val.astype("int32",casting = 'unsafe')
    y_test = y_test.astype("int32", casting = 'unsafe')

    return [X_train,y_train,X_test,y_test,X_val,y_val]
