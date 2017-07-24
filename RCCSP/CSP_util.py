import numpy as np
import scipy as sp
import random as rd
import scipy.io as sio

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