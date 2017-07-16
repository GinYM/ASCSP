import numpy as np
import scipy as sp
import random as rd

# ----------------------------------------------------------------------------------------------------------------------
# Define the functions needed
def calculate_csp(x1,x2):


    # compute covariance matrices of the two classes
    temp = np.shape(x1[0])
    n_chans = temp[0]
    n_chans1 = np.shape(x2[0])[0]

    temp = np.shape(x1)
    num_trials = temp[0]

    c1 = np.zeros([n_chans,n_chans])
    c2 = np.zeros([n_chans, n_chans])

    for ik in range(num_trials):
        c1 = c1+np.cov(x1[ik])/np.trace(np.cov(x1[ik]))
        c2 = c2+np.cov(x2[ik])/np.trace(np.cov(x2[ik]))
        #print(np.cov(x1[ik]).shape)
        #print(c1.shape)

    c1 = np.divide(c1,num_trials)
    c2 = np.divide(c2,num_trials)

    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)
    lambda_,u = sp.linalg.eigh(c1+c2)
    ind = np.argsort(lambda_)[::-1]
    lambda2_ = lambda_[ind[0:np.linalg.matrix_rank(c1+c2)]]

    u = u[:,ind[0:np.linalg.matrix_rank(c1+c2)]]
    p= np.dot(np.sqrt(sp.linalg.pinv(np.diag(lambda2_))),u.transpose())

    # solve the generalized eigen value problem
    w_1 = np.dot(np.dot(p,c1),p.transpose())
    w_2 = np.dot(np.dot(p,c2), p.transpose())


    d,v = sp.linalg.eigh(w_1,w_2)

    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    v = v.take(indx, axis=1)
    filts = np.dot(v.transpose(),p)

    filts = filts.transpose()
    a = sp.linalg.pinv(filts).transpose()

    #print(np.shape(filts))

    return filts, a

def apply_csp(X, filt, col_num):

    temp = np.shape(filt)
    columns = np.concatenate((np.arange(0,col_num),np.arange(temp[1]-col_num,temp[1])))

    f = filt[:, columns]

    f = np.transpose(f)

    temp = np.shape(X)
    num_trials = temp[0]

    #dat = np.zeros(np.shape(X), dtype = object)
    dat = list()
    for ik in range(num_trials):
        temp = np.mat(f) * np.mat(X[ik])
        dat.append(temp)

    return dat


def balance_classes(X1,X2):
    # this is to balance the number of instances in X1 and X2
    S1 = np.shape(X1)
    S2 = np.shape(X2)


    if S1[1]>S2[1]:
        while (S1[1]>S2[1]):
            t = rd.randint(0,S1[1]-S2[1])
            X1 = np.delete(X1,t,1)
            S1 = np.shape(X1)
            S2 = np.shape(X2)
    else:
        while (S1[1]<S2[1]):
            t = rd.randint(0,S2[1]-S1[1])
            X2 = np.delete(X2,t,1)
            S1 = np.shape(X1)
            S2 = np.shape(X2)
    return X1, X2

def balance_classes_1D(X1,X2):
    # this is to balance the number of instances in X1 and X2
    S1 = len(X1)
    S2 = len(X2)


    if S1>S2:
        while (S1>S2):
            t = rd.randint(0,S1-S2)
            X1 = np.delete(X1,t)
            S1 = len(X1)
            S2 = len(X2)
    else:
        while (S1<S2):
            t = rd.randint(0,S2-S1)
            X2 = np.delete(X2,t)
            S1 = len(X1)
            S2 = len(X2)
    return X1, X2

def log_power(X):

    # this is to calculate the log-power of each channel
    # X has dimensions : trial, channel, time
    # should be normalized with norm 1

    temp = np.shape(X)
    #print(np.shape())
    n_trials = temp[0]
    dat = list()
    for ik in range(n_trials):
        temp = np.var(X[ik],1)
        temp = temp/np.linalg.norm(temp,1)
        temp = np.log(temp)
        dat.append(temp)

    return dat

def load_and_balance_classes(data_W):
    data_Gr = list()
    data_Br = list()
    data_Gl = list()
    data_Bl = list()

    for il in range(len(data_W)):
        data = data_W[il]
        data_con = data['prepData']

        data_Gr.append(data_con[0, 0,]['G_r'])
        data_Gl.append(data_con[0, 0,]['G_l'])
        data_Br.append(data_con[0, 0,]['B_r'])
        data_Bl.append(data_con[0, 0,]['B_l'])

    data_Gr = np.squeeze(np.asarray(data_Gr))
    data_Gl = np.squeeze(np.asarray(data_Gl))
    data_Bl = np.squeeze(np.asarray(data_Bl))
    data_Br = np.squeeze(np.asarray(data_Br))

    # balance classes
    [data_Gr, data_Gl] = balance_classes(data_Gr, data_Gl)
    [data_Br, data_Bl] = balance_classes(data_Br, data_Bl)
    [data_Gr, data_Bl] = balance_classes(data_Gr, data_Bl)
    [data_Br, data_Gl] = balance_classes(data_Br, data_Gl)

    data_Gr = data_Gr.transpose()
    data_Br = data_Br.transpose()
    data_Gl = data_Gl.transpose()
    data_Bl = data_Bl.transpose()

    return data_Gr, data_Br, data_Gl, data_Bl




