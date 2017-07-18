from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m

import scipy.io
import theano
import theano.tensor as T

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from utils import augment_EEG, cart2sph, pol2cart

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

from theano.tensor import basic as tensor
from lasagne.layers import batch_norm

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes 64
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0 #192%64 = 0
    n_colors = features.shape[1] / nElectrodes # 11 
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints])) # 32*32
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]

if __name__ == '__main__':
    from utils import reformatInput

    # Load electrode locations
    print('Loading data...')
    locs = scipy.io.loadmat('../EEGLearn-master/Sample data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))
    
    # Just need the first 59 electrodes.
    locs_2d = locs_2d[0:59]
    feats = np.load('../data/data_dalstm.npy')
    #feats = scipy.io.loadmat('../EEGLearn-master/Sample data/FeatureMat_timeWin.mat')['features']
    #subj_nums = np.squeeze(scipy.io.loadmat('../EEGLearn-master/Sample data/trials_subNums.mat')['subjectNum'])
    subj_nums = np.load('../data/subj_num.npy')
    # Leave-Subject-Out cross validation
    fold_pairs = []

    min_crop = 100000
    last_idx = 0
    change_point = list()
    for i in range(feats.shape[0]-1):
        if feats[i][-1]!=feats[i+1][-1]:
            change_point.append(i+1)
            if i-last_idx +1 < min_crop:
                min_crop = i-last_idx+1
            last_idx = i+1
    if feats.shape[0]-1-last_idx+1 < min_crop:
        min_crop = feats.shape[0]-1-last_idx+1
    print(min_crop)

    #new_sub_idx = range(0,min_crop)
    shuf_idx = range(2*min_crop)
    np.random.shuffle(shuf_idx)
    new_sub_idx = list()
    
    for i in range(0,len(change_point),2):
        if i == 0:
            tmp = range(0,0+min_crop)+range(change_point[i],change_point[i]+min_crop)
        else:
            tmp = range(change_point[i-1],change_point[i-1]+min_crop)+range(change_point[i],change_point[i]+min_crop)
        new_tmp = [tmp[shuf_idx[ii]] for ii in range(len(shuf_idx))]
        #print(new_tmp)
        #tmp = tmp[shuf_idx]
        new_sub_idx += new_tmp
        
    subj_nums = subj_nums[new_sub_idx]
    print(len(subj_nums))


    for i in np.unique(subj_nums):
        ts = (subj_nums == i) #change here!!!!
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        #np.random.shuffle(tr)  # Shuffle indices
        #np.random.shuffle(ts)
        fold_pairs.append((tr, ts)) #training and testing sets

        
    
    # Conv-LSTM Mode
    print('Generating images for all time windows...')

    each_freq = 59*11
    image_size = 32
    
    images_timewin = np.array([gen_images(np.array(locs_2d),
                                                    feats[:, i * each_freq:(i + 1) * each_freq], image_size, normalize=False) for i in
                                         range(feats.shape[1] / each_freq)
                                         ])
    
    #images_timewin = np.array([gen_images(np.array(locs_2d),
    #                                                feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
    #                                     range(feats.shape[1] / 192)
    #                                     ])
    #images_timewin = np.load('../data/images_timewin.npy')
    np.save('../data/images_timewin_32.npy',images_timewin)
    print('\n')
    print('Done!')
