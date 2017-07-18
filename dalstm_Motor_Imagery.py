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

class ReverseGradient(theano.gof.Op):
    view_map = {0:[0]}
    __props__=('hp_lambda',)
    def __init__(self,hp_lambda):
        super(ReverseGradient,self).__init__()
        self.hp_lambda=hp_lambda
    def make_node(self,x):
        return theano.gof.graph.Apply(self,[x],[x.type.make_variable()])
    def perform(self,node,inputs,output_storage):
        xin,=inputs
        xout,=output_storage
        xout[0]=xin
    def grad(self,input,output_gradients):
        return [-self.hp_lambda*output_gradients[0]]
        #return [self.hp_lambda*output_gradients[0]]

class ReverseGradientLayer(lasagne.layers.Layer):
    def __init__(self, incoming, hp_lambda, **kwargs):
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)
        self.op = ReverseGradient(hp_lambda)

    def get_output_for(self, input, **kwargs):
        return self.op(input)

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


def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param w_init: Initial weight values
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers in each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """
    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
    # Input layer
    network = InputLayer(shape=(None, n_colors, imsize, imsize),
                                        input_var=input_var)
    #print(len(w_init))
    for i, s in enumerate(n_layers):
        for l in range(s):
            # Add ReLU
            #print(count)
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3),
                          W=w_init[count], pad='same')
            #add batch normalization
            weights.append(network.W)
            network = batch_norm(network)
            count += 1
            
        network = MaxPool2DLayer(network, pool_size=(2, 2))
    return network, weights


def build_convpool_max(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=3):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(convnet)
    # convpooling using Max pooling over frames
    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def build_convpool_conv1d(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=3):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    convpool = DimshuffleLayer(convpool, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    convpool = Conv1DLayer(convpool, 64, 3)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def build_convpool_lstm(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet 7
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, numTimeWin, features]    //[n_samples, features, numTimeWin] 
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.
    convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    
    

    return convpool


def build_convpool_dalstm(input_vars,input_vars_target, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
    """
    
    Builds the DALSTM (Domain Adaptation LSTM) network to maintain domain-invariant features of EEG signals
    
    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    n_layers = (4,2,1)
    domain_classes = 2
    dropout_value = 0
    is_gradient_reversal = False #True

    ## change here!!
    da_lambda = 0

    convnets = []
    daconvnets = []
    w_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            #print(input_vars[i])
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors,n_layers = n_layers)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors,n_layers = n_layers)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    #build DAN
    store_weight = []
    daconvpool = DimshuffleLayer(convpool, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    
    if is_gradient_reversal:
        daconvpool = ReverseGradientLayer(daconvpool,da_lambda)

    daconvpool = Conv1DLayer(daconvpool, 64, 3)
    store_weight.append(daconvpool.W)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    
    
    get_gradient_reversal_0 = daconvpool
    daconvpool = DenseLayer(lasagne.layers.dropout(daconvpool, p=dropout_value),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    store_weight.append(daconvpool.W)
    # And, finally, the output layer with 50% dropout on its inputs:
    daconvpool = DenseLayer(lasagne.layers.dropout(daconvpool, p=dropout_value),
            num_units=domain_classes, nonlinearity=lasagne.nonlinearities.softmax)
    store_weight.append(daconvpool.W)


    ######################Create Target Domain Adaptation Network
    convnets = []
    target_net = []
    for i in range(n_timewin):
        #if i == 0:
        #    convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        #else:
        convnet, _ = build_cnn(input_vars_target[i], w_init=w_init, imsize=imsize, n_colors=n_colors,n_layers = n_layers)
        convnets.append(FlattenLayer(convnet))
    target_convpool = ConcatLayer(convnets)
    target_convpool = ReshapeLayer(target_convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    ####build DAN
    target_net = DimshuffleLayer(target_convpool, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    
    if is_gradient_reversal:
        target_net = ReverseGradientLayer(target_net,da_lambda)

    target_net = Conv1DLayer(target_net, 64, 3,W=store_weight[0])
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    
    
    get_gradient_reversal_1 = target_net
    target_net = DenseLayer(lasagne.layers.dropout(target_net, p=dropout_value),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify,W=store_weight[1])
    #target_output_nosm = target_net
    # And, finally, the output layer with 50% dropout on its inputs:
    target_net = DenseLayer(lasagne.layers.dropout(target_net, p=dropout_value),
            num_units=domain_classes, nonlinearity=lasagne.nonlinearities.softmax,W=store_weight[2])
    target_output_nosm = target_net

    ###########################################################################################
    

    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    get_feature_extract = convpool
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.
    convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    # And, finally, the output layer with 50% dropout on its inputs:
    
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    

    

    return [convpool,daconvpool,target_net,target_output_nosm,get_gradient_reversal_0,get_gradient_reversal_1,get_feature_extract]
    

def build_convpool_mix(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
    """
    Builds the complete network with LSTM and 1D-conv layers combined

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    conv_out = FlattenLayer(conv_out)
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
    lstm_out = SliceLayer(lstm, -1, 1)
    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out])
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(convpool,
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]

def da_iterate_minibatches(input_s,input_t,source_label, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    
    if input_s.ndim == 4:
        #input_len = inputs.shape[0]
        input_len_s = input_s.shape[0]
        input_len_t = input_t.shape[0]
    elif input_s.ndim == 5:
        #input_len = inputs.shape[1]
        input_len_s = input_s.shape[1]
        input_len_t = input_t.shape[1]
    assert input_len_s == len(source_label)
    
    assert batchsize<input_len_t
    if shuffle:
        #indices = np.arange(input_len)
        indices_source = np.arange(input_len_s)
        indices_target = np.arange(input_len_t)
        np.random.shuffle(indices_source)
        np.random.shuffle(indices_target)
    for start_idx in range(0, input_len_s, batchsize):
        
        target_start = start_idx%input_len_t    
        target_end = (start_idx + batchsize)%input_len_t
        index1 = np.arange(target_start ,input_len_t)
        index2 = np.arange(0,target_end)
        index_all = np.append(index1,index2)
        if shuffle:
            excerpt_source = indices_source[start_idx:start_idx + batchsize]

            #target_start = start_idx%input_len_t    
            #target_end = (start_idx + batchsize)%input_len_t
            if target_start<target_end:
                excerpt_target = indices_target[target_start:target_end]
            else:
                #index1 = np.arange(start_idx,input_len_t)
                #index2 = np.arange(0,target_end)
                excerpt_target = indices_target[index_all]

            #excerpt_target = indices_target[start_idx%input_len_t:start_idx + batchsize]
        else:
            excerpt_source = slice(start_idx, start_idx + batchsize)
            if target_start < target_end:
                excerpt_target = slice(target_start,target_end)
            else:
                excerpt_target = index_all
        #print(excerpt_source)
        #print(excerpt_target)
        if input_s.ndim == 4:
            yield input_s[excerpt_source],input_t[excerpt_target],source_label[excerpt_source] 
        elif input_s.ndim == 5:
            yield input_s[:,excerpt_source],input_t[:,excerpt_target],source_label[excerpt_source] #inputs[:, excerpt], targets[excerpt]


def train(images, labels, fold, model_type, batch_size=32, num_epochs=5):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param images: input images (7,2670,3,32,32)
    :param labels: target labels
    :param fold: tuple of (train, test) index numbers
    :param model_type: model type ('cnn', '1dconv', 'maxpool', 'lstm', 'mix')
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :return: none
    """
    num_classes = len(np.unique(labels))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)

    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    # Prepare Theano variables for inputs and targets
    input_var = T.TensorType('floatX', ((False,) * 5))()
    target_var = T.ivector('targets')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    # Building the appropriate model
    if model_type == '1dconv':
        network = build_convpool_conv1d(input_var, num_classes)
    elif model_type == 'maxpool':
        network = build_convpool_max(input_var, num_classes)
    elif model_type == 'dalstm':
        [source_class,source_domain,target_domain,target_output_nosm,gd0,gd1] = build_convpool_dalstm(input_var,num_classes,100)
    elif model_type == 'lstm':
        network = build_convpool_lstm(input_var, num_classes, 100)
    elif model_type == 'mix':
        network = build_convpool_mix(input_var, num_classes, 100)
    elif model_type == 'cnn':
        input_var = T.tensor4('inputs')
        
        
        network, _ = build_cnn(input_var,imsize=images.shape[2], n_colors=images.shape[1])
        network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                             num_units=256,
                             nonlinearity=lasagne.nonlinearities.rectify)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                             num_units=num_classes,
                             nonlinearity=lasagne.nonlinearities.softmax)
    else:
        raise ValueError("Model not supported ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']")
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    


    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    # Finally, launch the training loop.
    print("Starting training...")
    best_validation_accu = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            #print(inputs)
            #print(targets)
            train_err += train_fn(inputs, targets)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        av_train_err = train_err / train_batches
        av_val_err = val_err / val_batches
        av_val_acc = val_acc / val_batches
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(av_train_err))
        print("  validation loss:\t\t{:.6f}".format(av_val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))
        if av_val_acc > best_validation_accu:
            best_validation_accu = av_val_acc
            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            av_test_err = test_err / test_batches
            av_test_acc = test_acc / test_batches
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(av_test_err))
            print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
            # Dump the network weights to a file like this:
            np.savez('weights_lasg_{0}'.format(model_type), *lasagne.layers.get_all_param_values(network))
    print('-'*50)
    print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))


def datrain(images, labels, fold, model_type, batch_size=32, num_epochs=5):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param images: input images
    :param labels: target labels
    :param fold: tuple of (train, test) index numbers
    :param model_type: model type ('cnn', '1dconv', 'maxpool', 'lstm', 'mix','dalstm')
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :return: none
    """
    #print(len(fold))
    num_classes = len(np.unique(labels))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
    
    domain_source = np.zeros((X_train.shape[0],1))
    domain_target = np.ones((X_test.shape[0],1))
    domain_val = np.zeros((X_val.shape[0],1))

    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    # Prepare Theano variables for inputs and targets
    input_var = T.TensorType('floatX', ((False,) * 5))()


    input_var_source = T.TensorType('floatX', ((False,) * 5))()
    input_var_target = T.TensorType('floatX', ((False,) * 5))()

    target_var = T.ivector('targets')

    source_target = T.ivector('source_target')
    class_label = T.ivector('class_label')
    source_domain_label = T.ivector('source_domain_label')
    target_domain_label = T.ivector('target_domain_label')
    domain_label = T.ivector('domain_label')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    # Building the appropriate model
    if model_type == '1dconv':
        network = build_convpool_conv1d(input_var, num_classes,imsize=images.shape[3], n_colors=images.shape[2])
    elif model_type == 'maxpool':
        network = build_convpool_max(input_var, num_classes)
    elif model_type == 'dalstm':
        #input_var_source = T.tensor4('inputs_source')
        #input_var_target = T.tensor4('inputs_target')
        [source_class,source_domain,target_domain,target_output_nosm,gd0,gd1,get_feature_extract] = build_convpool_dalstm(input_var_source,input_var_target,num_classes,100,imsize=images.shape[3], n_colors=images.shape[2])
    elif model_type == 'lstm':
        network = build_convpool_lstm(input_var, num_classes, 100)
    elif model_type == 'mix':
        network = build_convpool_mix(input_var, num_classes, 100)
    elif model_type == 'cnn':
        input_var = T.tensor4('inputs')
        network, _ = build_cnn(input_var)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                             num_units=256,
                             nonlinearity=lasagne.nonlinearities.rectify)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                             num_units=num_classes,
                             nonlinearity=lasagne.nonlinearities.softmax)
    else:
        raise ValueError("Model not supported ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn','dalstm']")
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    #print(lasagne.layers.get_all_params(source_class))
    #print(lasagne.layers.get_all_params(source_domain))
    #print(lasagne.layers.get_all_params(target_domain))
    source_class_prediction = lasagne.layers.get_output(source_class)
    source_domain_classifier = lasagne.layers.get_output(source_domain)
    target_domain_classifier = lasagne.layers.get_output(target_domain)
    
    #prediction = lasagne.layers.get_output(network)
    
    #source_class_label = T.ivector('source_class_label')
    #source_domain_label = T.ivector('source_domain_label')
    #target_domain_label = T.ivector('target_domain_label')
    
    source_class_loss = lasagne.objectives.categorical_crossentropy(source_class_prediction,class_label)
    source_domain_loss = lasagne.objectives.categorical_crossentropy(source_domain_classifier,source_domain_label)
    target_domain_loss = lasagne.objectives.categorical_crossentropy(target_domain_classifier,target_domain_label)
    #loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    
    #total_loss = source_class_loss+source_domain_loss+target_domain_loss
    #total_loss = total_loss.mean()
    #print(pp(total_loss))
    source_class_loss = source_class_loss.mean()
    source_domain_loss = source_domain_loss.mean()
    target_domain_loss = target_domain_loss.mean()
    #loss = loss.mean()
    total_domain_loss = source_domain_loss+target_domain_loss
    total_loss = source_class_loss+source_domain_loss+target_domain_loss
    #total_loss = T.add(source_class_loss,source_domain_loss,target_domain_loss)

    params_class = lasagne.layers.get_all_params(source_class,trainable=True)
    params_domain = lasagne.layers.get_all_params(source_domain,trainable=True)

    params_total = lasagne.layers.get_all_params([source_class,source_domain],trainable=True)

    #params = lasagne.layers.get_all_params(network, trainable=True)
    # change the learning rate
    updates_class = lasagne.updates.adam(source_class_loss,params_class,learning_rate=0.001)
    #updates_domain = lasagne.updates.adam(source_domain_loss,params_domain,learning_rate=0.001)
    updates_domain = lasagne.updates.adam(total_domain_loss,params_domain,learning_rate=0.001)
    #updates_total = lasagne.updates.adam(source_class_loss,params_class,learning_rate=0.001)
    updates_total = lasagne.updates.adam(total_loss,params_total,learning_rate=0.001)
    #theano.printing.debugprint(total_loss) 

    #updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    
    test_prediction_class=lasagne.layers.get_output(source_class,deterministic=True)

    #test_prediction = lasagne.layers.get_output(network, deterministic=True)
    ##???
    test_loss_class = lasagne.objectives.categorical_crossentropy(test_prediction_class,class_label)

    #test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    #                                                        target_var)
    test_loss_class = test_loss_class.mean()

    test_prediction_domain = lasagne.layers.get_output(target_domain,deterministic=True)
    test_loss_domain = lasagne.objectives.categorical_crossentropy(test_prediction_domain,target_domain_label)
    test_loss_domain = test_loss_domain.mean()


    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)

    test_acc_class = T.mean(T.eq(T.argmax(test_prediction_class, axis=1), class_label),
                      dtype=theano.config.floatX)
    test_acc_domain = T.mean(T.eq(T.argmax(test_prediction_domain, axis=1), target_domain_label),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    #train_fn = theano.function([input_var, target_var], loss, updates=updates)
    target_output = lasagne.layers.get_output(target_output_nosm,deterministic=True)
    gradient0 = lasagne.layers.get_output(gd0,deterministic=True)
    gradient1 = lasagne.layers.get_output(gd1,deterministic = True)
    feature_extract = lasagne.layers.get_output(get_feature_extract,deterministic=True)
    train_fn_class = theano.function([input_var_source, class_label], [source_class_loss,feature_extract], updates=updates_class)
    train_fn_domain = theano.function([input_var_source,input_var_target,source_domain_label,target_domain_label], total_domain_loss, updates=updates_domain)

    train_fn = theano.function([input_var_source,input_var_target,class_label,source_domain_label,target_domain_label],[total_loss,source_class_loss,source_domain_loss,target_domain_loss,target_output,gradient0,gradient1,feature_extract],updates=updates_total)


    ##here!!!!!!!!!!!!!!!!!!!!!! 
    #tomorrow continue!!! :)

    # Compile a second function computing the validation loss and accuracy:
    #val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    val_fn_class = theano.function([input_var_source,class_label],[test_loss_class,test_acc_class])
    
    val_fn_domain = theano.function([input_var_target,target_domain_label],[test_loss_domain,test_acc_domain,test_prediction_domain,target_output])
    #theano.printing.pydotprint(val_fn_domain,outfile="val_fn_domain.png",var_with_name_simple = True)
    #val_fn_all = theano.function([input_var_source,input_var_target,source_target])

    # Finally, launch the training loop.
    print("Starting training...")
    best_validation_accu = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        err_sc = 0
        err_sd = 0
        err_td = 0
        train_batches = 0
        start_time = time.time()

        for batch in da_iterate_minibatches(X_train,X_test, y_train, batch_size, shuffle=False):
            inputs_source,inputs_target,source_label= batch
            sd_label = np.squeeze(np.zeros([inputs_source.shape[1],1],dtype = np.int32))
            td_label = np.squeeze(np.ones([inputs_target.shape[1],1],dtype = np.int32))
            [train_loss,feature_output] = train_fn_class(inputs_source,source_label)
            print(feature_output)
            #train_loss = train_fn_domain(inputs_source,inputs_target,sd_label,td_label)
            
            ###change here!!!

            #[train_loss,loss_sc,loss_sd,loss_td,target_output,gd0,gd1] = train_fn(inputs_source,inputs_target,source_label,sd_label,td_label)
            train_err += train_loss
            #err_sc += loss_sc
            #err_sd += loss_sd   
            #err_td += loss_td 
            #print(target_output) 
            #print(gd0)
            #print(gd1)
            



            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_acc_domain = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs,targets = batch
            err, acc = val_fn_class(inputs, targets)
            td_label = np.squeeze(np.ones([inputs.shape[1],1],dtype = np.int32))
            sd_label = np.squeeze(np.zeros([inputs.shape[1],1],dtype = np.int32))
            _, domain_acc,output_result,output_target = val_fn_domain(inputs, sd_label)
            val_err += err
            val_acc += acc
            val_batches += 1
            val_acc_domain += domain_acc
            #np.savez('weights_target_domain_{0}'.format(model_type), *lasagne.layers.get_all_param_values(target_domain))
        #print(output_result)
        #print(output_target)
        av_train_err = train_err / train_batches
        av_train_class_err = err_sc / train_batches
        av_train_domain_err = (err_sd+err_td)/2/train_batches
        av_val_err = val_err / val_batches
        av_val_acc = val_acc / val_batches
        av_val_acc_domain = val_acc_domain / val_batches
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  total loss:\t\t{:.6f}".format(av_train_err))
        print("  class loss:\t\t{:.6f}".format(av_train_class_err))
        print("  domain loss:\t\t{:.6f}".format(av_train_domain_err))
        print("  validation loss:\t\t{:.6f}".format(av_val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))
        print("  validation domain accuracy:\t\t{:.2f} %".format(av_val_acc_domain * 100))
        if av_val_acc > best_validation_accu:
            best_validation_accu = av_val_acc
            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn_class(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            av_test_err = test_err / test_batches
            av_test_acc = test_acc / test_batches
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(av_test_err))
            print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
            # Dump the network weights to a file like this:
            np.savez('weights_lasg_{0}'.format(model_type), *lasagne.layers.get_all_param_values([source_class,source_domain,target_domain]))
            np.savez('weights_source_class_{0}'.format(model_type), *lasagne.layers.get_all_param_values(source_class))
            np.savez('weights_source_domain_{0}'.format(model_type), *lasagne.layers.get_all_param_values(source_domain))
            np.savez('weights_target_domain_{0}'.format(model_type), *lasagne.layers.get_all_param_values(target_domain))
    print('-'*50)
    print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
    

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
    
    #feats = scipy.io.loadmat('../EEGLearn-master/Sample data/FeatureMat_timeWin.mat')['features']
    #subj_nums = np.squeeze(scipy.io.loadmat('../EEGLearn-master/Sample data/trials_subNums.mat')['subjectNum'])
    
    # Leave-Subject-Out cross validation
    

    fold_pairs = []
    
    locs_2d = locs_2d[0:59]
    feats = np.load('../data/data_dalstm.npy')
    subj_nums = np.load('../data/subj_num.npy')
    
    

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

    each_freq = 59*11
    
    # CNN Mode
    print('Generating images...')
    # Find the average response over time windows
    #av_feats = reduce(lambda x, y: x+y, [feats[:, i*each_freq:(i+1)*each_freq] for i in range(feats.shape[1] / each_freq)]) #7 time windows
    #av_feats = av_feats / (feats.shape[1] / each_freq)
    images = np.load('../data/images_cnn.npy')
    #images = gen_images(np.array(locs_2d),
    #                              av_feats,
    #                              16, normalize=False)
    #np.save('../data/images_cnn.npy',images)
    print('\n')

    # Class labels should start from 0
    print('Training the CNN Model...')
    print(images.shape)
    print(feats.shape)
    train(images, np.squeeze(feats[:, -1]) , fold_pairs[2], 'cnn',batch_size=36)
    

    '''
    # Conv-LSTM Mode
    print('Generating images for all time windows...')

    
    image_size = 16
    '''
    images_timewin = np.array([gen_images(np.array(locs_2d),
                                                    feats[:, i * each_freq:(i + 1) * each_freq], image_size, normalize=False) for i in
                                         range(feats.shape[1] / each_freq)
                                         ])
    '''
    #images_timewin = np.array([gen_images(np.array(locs_2d),
    #                                                feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
    #                                     range(feats.shape[1] / 192)
    #                                     ])
    images_timewin = np.load('../data/images_timewin.npy')
    #np.save('../data/images_timewin.npy',images_timewin)
    print('\n')
    print('Training the LSTM-CONV Model...')
    #print(len(fold_pairs[2]))
    #print(fold_pairs[2][0])
    #print(fold_pairs[2][1])
    datrain(images_timewin, np.squeeze(feats[:, -1]) , fold_pairs[2], 'dalstm',batch_size=36,num_epochs=10)


    
    
    '''
    print('Done!')
