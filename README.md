# Motor Imagery BCI using Transfer Learning

Summer research in UCSD, de sa's lab. Referenced to "[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)" and "[Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks](https://arxiv.org/pdf/1511.06448.pdf)", I propose a Domain Adaptation LSTM to classify Motor Imagery.

# Installation and Dependencies

In order to run this code you need to install the following modules:

Numpy and Scipy ([http://www.scipy.org/install.html](http://www.scipy.org/install.html))

Scikit-Learn ([http://scikit-learn.org/stable/install.html](http://scikit-learn.org/stable/install.html))

Theano ([http://deeplearning.net/software/theano/install.html](http://deeplearning.net/software/theano/install.html))

Lasagne ([http://lasagne.readthedocs.org/en/latest/user/installation.html](http://lasagne.readthedocs.org/en/latest/user/installation.html))

```
pip install -r requirements.txt
```

```
pip install [path_to_EEGLearn]
```

# Notation
This code is modified from [EEGLearn](https://github.com/pbashivan/EEGLearn)

