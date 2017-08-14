# Adaptive Selective CSP based Motor Imagery Classification for Subject to Subject Transfer

Summer research in De Sa's lab supervised by Prof. Virginia De Sa. EEG signals are used to interpret the inner brain activity.

## ASCSP

C1, C2 is initialized as mean covariance matrix in source subject.

Then repeatedly do the following two steps.

- Find two trials in target subjetc such that the difference of the mean is minimized and variance after subspace alignment is maximized.
- Update the covariance matrix with newly selected covariance matrix. C1 = C1\*n/(n+1)+Cnew1/(n+1) and C2 = C2\*n/(n+1) + Cnew2/(n+1)

After adapting C1, C2 with target trials, triditionals CSP is adpoted as spatial filter to reduce the number of electrodes from 47 int 6 which is 3 minimal eigen vactor and 3 maximal eigen vector. Log of variance is used to extract the feature in each dimension in each trial.

Because the subject difference, Subspace Alignment is adopted to reduce the source and target variance. After Subspace Alignment, the dimension is reduced to 2.

After domain adaptation, LDA is used to classify.

![no_sa](np_sa.eps)

