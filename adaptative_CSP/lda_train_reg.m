% perform LDA on some data
% based on "Fisher Linear Discriminant Analysis" -- Max Welling
% and misc. random stuff I found on the internet ...
%
% INPUT
% x - design matrix -- row vectors
% y - data set outputs, assumed to be passed in as a single column vector
%     and consist of classes 0 and 1
%
% OUTPUT
% w  - weight vector
%
% 09/28/07 -- created
% 02/17/09 -- modified for current classification format
%
% function w = train_LDA(X, Y)
function [W B class_mv] = lda_train_reg(X, Y, k_reg)

% get the list of classes
class_list = unique(Y);

% extract a few useful things
ind0 = find(Y == class_list(2));
ind1 = find(Y == class_list(1));
num0 = length(ind0);
num1 = length(ind1);

% first find the mean for each class
%size(X)
m0 = mean(X(ind0, :), 1)';
m1 = mean(X(ind1, :), 1)';

% compute the within-class scatter matrices
% be lazy -- use cov and multiply by class count
% NOTE: need to nomalize by n, not n - 1 for this to work ...
S_0 = cov(X(ind0, :), 1) * num0;
S_1 = cov(X(ind1, :), 1) * num1;

% % Sanity Check for scatter matrices
% S_0_test = zeros(2);
% for k = 1 : num0
%     S_W0_test = S_W0_test + (X(ind0(k), :)' - m0) * (X(k, :)' - m0)';
% end

% total within-class scatter
S_W = S_0 + S_1;

% Regularization
% from Li et al. "Using Discriminant Analysis for Multi-Class Classification"
% Proc. 3rd IEEE International Conference on Data Mining (ICDM'03)
d = mean(diag(S_W)) * k_reg;
[dim1, dim2] = size(S_W);
S_W = (1-k_reg)*S_W + (eye(dim1) * d);
opts.disp = 0;  % display less diagnostic data

% solve for optimal projection
W = pinv(S_W) * (m0 - m1);

B = (m0'*W+m1'*W)/2;

% project the data onto this line
% X_LDA = w' * X';
X_LDA = X  * W;  % same as (w' * X')'

X_LDA = X_LDA-B;

class_mv.mu0 = m0;
class_mv.mu1 = m1;
class_mv.Sigma = (S_0 + S_1)/(num0+num1);
