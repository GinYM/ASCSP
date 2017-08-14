% L2 regularized CSP
%
% INPUTS
% input_data: data separated by class(data can be either matrix or cell
% format)
% csp_dim: # of CSP filters per class
% C: Regularization parameter
%
% OUTPUTS
% csp_coeff: selected CSP filters (n_classes*csp_dim by n_channels)
%            (NOTE: filters are reorganized so [ c1 c2 c1 c2];
% result: all CSP spatial filter coefficients(n_channels by n_channels)
%         result(1,:) -- most discriminant for class1
%         result(n_channels,:) -- most discriminant for class2
%         for visulization of mixing patterns: A = inv(result')'


function [ csp_coeff result ] = csp_L2_reg_csp_norm_cov(input_data, csp_dim, C)

% extract useful values
n_classes = length(input_data);

% if input_data has matrix format change to cell format
if(~iscell(input_data{1}))
    input_classes = cell(1,n_classes);
    for class = 1:n_classes
        input_classes{class} = mat_to_cell(input_data{class});
    end
else
    input_classes = input_data;
end

[ n_channels n_samples ]  = size(input_classes{1}{1});

n_trials = zeros(1,n_classes);

for class = 1:n_classes
    n_trials(class) = length(input_classes{class});
end

cov_classes = cell(1,n_classes);

for i = 1:n_classes
    for j = 1:n_trials(i)
        cov_classes{i}{j} = cov(input_classes{i}{j}',1)/trace(cov(input_classes{i}{j}',1));
    end
end

R = cell(1,n_classes);

for i = 1:n_classes
    R{i} = zeros(n_channels, n_channels);
    for j = 1:n_trials(i)
        R{i} = R{i}+cov_classes{i}{j};
    end
    R{i} = R{i}/n_trials(i);
end

Rsum = R{1} + R{2};

% Regularize the common Cov matrix(L2 regularization)
% if C=0, normal CSP
d_C = mean(diag(Rsum)) * C;
Rsum = Rsum + d_C*eye(n_channels);

[EVecsum, EValsum] = eig(Rsum);
%[EValsum, ind] = sort(diag(EValsum),'descend');
%EVecsum = EVecsum(:,ind);

% Find the transpose of the Whitening Transformation Matrix
W_T = sqrt(inv((EValsum))) * EVecsum';

% Whiten Data Using Whiting Transform
for k = 1:n_classes
    S{k} = W_T * R{k} * W_T';
end

%generalized eigenvectors/values
[B, D] = eig(S{1},S{2});
% Simultanous diagonalization
% Should be equivalent to [B,D]=eig(S{1});

%verify algorithim
%disp('test1:Psi{1}+Psi{2}=I')
%Psi{1}+Psi{2}

%sort
[D, ind]=sort(diag(D), 'descend');
B = B(:,ind);

%Resulting Projection Matrix-these are the spatial filter coefficients
% result = (W*B)'
result = B'*W_T;

% resort CSP coefficients
dimm = n_classes*csp_dim;

[m n] = size(result);

%check for valid dimensions
if(m<dimm)
    disp('Cannot reduce to a higher dimensional space!');
    return
end

%instantiate filter matrix

csp_coeff = zeros(dimm,n);

% create the n-dimensional filter by sorting
% each row is a filter
i=0;
for d = 1:dimm
    
    if(mod(d,2)==0)
        csp_coeff(d,:) = result(m-i,:);
        i=i+1;
    else
        csp_coeff(d,:) = result(1+i,:);
    end
    
end
