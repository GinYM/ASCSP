% perform 2-class CSP on input data
%
% INPUTS
% input_data: Data divided into classes by cell array or matrix form
%                each cell contains - N cells with CxT matrices
%                each matrix is size CxTxN
%                (C:# channels, T: # samples, N: # trials)
% varagin: various parameters for CSP
%   | type | csp_dim |...
%       type: 0 -- Non-regularized(default)
%       csp_dim: number of CSP filters per class
%       BY CSP TYPE:
%             1 -- L1 regularized(sparse CSP)
%                  | rho |
%             2 -- L2 regularized
%                  | C |
%                  C: regularizing coefficient(deafult: 0)
%             3 -- spatially regularized(SRCSP)
%                  SRCSP does not output all CSP coefficients
%                  SRCSP needs additional inputs
%                  | r | alpha | chanlocs |
%             4 -- specCSP(uses functions from bcilab)
%                  specCSP does not output all CSP coefficients
%                  specCSP needs additional inputs
%                  | srate | priorfilter | chanlocs | wndofs |
%
%
% OUTPUTS
% csp_coeff: result of CSP analysis(output varies by csp type)
%            (n_csp by n_channel) matrix
%            for filtering do: csp_coeff*X
%
% 11/02/11: implement R-csp

function [ csp_coeff all_coeff] = csp_analysis(input_data, varargin)


if(nargin < 1)
    disp('Error: Incorrect number of input variables.')
    return
end

% extract useful values
n_classes = length(input_data);

% if input_data has matrix format change to cell format
if(~isstruct(input_data))
    if(~iscell(input_data{1}))
        input_classes = cell(1,n_classes);
        for class = 1:n_classes
            input_classes{class} = mat_to_cell(input_data{class});
        end
    else
        input_classes = input_data;
    end
    

    
%     if (length(input_classes) ~= 2)
%         disp('Must have 2 classes for CSP!')
%         return
%     end
    


n_classes = length(input_classes);

end

if( nargin == 1 )
    csp_type = 0;
    csp_dim = size(input_classes{1}{1}(:,1));
elseif( nargin == 2)
    csp_type = varargin{1};
    csp_dim = size(input_classes{1}{1}(:,1));
else
    csp_type = varargin{1};
    csp_dim = varargin{2};
end

% non-regularized CSP
if( csp_type == 0 )
        % get parameters
    if(nargin < 4)
        C = 0;
    elseif(nargin == 4)
        C = varargin{3};
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
        
    [ csp_coeff all_coeff] = csp_L2_reg_csp_norm_cov(input_classes, csp_dim, C);
    
    % each column of the matrix is a filter
    % for visulization of patterns: A = inv(all_coeff)'
    all_coeff = all_coeff';

% L1-regularized CSP(sparse CSP)
elseif( csp_type == 1)
    
    % get parameters
    if(nargin < 4)
        rho = 0;
    elseif(nargin == 4)
        rho = varargin{3};
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
    
    [ csp_coeff ] = csp_sparse_csp(input_classes, csp_dim, rho);
    all_coeff = [];
% L2-regularized CSP
elseif( csp_type == 2 )
    
    % get parameters
    if(nargin < 4)
        C = 0;
    elseif(nargin == 4)
        C = varargin{3};
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
        
    [ csp_coeff all_coeff] = csp_L2_reg_csp(input_classes, csp_dim, C);
    
    % each column of the matrix is a filter
    % for visulization of patterns: A = inv(all_coeff)'
    all_coeff = all_coeff';

% spatially regularized CSP
elseif( csp_type == 3 )

    if(nargin == 6)
        r = varargin{3};
        alpha = varargin{4};
        chanlocs = varargin{5};
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
    
    [ csp_coeff all_coeff] = csp_srcsp(input_data, csp_dim, r, alpha, chanlocs);
%     all_coeff = all_coeff';
% specCSP
elseif( csp_type == 4 )
    
    % convert to matrix format
    for class = 1:n_classes
        input_data{class} = cell_to_mat(input_classes{class});
    end
    
    % get parameters
    [ n_channels n_samples n_trials] = size(input_data{1});
    
    if(nargin == 7)
        srate = varargin{3};
        priorfilter = varargin{4};
        chanlocs = varargin{5};
        wndofs = varargin{6};
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
    
    % [patterns flt] = ml_trainspeccsp(trials,srate,nof,p,q,wndofs,wndlen,chanlocs,priorfilter,numsteps)
    [patterns flt] = ml_trainspeccsp_custom(input_data, srate, csp_dim, 0, 1, wndofs, n_samples, chanlocs, priorfilter);
    csp_coeff.patterns = patterns;
    csp_coeff.flt = flt;
    all_coeff = [];
elseif(csp_type == 5)
    
    if(nargin == 5)
        user = varargin{3};
        beta_gamma = varargin{4};
        
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
    
    [ csp_coeff all_coeff ] = csp_aggcsp(input_data, csp_dim, user, beta_gamma);
    % non-regularized CSP
elseif( csp_type == 6 )
        % get parameters
    if(nargin < 4)
        C = 0;
    elseif(nargin == 4)
        C = varargin{3};
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
        
%     [ csp_coeff all_coeff] = csp_MM(input_classes, csp_dim, C);

    [ csp_coeff all_coeff] = csp_L2_reg_csp_norm_cov_rank(input_classes, csp_dim, C);
    
    % each column of the matrix is a filter
    % for visulization of patterns: A = inv(all_coeff)'
    all_coeff = all_coeff';
    
elseif( csp_type == 7 )
        % get parameters
    if(nargin < 4)
        alpha = 0.5;
        beta = 0.5; 
    elseif(nargin > 3)
        alpha = varargin{3};
        beta = varargin{4}; 
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
        
    [ csp_coeff all_coeff] = icsp_MM(input_classes, csp_dim, alpha, beta);
    
    % all_coeff is a cell with both set of coefficients
    
    elseif( csp_type == 8 )
        % get parameters
    if(nargin < 4)
        alpha = 0.5;
        beta = 0.5; 
    elseif(nargin > 3)
        alpha = varargin{3};
        beta = varargin{4}; 
    else
        disp('Error: Incorrect number of input variables.')
        return
    end
        
    [ csp_coeff all_coeff] = icsp_MM_V2(input_classes, csp_dim, alpha, beta);
    
    % all_coeff is a cell with both set of coefficients
elseif( csp_type == 9 )
    % get parameters
    C = varargin{3};
    general_matrix = varargin{4};
    if(nargin ~= 5)
        disp('Error: Incorrect number of input variables.')
        return
    end
        
    [ csp_coeff all_coeff] = csp_L2_reg_csp_norm_cov_general(input_classes, csp_dim, C,general_matrix);
    
    % each column of the matrix is a filter
    % for visulization of patterns: A = inv(all_coeff)'
    all_coeff = all_coeff';    
    
else
    disp('Error: invalid CSP type');
end
    
    