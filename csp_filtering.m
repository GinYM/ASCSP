% CSP filtering on given trials

function [ output_classes ] = csp_filtering(input_classes, csp_coeff)

% extract useful parameters
n_classes = length(input_classes);

% if csp_type == 1: specCSP
if(~isstruct(csp_coeff))
    csp_type = 0;
else
    csp_type = 1;
end

[ n_filters, n_channels] = size(csp_coeff);

if(csp_type == 0)
    
    for class = 1:n_classes
        N = length(input_classes{class});
        for trial = 1:N            
            output_temp = csp_coeff*input_classes{class}{trial};
            output_classes{class}{trial} = output_temp;
        end
    end

% specCSP
elseif(csp_type == 1)
    
    % convert to matrix format
    for class = 1:n_classes
        input_data{class} = cell_to_mat(input_classes{class});
    end
    
    n_samples = length(input_data{1}(1,:,1));
    
    % specCSP filtering on input data
    for class = 1:n_classes
        % filtered = ml_filterspeccsp(data, patterns, flt, wndofs, wndlen, logvar)
        output_data{class} = ml_filterspeccsp_custom(input_data{class}, csp_coeff.patterns, csp_coeff.flt, 0, n_samples, 0);
    end
    
        % convert to matrix format
    for class = 1:n_classes
        output_classes{class} = mat_to_cell(output_data{class});
    end
    
end