% calculate the normalized log power of given inputs
%
% 

function [ log_BP ] = log_norm_BP(input_data)

% if input_data has matrix format change to cell format
if(~iscell(input_data))
    
    input_data = mat_to_cell(input_data);
end


n_trials = length(input_data);

log_BP = cell(1,n_trials);

for trial = 1:n_trials
    % squared sum/num of samples for each channel(component)
    temp_value = var(input_data{trial}',1)';
    
    % normalize
    norm_temp_value = temp_value/sum(temp_value);
    
    log_BP{trial} = log(norm_temp_value);
end