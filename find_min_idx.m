function find_min_idx(data_source,data_target)


for i = 1:size(data_source{1},2)
    tmp_l = eig(cov(data_source{1}{i}')/trace(cov(data_source{1}{i}')));
    tmp_r = eig(cov(prepData.G_r{i}')/trace(cov(prepData.G_r{i}')));
    for dim_idx = 1:dim
        data_target(i,dim_idx)=tmp_l(end-(dim_idx-1));
        data_target(i+num_tmp,dim_idx) = tmp_r(end-(dim_idx-1));
    end
    tmp = (data_train-data_target(i,:)).^2;
    
    
    
    [d,tmp_idx] = min(sum(tmp,2));
    
    
    
    if tmp_idx<=num_tmp
        count=count+1;
    end
    
    tmp = (data_train-data_target(i+num_tmp,:)).^2;
    [d,tmp_idx] = min(sum(tmp,2));
    
    
    if tmp_idx>num_tmp
        count=count+1;
    end
    
    data_target_label(i) = 1;
    data_target_label(i+num_tmp) = 2; 


end
save tmp_result data_train data_target

disp(count/num_tmp/2)
