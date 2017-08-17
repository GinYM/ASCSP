function result = update_gm(gm,data_source,data_target)
csp_per_class=3;
result=gm;
n = length(data_source{1});
acc = 0;
mean_pre = 0;
mean_delta = 0;
record =  zeros(1,n);
for label = 1:2
for trial_idx = 1:length(data_target{1})
    
    %C_new = cov(data_target{label}{trial_idx}')/trace(cov(data_target{label}{trial_idx}'));
    
    %% type 1
%     result_tmp_1 = result;
%     result_tmp_1{1} = 1/(n+1)*C_new+n/(n+1)*result{1}; 
    %result_tmp_1 = 1/(n+1)*C_new+n/(n+1)*result{1}; 
    %result{1} = 1/(n+1)*C_new+n/(n+1)*result{1}; 
    %% CSP
    [ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,result); % actually don't need data_source
    [ data_source_filter ] = csp_filtering(data_source, csp_coeff);
    [data_target_filter] = csp_filtering(data_target,csp_coeff);
    data_source_log{1} = log_norm_BP(data_source_filter{1}); 
    data_source_log{2} = log_norm_BP(data_source_filter{2});
    data_target_log{1} = log_norm_BP(data_target_filter{1}); 
    data_target_log{2} = log_norm_BP(data_target_filter{2});
    
    %% SA
    X = zeros(0,6);
    Y = zeros(0,1);
    maLabeled = zeros(0,1,'logical');
    for idx = 1:2
    for i = 1:size(data_source_log{idx},2)
        X = [X;data_source_log{idx}{i}'];
        Y = [Y;idx];
        maLabeled = [maLabeled;true];
    end
    end

    for idx = 1:2
    for i = 1:size(data_target_log{1},2)
        X = [X;data_target_log{idx}{i}'];
        Y = [Y;idx];
        maLabeled = [maLabeled;false];
    end
    end
    param = []; param.pcaCoef = 2;
    [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
    mean1 = mean(X(1:n*2,:));
    mean2 = mean(X(2*n+1:end,:));
    mean_total_1 = mean(abs(mean1-mean2));
    if mean_pre ~= 0 && mean_total_1-mean_pre > mean_delta
        continue;
    end
    dif1 = sum(var(Xproj(1:n*2,:)));
    dif2 = sum(var(Xproj(2*n+1:end,:)));
    dif_total_1 = abs(dif1-dif2);
    if mod(trial_idx,50) == 0
        dif1 = sum(var(Xproj(1:n*2,:)));
        dif2 = sum(var(Xproj(2*n+1:end,:)));
%         disp(['Dif1: ' num2str(dif1) ' Dif2: ' num2str(dif2)]);
%         disp(['Mean1: ' num2str(mean(mean1)) ' Mean2: ' num2str(mean(mean2))]);
        figure
        title(['Trial ' num2str(trial_idx)]);
        plot(1:n,Xproj(1:n,1),'r*');
        hold on
        plot(1:n,Xproj(n+1:n*2,1),'k*')
        hold on
        plot(n+1:n*2,Xproj(n*2+1:n*3,1),'bo')
        hold on
        plot(n+1:n*2,Xproj(n*3+1:n*4,1),'go');
        
    end
    
%     %% type 2
%     result_tmp_2 = result;
%     result_tmp_2{2} = 1/(n+1)*C_new+n/(n+1)*result{2}; 
%     %result{2} = 1/(n+1)*C_new+n/(n+1)*result{1}; 
%     %% CSP
%     [ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,result_tmp_2); % actually don't need data_source
%     [ data_source_filter ] = csp_filtering(data_source, csp_coeff);
%     [data_target_filter] = csp_filtering(data_target,csp_coeff);
%     data_source_log{1} = log_norm_BP(data_source_filter{1}); 
%     data_source_log{2} = log_norm_BP(data_source_filter{2});
%     data_target_log{1} = log_norm_BP(data_target_filter{1}); 
%     data_target_log{2} = log_norm_BP(data_target_filter{2});
%     
%     %% SA
%     X = zeros(0,6);
%     Y = zeros(0,1);
%     maLabeled = zeros(0,1,'logical');
%     for idx = 1:2
%     for i = 1:size(data_source_log{idx},2)
%         X = [X;data_source_log{idx}{i}'];
%         Y = [Y;idx];
%         maLabeled = [maLabeled;true];
%     end
%     end
% 
%     for idx = 1:2
%     for i = 1:size(data_target_log{1},2)
%         X = [X;data_target_log{idx}{i}'];
%         Y = [Y;idx];
%         maLabeled = [maLabeled;false];
%     end
%     end
%     param = []; param.pcaCoef = 2;
%     [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
%     mean1 = mean(X(1:n*2,:));
%     mean2 = mean(X(2*n+1:end,:));
%     mean_total_2 = mean(abs(mean1-mean2));
%     dif1 = sum(var(Xproj(1:n*2,:)));
%     dif2 = sum(var(Xproj(2*n+1:end,:)));
%     dif_total_2 = abs(dif1-dif2);
%     if mod(trial_idx,50) == 0
%         dif1 = sum(var(Xproj(1:n*2,:)));
%         dif2 = sum(var(Xproj(2*n+1:end,:)));
%         disp(['Dif1: ' num2str(dif1) ' Dif2: ' num2str(dif2)]);
%         disp(['Mean1: ' num2str(mean(mean1)) ' Mean2: ' num2str(mean(mean2))]);
%         figure
%         title(['Trial ' num2str(trial_idx)]);
%         plot(1:n,Xproj(1:n,1),'r*');
%         hold on
%         plot(1:n,Xproj(n+1:n*2,1),'k*')
%         hold on
%         plot(n+1:n*2,Xproj(n*2+1:n*3,1),'bo')
%         hold on
%         plot(n+1:n*2,Xproj(n*3+1:n*4,1),'go');
%         
%     end
%     
%     if dif_total_1 + mean_total_1 < dif_total_2 + mean_total_2
%         if label == 1
%             acc = acc+1;
%         end
%         result = result_tmp_1;
%     else
%         
%         if label == 2
%             acc = acc+1;
%         end
%         result = result_tmp_2;
%     end
        
    
    Xproj_1 = mean(Xproj(1:162,:));
    Xproj_2 = mean(Xproj(163:162*2,:));
    dist_1 = sqrt(sum((Xproj_1-Xproj(trial_idx,:)).^2));
    dist_2 = sqrt(sum((Xproj_2-Xproj(trial_idx,:)).^2));
    %disp([num2str(dist_1) ' ' num2str(dist_2)])
    theta1 = 1-(dist_1/(dist_1+dist_2));
    C_new = cov(data_target{label}{trial_idx}')/trace(cov(data_target{label}{trial_idx}'));
    result{1} = theta1/(n+1)*C_new+n/(n+1)*result{1};
    result{2} = (1-theta1)/(n+1)*C_new+n/(n+1)*result{2};
    
    if (sign(dist_1-dist_2)+3)/2 == label
        acc = acc+1;
    end
    
end
end
disp(['Acc: ' num2str(acc/2/n)]);
save tmp_filter data_source_filter data_target_filter