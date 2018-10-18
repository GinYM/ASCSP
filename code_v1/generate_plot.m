close all

csp_per_class = 3;

SUBS_NAM = {'S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216'};
freqsRange = [[1, 3]; [2, 5]; [4, 7];[6, 10]; [7, 12]; [10, 15]; [12, 19]; [18, 25]; [19, 30]; [25, 35]; [30, 40]];

load('CSP_covariance_matrix_new.mat');

for sub_idx = 2
      
freqs_idx=5;
name = ['/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' SUBS_NAM{sub_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
load(name);

num_tmp = 162;

load('CSP_covariance_matrix_new.mat')

data_source = cell(1,2); %1 left 2 right

for i = 1:num_tmp
    data_source{1}{i} = prepData.G_l{i};
    data_source{2}{i} = prepData.G_r{i};
end


for target_idx= 5 %[1:sub_idx-1 sub_idx+1:6]
name = ['/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' SUBS_NAM{target_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
load(name)
data_target = cell(1,2);

for target_trial = 1:num_tmp
    data_target{1}{target_trial} = prepData.G_l{target_trial};
    data_target{2}{target_trial} = prepData.G_r{target_trial};
end


gm = cell(1,2); %cm 1 right 2 left
gm{2} = reshape(cm(sub_idx,freqs_idx,1,:,:),[47,47]);
gm{1} = reshape(cm(sub_idx,freqs_idx,2,:,:),[47,47]);

[gm,store_idx] = update_v1(gm,data_source,data_target);

%% CSP new gm source
[ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
[ data_source_filter ] = csp_filtering(data_source, csp_coeff);

data_source_log{1} = log_norm_BP(data_source_filter{1}); 
data_source_log{2} = log_norm_BP(data_source_filter{2});



%% CSP new gm target

 data_target_remove = cell(1,2);
 for i = 1:num_tmp
     data_target_remove{1}{i}=prepData.G_l{i};
     data_target_remove{2}{i}=prepData.G_r{i};
 end
 
%[ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
[ data_target_filter ] = csp_filtering(data_target_remove, csp_coeff);

%figure
data_target_log{1} = log_norm_BP(data_target_filter{1}); 
data_target_log{2} = log_norm_BP(data_target_filter{2});


%% LDA
trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
[W B class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
[X_LDA predicted_y_class1] = lda_apply(cell2mat(data_target_log{1})', W, B);
predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
predicted_y_class1(predicted_y_class1 == -1) = 1;
[X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1! 
predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
temp = [predicted_y_class1; predicted_y_class2];
acc1 = sum(temp)/length(temp);   % this is the percent correct classification 
disp(['Acc: ' num2str(acc1)])

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

% figure
% plot(1:num_tmp,X(1:num_tmp,1),'r*')
% hold on
% plot(1:num_tmp,X(num_tmp+1:num_tmp*2,1),'k*')
% hold on
% plot(num_tmp+1:num_tmp*2,X(325:num_tmp*3,1),'bo')
% hold on
% plot(num_tmp+1:num_tmp*2,X(num_tmp*3+1:end,1),'go')
% figure
% plot(1:num_tmp,Xproj(1:num_tmp,1),'r*');
% hold on
% plot(1:num_tmp,Xproj(num_tmp+1:num_tmp*2,1),'k*')
% hold on
% plot(num_tmp+1:num_tmp*2,Xproj(num_tmp*2+1:num_tmp*3,1),'bo')
% hold on
% plot(num_tmp+1:num_tmp*2,Xproj(num_tmp*3+1:num_tmp*4,1),'go');


%subplot(r,c,4)
%draw1(Xproj,Y,domainFt,{'z_1','z_2'},'SA',acc)


%% LDA train
trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
%size(trainY)
%size(Xproj)
[W,B,class_means] = lda_train_reg(Xproj(1:324,:), trainY, 0);


%% LDA
[X_LDA predicted_y_class1] = lda_apply(Xproj(325:486,:), W, B);
predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
predicted_y_class1(predicted_y_class1 == -1) = 1;
[X_LDA predicted_y_class2] = lda_apply(Xproj(487:648,:), W, B);    % there is a vector output! should all be -1! 
predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
temp = [predicted_y_class1; predicted_y_class2];
acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
disp(['Acc: ' num2str(acc2)])

save store_plot_2_5 X Xproj

%fprintf(fileID,'Test ID %d ASACSP: %f  CSP: %f\n',target_idx,acc2,acc1);
end
%fprintf(fileID,'\n');
end