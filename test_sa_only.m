
close all
csp_per_class = 3;

SUBS_NAM = {'S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216'};
freqsRange = [[1, 3]; [2, 5]; [4, 7];[6, 10]; [7, 12]; [10, 15]; [12, 19]; [18, 25]; [19, 30]; [25, 35]; [30, 40]];
sub_idx=1;
%target_idx = 6;
freqs_idx=1;
name = ['/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' SUBS_NAM{sub_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
load(name);

num_tmp = 162;
data_l = zeros(num_tmp,2);  % cell(2,1);
data_r = zeros(num_tmp,2); %cell(2,1);

load('CSP_covariance_matrix_new.mat')
data_label = zeros(num_tmp*2,1);
dim = 1;
data_source = cell(1,2); %1 left 2 right
gm = cell(1,2); %cm 1 right 2 left
gm{1} = zeros(47,47);
gm{2} = zeros(47,47);
for i = 1:num_tmp
    data_source{1}{i} = prepData.G_l{i};
    data_source{2}{i} = prepData.G_r{i};
    gm{1} = gm{1}+cov(prepData.G_l{i}')/trace(cov(prepData.G_l{i}'));
    gm{2} = gm{2}+cov(prepData.G_r{i}')/trace(cov(prepData.G_r{i}'));
end
gm{1} = gm{1}/num_tmp;
gm{2} = gm{2}/num_tmp;
save g1 gm
%figure
%plot_eigen(data_source);

[ csp_coeff,all_coeff] = csp_analysis(data_source,6,csp_per_class, 0);
[ data_source_filter ] = csp_filtering(data_source, csp_coeff);

%figure
data_source_log{1} = log_norm_BP(data_source_filter{1}); 
data_source_log{2} = log_norm_BP(data_source_filter{2});
%figure
%plot_power(data_source_log);
%ylim([-4.5,0])

%% LDA train
trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
[W B class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);

%save tmp_power data_source_log
%plot_eigen(data_source_filter);

load('CSP_covariance_matrix_new.mat');
%C1 = reshape(cm(sub_idx,1,1,:,:),[47,47]);

for target_idx=5:5
name = ['/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' SUBS_NAM{target_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
load(name)
data_target = cell(1,2);

for target_trial = 1:num_tmp
    data_target{1}{target_trial} = prepData.G_l{target_trial};
    data_target{2}{target_trial} = prepData.G_r{target_trial};
end

[data_target_filter] = csp_filtering(data_target,csp_coeff);
data_target_log{1} = log_norm_BP(data_target_filter{1}); 
data_target_log{2} = log_norm_BP(data_target_filter{2});
%figure
%plot_power(data_target_log);
%ylim([-4.5,0])
%plot_eigen(data_target_filter);

%% LDA
[X_LDA predicted_y_class1] = lda_apply(cell2mat(data_target_log{1})', W, B);
predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
predicted_y_class1(predicted_y_class1 == -1) = 1;
[X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1! 
predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
temp = [predicted_y_class1; predicted_y_class2];
acc = sum(temp)/length(temp);   % this is the percent correct classification 
disp(['Acc: ' num2str(acc)])
end

count_source = num_tmp;
gm = cell(1,2); %cm 1 right 2 left
gm{2} = reshape(cm(sub_idx,1,1,:,:),[47,47]);
gm{1} = reshape(cm(sub_idx,1,2,:,:),[47,47]);

% gm_new = zeros(47,47);
% 
% for  i= 1:140
%     
%     if mod(i,2) == 0 
%         gm{1} = gm{1}*num_tmp*2/(num_tmp*2+1)+1/(num_tmp*2+1)*cov(data_target{1}{mod(i-1,num_tmp*2)+1}')/trace(cov(data_target{1}{mod(i-1,num_tmp*2)+1}'));
%     else
%         gm{2} = gm{2}*num_tmp*2/(num_tmp*2+1)+1/(num_tmp*2+1)*cov(data_target{2}{mod(i-1,num_tmp*2)+1}')/trace(cov(data_target{2}{mod(i-1,num_tmp*2)+1}'));
%     end
% end



% gm_new = gm_new/2/num_tmp;
% gm{1} = (gm{1} + gm_new)/2;
% gm{2} = (gm{2}+gm_new)/2;

for i = 1:20%num_tmp
        gm{1} = gm{1}*num_tmp*2/(num_tmp*2+1)+1/(num_tmp*2+1)*cov(data_target{1}{i}')/trace(cov(data_target{1}{i}'));
        gm{2} = gm{2}*num_tmp*2/(num_tmp*2+1)+1/(num_tmp*2+1)*cov(data_target{2}{i+50}')/trace(cov(data_target{2}{i+50}'));

end

gm{2} = gm{2}*num_tmp*2/(num_tmp*2+1)+1/(num_tmp*2+1)*cov(data_target{2}{i+50}')/trace(cov(data_target{2}{i+50}'));
% for i = 90:90 %num_tmp
%         gm{1} = gm{1}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{1}{i}')/trace(cov(data_target{1}{i}'));
%         gm{2} = gm{2}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{2}{i}')/trace(cov(data_target{2}{i}'));
% 
% end
% % 
% for i = 36:40
%     gm{2} = gm{2}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{1}{i}')/trace(cov(data_target{1}{i}'));
% end

% for i = 101:101
%     %if mod(i,2) == 0
%         gm{1} = gm{1}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{1}{i}')/trace(cov(data_target{1}{i}'));
%         gm{2} = gm{2}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{2}{i}')/trace(cov(data_target{2}{i}'));
% %     else
% %         gm{1} = gm{1}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{2}{i}')/trace(cov(data_target{2}{i}'));
% %         gm{2} = gm{2}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{1}{i}')/trace(cov(data_target{1}{i}'));
%     %end
% %  gm{1} = gm{1}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{1}{i}')/trace(cov(data_target{1}{i}'));
% %  gm{2} = gm{2}*num_tmp/(num_tmp+1)+1/(num_tmp+1)*cov(data_target{2}{i}')/trace(cov(data_target{2}{i}'));
% end

%gm = update_gm(gm,data_source,data_target);

%% CSP new gm source
[ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
[ data_source_filter ] = csp_filtering(data_source, csp_coeff);

%figure
data_source_log{1} = log_norm_BP(data_source_filter{1}); 
data_source_log{2} = log_norm_BP(data_source_filter{2});
%size(data_source_log{1}{1});
%figure
%save power data_target_log
%plot_power(data_source_log);

%% LDA train
trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
[W B class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);

%% LDA test
[X_LDA predicted_y_class1] = lda_apply(cell2mat(data_source_log{1})', W, B);
predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
predicted_y_class1(predicted_y_class1 == -1) = 1;
[X_LDA predicted_y_class2] = lda_apply(cell2mat(data_source_log{2})', W, B);    % there is a vector output! should all be -1! 
predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
temp = [predicted_y_class1; predicted_y_class2];
acc = sum(temp)/length(temp);   % this is the percent correct classification 
disp(['Acc: ' num2str(acc)])




%% CSP new gm target

 data_target_remove = cell(1,2);
 for i = 1:num_tmp
     data_target_remove{1}{i}=prepData.G_l{i};
     data_target_remove{2}{i}=prepData.G_r{i};
 end
 
[ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
[ data_target_filter ] = csp_filtering(data_target_remove, csp_coeff);

%figure
data_target_log{1} = log_norm_BP(data_target_filter{1}); 
data_target_log{2} = log_norm_BP(data_target_filter{2});
%size(data_target_log{1}{1})
%figure
%save power data_target_log
%plot_power(data_target_log);
%ylim([-4.5,0])


%% LDA
[X_LDA predicted_y_class1] = lda_apply(cell2mat(data_target_log{1})', W, B);
predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
predicted_y_class1(predicted_y_class1 == -1) = 1;
[X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1! 
predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
temp = [predicted_y_class1; predicted_y_class2];
acc = sum(temp)/length(temp);   % this is the percent correct classification 
disp(['Acc: ' num2str(acc)])

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
m1 = mean(X(1:num_tmp*2,:));
m1_avg = sum(abs(m1));
m2 = mean(X(num_tmp*2+1:end,:));
m2_avg = sum(abs(m2));
disp(['Mean1: ' num2str(m1_avg) ' Mean2: ' num2str(m2_avg)])
dif1 = sum(var(Xproj(1:num_tmp*2,:)));
dif2 = sum(var(Xproj(2*num_tmp+1:end,:)));
disp(['Dif1: ' num2str(dif1) ' Dif2: ' num2str(dif2)])
% figure
% plot(X(1:num_tmp,2),X(1:num_tmp,1),'r*')
% hold on
% plot(X(num_tmp+1:num_tmp*2,2),X(num_tmp+1:num_tmp*2,1),'k*')
% hold on
% plot(X(325:num_tmp*3,2),X(325:num_tmp*3,1),'bo')
% hold on
% plot(X(num_tmp*3+1:end,2),X(num_tmp*3+1:end,1),'go')
% figure
% plot(Xproj(1:num_tmp,2),Xproj(1:num_tmp,1),'r*');
% hold on
% plot(Xproj(num_tmp+1:num_tmp*2,2),Xproj(num_tmp+1:num_tmp*2,1),'k*')
% hold on
% plot(Xproj(num_tmp*2+1:num_tmp*3,2),Xproj(num_tmp*2+1:num_tmp*3,1),'bo')
% hold on
% plot(Xproj(num_tmp*3+1:num_tmp*4,2),Xproj(num_tmp*3+1:num_tmp*4,1),'go');



figure
plot(1:num_tmp,X(1:num_tmp,1),'r*')
hold on
plot(1:num_tmp,X(num_tmp+1:num_tmp*2,1),'k*')
hold on
plot(num_tmp+1:num_tmp*2,X(325:num_tmp*3,1),'bo')
hold on
plot(num_tmp+1:num_tmp*2,X(num_tmp*3+1:end,1),'go')
figure
plot(1:num_tmp,Xproj(1:num_tmp,1),'r*');
hold on
plot(1:num_tmp,Xproj(num_tmp+1:num_tmp*2,1),'k*')
hold on
plot(num_tmp+1:num_tmp*2,Xproj(num_tmp*2+1:num_tmp*3,1),'bo')
hold on
plot(num_tmp+1:num_tmp*2,Xproj(num_tmp*3+1:num_tmp*4,1),'go');
%subplot(r,c,4)
%draw1(Xproj,Y,domainFt,{'z_1','z_2'},'SA',acc)

var1 = mean(var(Xproj(1:num_tmp*2,:)));
var2 = mean(var(Xproj(num_tmp*2:end,:)));
disp(['Var1: ' num2str(var1) ' Var2: ' num2str(var2)]);


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
acc = sum(temp)/length(temp);   % this is the percent correct classification 
disp(['Acc: ' num2str(acc)])