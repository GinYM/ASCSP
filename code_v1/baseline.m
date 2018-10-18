%% baselne method for CSP and CSP with subspace alignment
close all
fileID = fopen('result/result_baseline.txt','w');
csp_per_class = 3;

SUBS_NAM = {'S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216'};
freqsRange = [[1, 3]; [2, 5]; [4, 7];[6, 10]; [7, 12]; [10, 15]; [12, 19]; [18, 25]; [19, 30]; [25, 35]; [30, 40]];
for sub_idx = 1:10
    if sub_idx == 7
        continue
    end
    
    fprintf(fileID,'Train Subj %d\n',sub_idx);
    freqs_idx=5;
    name = ['/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' SUBS_NAM{sub_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
    load(name);

    num_tmp = 162;

    load('CSP_covariance_matrix_new.mat')
    data_label = zeros(num_tmp*2,1);
    dim = 1;
    
    %% prepare source data
    data_source = cell(1,2); %1 left 2 right
    for i = 1:num_tmp
        data_source{1}{i} = prepData.G_l{i};
        data_source{2}{i} = prepData.G_r{i};
    end

    %% apply CSP
    [ csp_coeff,all_coeff] = csp_analysis(data_source,6,csp_per_class, 0);
    [ data_source_filter ] = csp_filtering(data_source, csp_coeff);

    data_source_log{1} = log_norm_BP(data_source_filter{1}); 
    data_source_log{2} = log_norm_BP(data_source_filter{2});

    for target_idx=[1:sub_idx-1 sub_idx+1:6]
        name = ['/media/gin/hacker/UCSD_Summer_Research/data/output_new1/' SUBS_NAM{target_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
        load(name)
        
        %% prepare target data
        data_target = cell(1,2);
        for target_trial = 1:num_tmp
            data_target{1}{target_trial} = prepData.G_l{target_trial};
            data_target{2}{target_trial} = prepData.G_r{target_trial};
        end

        [data_target_filter] = csp_filtering(data_target,csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1}); 
        data_target_log{2} = log_norm_BP(data_target_filter{2});

        %% SA
        X = zeros(0,6);
        Y = zeros(0,1);
        maLabeled = zeros(0,1,'logical');
        for idx = 1:2
        for ii = 1:size(data_source_log{idx},2)
            X = [X;data_source_log{idx}{ii}'];
            Y = [Y;idx];
            maLabeled = [maLabeled;true];
        end
        end

        for idx = 1:2
        for ii = 1:size(data_target_log{1},2)
            X = [X;data_target_log{idx}{ii}'];
            Y = [Y;idx];
            maLabeled = [maLabeled;false];
        end
        end
        param = []; param.pcaCoef = 2;
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);


        %% LDA train without SA (Subspace Alignment)
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W B class_means] = lda_train_reg(X(1:324,:), trainY, 0);

        [X_LDA predicted_y_class1] = lda_apply(X(325:486,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        [X_LDA predicted_y_class2] = lda_apply(X(487:648,:), W, B);    % there is a vector output! should all be -1! 
        predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
        temp = [predicted_y_class1; predicted_y_class2];
        acc1 = sum(temp)/length(temp);   % this is the percent correct classification 
        disp(['Acc: ' num2str(acc1)])


        %% with SA
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W B class_means] = lda_train_reg(Xproj(1:324,:), trainY, 0);

        [X_LDA predicted_y_class1] = lda_apply(Xproj(325:486,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        [X_LDA predicted_y_class2] = lda_apply(Xproj(487:648,:), W, B);    % there is a vector output! should all be -1! 
        predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
        temp = [predicted_y_class1; predicted_y_class2];
        acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
        disp(['Acc: ' num2str(acc2)])


        fprintf(fileID,'Target Subj%d: CSP: %f SACSP: %f\n',target_idx,acc1,acc2);
    end
    fprintf(fileID,'\n');
end