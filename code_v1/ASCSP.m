% Main file for ASCSP with subspace alignment.
close all
fileID = fopen('result/result_ASACSP_simplify.txt','w');

%extract 3 largest eigen value and 3 smallest.
csp_per_class = 3;

SUBS_NAM = {'S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216'};
freqsRange = [[1, 3]; [2, 5]; [4, 7];[6, 10]; [7, 12]; [10, 15]; [12, 19]; [18, 25]; [19, 30]; [25, 35]; [30, 40]];

%load general matrix to cm
%cm is a 5-D matrix with the size of (6,11,2,47,47)
%where 6 is the subject index, 11 is the 11 frequencies, 2 the two class
%label, 47x47 is the covariance matrix.
%cm stores the averaged normalized covariance in each 6 subject each class.
load('CSP_covariance_matrix.mat');

freqs_idx=5;
%except 7, someting wrong
%training subject index from 1 to 6
for sub_idx = 1:10
    if sub_idx == 7
        continue
    end
    fprintf(fileID,'Tain ID %d\n',sub_idx);

    % Only use the frequency of 7-12
    
    name = ['/Volumes/Seagate Backup Plus Drive/research/data/S',int2str(sub_idx), '_I1_allFreqs_U.mat'];
    load(name);
    

    %number of trials in one subject one class
    num_tmp = size(train_Bl,2);

    %prepare source data
    % left: Gl ang Bl
    data_source = cell(1,2); %1 left 2 right
    for i = 1:num_tmp
        %trial_num = size(train_Gl, 2);
        
        % left data both good and bad with instance 1
        data_source{1}{i} = squeeze(train_Gl(freqs_idx,i,:,:) );
        
        %data_source{1}{i} = [reshape( train_Gl(freqs_idx,i), [] ) ; train_Bl()];
        
        data_source{2}{i} = squeeze(train_Gr(freqs_idx, i,:,:));
        
        %data_source{2}{i} = prepData.G_r{i};
    end
   
    for i = 1:num_tmp
        %trial_num = size(train_Gl, 2);
        
        % left data both good and bad with instance 1
        data_source{1}{i+num_tmp} = squeeze(train_Bl(freqs_idx,i,:,:) );
        
        %data_source{1}{i} = [reshape( train_Gl(freqs_idx,i), [] ) ; train_Bl()];
        
        data_source{2}{i+num_tmp} = squeeze(train_Br(freqs_idx, i,:,:));
        
        %data_source{2}{i} = prepData.G_r{i};
   end

    for target_idx= [1:sub_idx-1 sub_idx+1:10]
        if target_idx == 7
            continue
        end
        name = ['/Volumes/Seagate Backup Plus Drive/research/data/S',int2str(target_idx), '_I1_allFreqs_U.mat'];
        load(name)
        
        num_tmp = size(train_Bl,2);
        
        % store target data in data_target
        data_target = cell(1,2);
        for target_trial = 1:num_tmp
            
            data_target{1}{target_trial} = squeeze(train_Gl(freqs_idx,target_trial,:,:) );
            data_target{2}{target_trial} = squeeze(train_Gr(freqs_idx, target_trial,:,:));
            %data_target{1}{target_trial} = prepData.G_l{target_trial};
            %data_target{2}{target_trial} = prepData.G_r{target_trial};
        end
        
        for target_trial = 1:num_tmp
            
            data_target{1}{target_trial+num_tmp} = squeeze(train_Bl(freqs_idx,target_trial,:,:) );
            data_target{2}{target_trial+num_tmp} = squeeze(train_Br(freqs_idx, target_trial,:,:));
            %data_target{1}{target_trial} = prepData.G_l{target_trial};
            %data_target{2}{target_trial} = prepData.G_r{target_trial};
        end

        % covariance matrix c1, c2 which is initialized as averaged
        % normalized covariance matrix in source subject
        gm = cell(1,2); %cm 1 left 2 right 
        
        %gm{2} = reshape(cm(sub_idx,freqs_idx,1,:,:),[47,47]);
        %gm{1} = reshape(cm(sub_idx,freqs_idx,2,:,:),[47,47]);

        % Structure of cm [Subject_id, instance_id, Frequency, left or right, channel, channel ] 
        % Get instance 1
        gm{2} = squeeze(cm(sub_idx, 1, freqs_idx, 2, :,:));
        gm{1} = squeeze(cm(sub_idx, 1, freqs_idx, 1, :,:));
        
        %update covariance matrix using ASCSP algorithm
        [gm,store_idx] = update_v1_simplify(gm,data_source,data_target);
        
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
        [ data_source_filter ] = csp_filtering(data_source, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1}); 
        data_target_log{2} = log_norm_BP(data_target_filter{2});


        %% train LDA. This is ASCSP without subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W B class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
        [X_LDA predicted_y_class1] = lda_apply(cell2mat(data_target_log{1})', W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        [X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1! 
        predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
        predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
        temp = [predicted_y_class1; predicted_y_class2];
        acc1 = sum(temp)/length(temp);   % this is the percent correct classification 
        disp(['Acc: ' num2str(acc1)])

        %% Subapace alignment 
        X = zeros(0,6);
        Y = zeros(0,1);
        maLabeled =false(0,1);
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

        %% if you want to visualize the features, just uncomment the following codes.
        
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

        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W,B,class_means] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        
        [X_LDA predicted_y_class1] = lda_apply(Xproj(size_xproj+1:size_xproj+left_size/2,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1! 
        predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
        predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
        temp = [predicted_y_class1; predicted_y_class2];
        acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
        disp(['Acc: ' num2str(acc2)])


        fprintf(fileID,'Test ID %d ASCSP: %f  ASCSP SA: %f\n',target_idx,acc1,acc2);
    end
    fprintf(fileID,'\n');
end