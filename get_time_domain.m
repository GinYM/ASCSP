% this file will perform classification over GvB and RvL classes for one
% back and two back groups 

% Mahta: Nov. 2015 

%% addpaths

clc
clear 

close all

addpath /media/gin/hacker/UCSD_Summer_Research/eeglab14_1_1b


%addpath ~/Matlab/RvL_sham_bars/bcitools_Mahta/

%addpath ~/Matlab/RvL_sham_bars/bcitools/csp/
%addpath ~/Matlab/RvL_sham_bars/bcitools/

%addpath ~/Matlab/sham_feedback_paradigm/toolsBP/

path = '/media/gin/hacker/UCSD_Summer_Research/motor_imagery/data/';

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB under Matlab

% dirs = {
%        '~/Matlab/sham_feedback_paradigm/data/BP-220416-2';
%        '~/Matlab/sham_feedback_paradigm/data/BP-240416-1';
%        '~/Matlab/sham_feedback_paradigm/data/BP-240416-2';
%        '~/Matlab/sham_feedback_paradigm/data/BP-240416-3';
%        '~/Matlab/sham_feedback_paradigm/data/BP-270416-2';
%        '~/Matlab/sham_feedback_paradigm/data/BP-130516-1';
%        '~/Matlab/sham_feedback_paradigm/data/BP-141216';
%        '~/Matlab/sham_feedback_paradigm/data/BP-191216';
%        '~/Matlab/sham_feedback_paradigm/data/BP-010117';
%        '~/Matlab/sham_feedback_paradigm/data/BP-011217';
%         };


nams = {
    'BP-130516-1-shams-ica-pruned-0.1.set';
    'BP-220416-2-shams-ica-pruned-0.1.set';
    'BP-240416-1-shams-ica-pruned-0.1.set';
    'BP-240416-2-shams-ica-pruned-V3-0.1.set';
    'BP-240416-3-shams-ica-pruned-0.1.set';
    'BP-270416-2-shams-ica-pruned-0.1.set';
    };
%% extract data 

close all 

IDs = {'BP-220416-2','BP-240416-1','BP-240416-2','BP-240416-3','BP-270416-2','BP-130516-1','BP-141216','BP-191216','BP-010117','BP-011217'}; 


kF = 10; 
CV_num = 3; 

freqsCell = {[1 3],[2 5], [4 7],[6 10], [7 12], [10 15], [12 19], [18 25], [19 30], [25 35], [30 40]};

channels = {'ALL'};
csp_per_class = 3;


% class_rate_allFreq_vanilla = zeros(length(IDs),2);  

for i = 1:6%10%1:length(IDs)
    
    filepath = path ;%[path 'output/' nams{i}];

    filename = [nams{i}];
    EEG = pop_loadset(filename, filepath);
    

    

    %[class_rate_allFreq_vanilla, rate, selectedFreqs] = classification_infoFreq_combined_vanilla_BP(EEG, IDs{i},freqsCell, kF, CV_num, channels, csp_per_class)

%     save(['~/Matlab/sham_feedback_paradigm/res-class/S_',IDs{i},'_infoFreq_vanilla_Kaiser.mat'],...
%         'class_rate_allFreq_vanilla','rate', 'selectedFreqs')
    
 % save(['~/Matlab/sham_feedback_paradigm/res-class/S_',IDs{i},'_infoFreq_vanilla_FP.mat'],...
 %       'class_rate_allFreq_vanilla','rate', 'selectedFreqs')
%     

end        

