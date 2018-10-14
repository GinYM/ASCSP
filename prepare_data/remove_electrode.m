function remove_electrode

% remove specific electrode for transfer learning
addpath /media/gin/hacker/UCSD_Summer_Research/eeglab14_1_1b/
addpath /media/gin/hacker/UCSD_Summer_Research/eeglab14_1_1b/functions/popfunc
addpath /media/gin/hacker/UCSD_Summer_Research/code/Yiming_Jin/bcitools/
addpath /media/gin/hacker/UCSD_Summer_Research/eeglab14_1_1b/functions/adminfunc

nams = {
    %'BP-220416-2-shams-ica-pruned.set';
    'BP-240416-1-shams-ica-pruned.set';
    'BP-240416-2-shams-ica-pruned-V3.set';
    'BP-240416-3-shams-ica-pruned.set';
    'BP-270416-2-shams-ica-pruned.set';
    'BP-130516-1-shams-ica-pruned.set';
    'BP-141216-shams-ica-pruned-V2.set';
    'BP-191216-shams-ica-pruned-V3.set';
    'BP-010117-shams-ica-pruned.set';
    'BP-011217-shams-ica-pruned.set';
    };
filepath = '/media/gin/hacker/UCSD_Summer_Research/code/Yiming_Jin/data';
chan_loc = cell(length(nams),1);
for i = 1:length(nams)
    filename = nams{i};
    EEG = pop_loadset(filename, filepath);
    tmp = cell(size(EEG.chanlocs,2),1);
    for idx = 1:size(EEG.chanlocs,2)
        tmp{idx} = EEG.chanlocs(idx).labels;
    end
    chan_loc{i} = tmp;      
end



save chan_loc chan_loc
load chan_loc
max_idx = 0;
max_len = 0;
for i = 1:size(chan_loc,1)
    if size(chan_loc{i}) > max_idx
        max_len =size(chan_loc{i});
        max_idx = i;
    end
end
max_loc = chan_loc{max_idx};
count = 0;
for i = 1:size(max_loc,1)
    target = max_loc{i};
    all_result = 1;
    for idx1 = 1:size(chan_loc,1)
        result = 0;
        for idx2=1:size(chan_loc{idx1})
            if strcmp(chan_loc{idx1}{idx2},target)
                result = 1;
                break;
            end
        end
        if result == 0
            all_result = 0;
            break;
        end
    end
    if all_result
        count = count+1;
        crop_loc{count} = target;
        disp(target);
    end
end
size(crop_loc)

crop_idx = zeros(length(nams),size(crop_loc,2));
EEG = pop_loadset(nams{1}, filepath);
for subj_idx = 1:size(chan_loc,1)
    count = 1;
    for loc_idx = 1:size(chan_loc{subj_idx},1)
        if count > size(crop_loc,2)
            break
        end
        if strcmp(crop_loc{count},chan_loc{subj_idx}{loc_idx})
            crop_idx(subj_idx,count) = loc_idx;
            count = count+1;
        end
    end
end

%size(crop_loc,1)
elec_loc = zeros(size(crop_loc,2),3);
size(elec_loc)
for i = 1:size(elec_loc,1)
    x = EEG.chanlocs(crop_idx(1,i)).X;
    y = EEG.chanlocs(crop_idx(1,i)).Y;
    z = EEG.chanlocs(crop_idx(1,i)).Z;
    disp([x,y,z])
    elec_loc(i,1) = x;
    elec_loc(i,2) = y;
    elec_loc(i,3) = z;
end
save elec_loc elec_loc


%SUBS_NAM = {'S_BP-130516-1','S_BP-220416-2','S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2'};
SUBS_NAM = {'S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216'};
freqsRange = [[1, 3]; [2, 5]; [4, 7];[6, 10]; [7, 12]; [10, 15]; [12, 19]; [18, 25]; [19, 30]; [25, 35]; [30, 40]];
%crop_size = 59;
path_prefix = '/media/gin/hacker/UCSD_Summer_Research/code/Yiming_Jin/data-proc/';
for sub_idx = 1:size(SUBS_NAM,2)
    for freqs_idx = 1:size(freqsRange,1)
        
        name = [SUBS_NAM{sub_idx},'freqs',num2str(freqsRange(freqs_idx,1)),'_',num2str(freqsRange(freqs_idx,2)),'_','shams_FP.mat'];
        
        disp(name)
        load([path_prefix,name]);
        G_r = prepData.G_r;
        B_r = prepData.B_r;
        G_l = prepData.G_l;
        B_l = prepData.B_l;
        name_GB = {'G_r','B_r','G_l','B_l'};
        
        for GB_idx = 1:4
            %eval(['tmp=prepData.' name_GB ';'])
            tmp = prepData.(name_GB{GB_idx});
            for i = 1:size(tmp,2)
                %size(tmp{i})
                tmp1 = tmp{i}(crop_idx(sub_idx,:),:);
                %size(tmp1)
                prepData.(name_GB{GB_idx}){i} = tmp1;
                %eval(['prepData.' name_GB '{' num2str(i) '}=tmp1;'])
                %G_r{i} = tmp;
            end
        end
        output_name = ['../data/output_new1/',name];
        disp(output_name)
        eval(['save ' output_name ' prepData'])
        %save output_name prepData
    end
end
            
    