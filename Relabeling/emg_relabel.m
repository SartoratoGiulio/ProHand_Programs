close all
clear all
clc

%% Get emg signal

%pathname = "C:\Users\sarto\OneDrive\Desktop\Tesi_Magistrale\DATA\ESP32\Giulio_offset\";
pathname = "D:\Tesi_Magistrale\DATA\Giulio_offset_relabel\day1\csv";
filename = "Giulio_6rep_5sec_angle_3_ESP32_1000Hz_relabel";
data = readmatrix(fullfile(pathname,strcat(filename,".csv")));
relabel_stim = data(:,12);
data = data(:, [1,2,3,4,5,6,7,8,9,10,13]);
%%
Fs = 1000;%Hz
[emg,t,repetition,stimulus,angle] = data2emg(data,Fs);
poses = max(stimulus);
reps = max(repetition);
rest = 3*Fs; 
slide = 8*6*Fs; %time for all the pose repetition
%%
% new_emg=[];
% for i=1:8
%    new_emg = [new_emg rolling_mean(emg(:,i), 50)']; 
% end
% emg = new_emg;
% stimulus = stimulus(1:length(emg));
% repetition = repetition(1:length(emg));
% t = t(1:length(emg));
%% 
total_stimulus_len = 0;
total_emg_len = 0;
total_new_stimulus = [];
used_emg = [];
channels = [0,0,0,0,0,0,0,0,0,0];

minActivationTime = 3000; %default 3000ms

% Relabel all repetition one by one for added accuracy
% Because of that the min number of activation has been reduced to 1 for
% all segments
for pose = 1:poses
    for rep = 1:reps
        if pose == 1 && rep == 1
            first_idx = 1;
            last_idx = find(stimulus==pose & repetition == rep,1,'last')+rest/2;
        elseif pose == poses && rep == reps
            first_idx = last_idx+1;
            last_idx = length(emg);
        else
            first_idx = last_idx+1;
            last_idx = find(stimulus==pose & repetition == rep,1,'last')+rest/2;
        end
        current_emg = emg(first_idx:last_idx,:);
        current_stimulus = stimulus(first_idx:last_idx);
        current_repetition = repetition(first_idx:last_idx);
        fprintf("\n\n")
        fprintf("Current pose: %d  Current rep: %d\n" , pose, rep)
        switch pose
            case 1
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(1), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 2
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(2), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 3
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(3), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 4
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(4), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 5
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(5), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 6
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(6), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 7
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(7), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 8
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(8), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 9
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(9), 'minActivationTime', minActivationTime, 'minActivation', 1);
            case 10
                [used_signal, new_stimulus] = relabelling(current_emg, current_stimulus,'channel',channels(10), 'minActivationTime', minActivationTime, 'minActivation', 1);
        end
        if size(new_stimulus,1) ~= 1
            new_stimulus = new_stimulus';
        end
    %     if length(new_stimulus) ~= length(current_stimulus)
    %         new_stimulus = [0 new_stimulus];
    %     end
        total_new_stimulus = [total_new_stimulus new_stimulus];
        total_emg_len = total_emg_len + length(current_emg);
        total_stimulus_len = total_stimulus_len + length(new_stimulus);
        used_emg = [used_emg; used_signal];
    end
end
%size_diff = total_emg_len - total_stimulus_len;
total_new_stimulus = total_new_stimulus';


%%
% M = max(emg);
% for i=1:8
%     figure(i)
%     hold on
%     area(t,(stimulus)/10*M(i)*1.1,'FaceAlpha',0.2, "EdgeAlpha", 0.6)
%     %area(total_new_stimulus*M(i)*0.9,'FaceAlpha',0.2, "EdgeAlpha", 0.6)
%     area(t,emg(:,i), 'FaceAlpha',0.2);
% end

%% Find begin and end of stimulus
pose_change_idx = [];
for j = 2:length(total_new_stimulus)
    if total_new_stimulus(j)-total_new_stimulus(j-1)~=0
        pose_change_idx = [pose_change_idx j];
    end
end

%%
new_stimulus = zeros(length(total_new_stimulus),1);
new_repetition = zeros(length(total_new_stimulus),1);
for j = 1:2:length(pose_change_idx)
    start = pose_change_idx(j);
    stop = pose_change_idx(j+1);
    max_stim = max(stimulus(start:stop));
    max_rep = max(repetition(start:stop));
    new_stimulus(start:stop-1) = max_stim;
    new_repetition(start:stop-1) = max_rep;
end

%%
figure
hold on
M = max(used_emg);
area(relabel_stim/10*M*1,2,'FaceAlpha',0.2, "EdgeAlpha", 0.6)
%area(new_repetition/6*M,'FaceAlpha',0.2, "EdgeAlpha", 0.6)
area(new_stimulus/10*M*1.1,'FaceAlpha',0.2, "EdgeAlpha", 0.6)
area(used_emg,'FaceAlpha',0.2, "EdgeAlpha", 0.6)
%area(emg(:,2),'FaceAlpha',0.2, "EdgeAlpha", 0.6)

% Notice: some repetition might be a bit off. It's better to check and fix
% the activation by hand. I also wrote a program to fix the data if you
% changed the stimulus but forgot about the repetition (happened to me...).
%%

diff = (new_repetition>0)-(new_stimulus>0);
sum(diff)

% %% Save file as struct
% 
% %pathname = "D:\Tesi_Magistrale\Programmi\DATA\ESP32\Manfredo_offset_relabel\day1\mat";
% s = struct('sample_frequency',1000,'emg',emg,'stimulus',stimulus,'repetition',repetition,'relabel_stimulus',new_stimulus,'relabel_repetition',new_repetition,'angle',angle);
% save(fullfile(pathname, strcat(filename,"_relabel")),'-struct','s','-v7')
% %% Save file as csv
% labels = ["Channel_1","Channel_2","Channel_3","Channel_4","Channel_5","Channel_6","Channel_7","Channel_8","Repetition", "Stimulus", "Relabelled_repetition","Relabelled_stimulus", "Angle"];
% data = [emg(:,:), repetition, stimulus,new_repetition, new_stimulus,angle];
% data = [labels; data];
% 
% savepath = "D:\Tesi_Magistrale\Programmi\DATA\ESP32\Manfredo_offset_relabel\day1\csv";
% writematrix(data,fullfile(savepath,strcat(filename,"_relabel.csv")));
