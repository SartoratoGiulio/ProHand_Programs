close all
clear all
clc

%% Get emg signal
pathname = "C:\Users\sarto\OneDrive\Desktop\Tesi_Magistrale\DATA\ESP32\Franco_offset_relabel\day1\csv\";
%pathname = "D:\Tesi_Magistrale\Programmi\DATA\ESP32\Franco_offset";
filename = "Franco_6rep_5sec_angle_1_ESP32_1000Hz_relabel";
data = readmatrix(fullfile(pathname,strcat(filename,".csv")));
Fs = 1000;%Hz
a = 7;
change_th = 0.2;
channel = 0;
start_rep = 1;
min_act = 6;
t1 = 100;%saples
t2 = 100;%samples
%[emg,t,repetition,stimulus,angle] = data2emg(data(1+slide:54000+slide,:),Fs);
[emg,t,repetition,stimulus,angle] = data2emg(data,Fs);

pose = [find(stimulus==a & repetition == start_rep,1,'first'), find(stimulus==a & repetition == start_rep+min_act-1,1,'last')];

slide = 2000;
emg = emg(pose(1)-slide:pose(2)+slide,:);
t = t(pose(1)-slide:pose(2)+slide);
stimulus = stimulus(pose(1)-slide:pose(2)+slide);
repetition = repetition(pose(1)-slide:pose(2)+slide);
M = max(emg,[],'all');

%% Plot raw emg
names = ["Channel 1", "Channel 2", "Channel 3","Channel 4", "Channel 5","Channel 6","Channel 7","Channel 8"];
figure("Name","Preprocessing");
sgtitle("Raw EMG")
for i=1:8
    subplot(2,4,i)
        area(t, stimulus*M/a, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.2)
        hold on
        area(t, emg(:,i), 'FaceColor', 'r' ,'FaceAlpha', 0.2, 'EdgeColor', 'r')
        ylim([0 M])
        title(names(i))
end
% %%
% mean_emg = mean(emg,2);
% figure
% hold on
% M = max(mean_emg);
% area(t, stimulus*M/a, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.2)
% area(t, mean_emg, 'FaceColor', 'r' ,'FaceAlpha', 0.2, 'EdgeColor', 'r')

%% Find Changes
emg_rescale = zeros(size(emg));
for i = 1:8
    emg_rescale(:,i) = rescale(emg(:,i));
end
mean_signal = mean(emg_rescale(stimulus>0,:));
mean_noise = mean(emg_rescale(stimulus==0,:));
mean_ratio = mean_signal./mean_noise;
max_avg = max(mean_ratio);
if channel == 0
    channel = find(mean_ratio==max_avg);
end
change = ischange(emg(:,channel),'mean',1, 'Threshold', change_th);
%change = islocalmax(emg(:,channel), 1);

change_idx = find(change==1);
change_values = emg(change,channel);
%%
numIntervals = 0;
while numIntervals<1
[label, model, llh] = emgm(change_values', 2);
label = label';

label = label -1;
labelled_change_idx = [change_idx label];
% Swap the labelling if needed
mean0 = mean(emg(labelled_change_idx(labelled_change_idx(:,2) == 0,1),channel));
mean1 = mean(emg(labelled_change_idx(labelled_change_idx(:,2) == 1,1),channel));
if mean0>mean1
    label = (label-1)*-1;
    labelled_change_idx = [change_idx label];
end
x = zeros(size(emg(:,channel),1),1);
for i = 1:size(labelled_change_idx,1)
    x(labelled_change_idx(i,1)) = labelled_change_idx(i,2);
end
if(size(x,1)<size(x,2))
    x = x';
end
strel1 = strel('line', t1, 90);
strel2 = strel('line', t2, 90);
% Dilation with t1
act = imdilate(x, strel1);
% Erosion with t1
act_close = imerode(act, strel1);
% Erosion with t2
act = imerode(act_close, strel2);
% Dilation with t2
act_open = imdilate(act, strel2)';
[act2, numIntervals] = deleteFalsePositives(act_open, 3000);
end
size_diff = (length(emg(:,channel))-length(act2))
%% Plot results
if size_diff == 1
    interval = 1:length(emg)-1;
elseif rem(size_diff,2)==1
    interval = ((size_diff-1)/2:length(emg)-(size_diff-1)/2);
else
    interval = (size_diff)/2+1:(length(emg)-size_diff/2);
end
M = max(emg(:,channel));
f = figure();
f.Position = [450   41  725.8  740.8];
subplot(5,1,1)
hold on
area(t,stimulus/a,'Facealpha', 0.2);
area(t,emg(:,channel)/M,'Facealpha', 0.2)
title("a) Rectified EMG with original stimulus")
subplot(5,1,2)
plot(t,x, 'Color', 'black')
title("b) Change points after clustering")
subplot(5,1,3)
area(t,act_close,'Facealpha', 0.2, 'FaceColor','b');
title(sprintf("c) Closing operation with t_1 = %d",t1))
subplot(5,1,4)
area(t,act_open, 'Facealpha', 0.2, 'FaceColor','r');
title(sprintf("d) Opening operation with t_2 = %d",t2))
subplot(5,1,5)
hold on
area(t,act2,'Facealpha', 0.2);
area(t,emg(:,channel)/M,'Facealpha', 0.2)
title("e) Rectified EMG with new stimulus")
xlabel("t [ms]")
xlim = [t(1), t(end)];
ylim = [0, 1.2];




