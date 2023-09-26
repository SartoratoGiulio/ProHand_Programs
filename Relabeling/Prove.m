close all
clear all
clc
%%
%pathname = "C:\Users\sarto\OneDrive\Desktop\Tesi_Magistrale\DATA\ESP32\Franco_offset_relabel\day1\csv";
pathname = "D:\Tesi_Magistrale\DATA\Franco_offset_relabel\day1\csv\";
angle = 5;
filename = sprintf("Franco_6rep_5sec_angle_%d_ESP32_1000Hz_relabel",angle);
%load(fullfile(pathname,strcat(filename,".mat")));
data = readmatrix(fullfile(pathname,strcat(filename,".csv")));

emg = data(:,1:8);
repetition = data(:,9);
stimulus = data(:,10);
relabel_repetition = data(:,11);
relabel_stimulus = data(:,12);
sub_angle = data(:,13)*-1;

Fs = 1000;%Hz
t = 0:1/Fs:(length(emg)-1)/Fs;
n_channels = size(emg,2);
a = 8;
start_rep = 1;
min_act = 6;
pose = [find(relabel_stimulus==a & relabel_repetition == start_rep,1,'first'), find(relabel_stimulus==a & relabel_repetition == start_rep+min_act-1,1,'last')];
slide = 2000;
% emg = emg(pose(1)-slide:pose(2)+slide,:);
% t = t(pose(1)-slide:pose(2)+slide);
% relabel_stimulus = relabel_stimulus(pose(1)-slide:pose(2)+slide);
% relabel_repetition = relabel_repetition(pose(1)-slide:pose(2)+slide);
% data = data(pose(1)-slide:pose(2)+slide,:);
M = max(emg,[],'all');
%%
figure("Name","Preprocessing");
sgtitle("Raw EMG")
for i=1:n_channels
    subplot(2,4,i)
        hold on
        area( relabel_stimulus*M/10, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.2)
        %area( relabel_repetition*M/6, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.2)
        area( emg(:,i), 'FaceColor', 'r' ,'FaceAlpha', 0.2, 'EdgeColor', 'r')
        ylim([0 M])
        title(sprintf("Channel_%d",i+1))
end
%%
emg_rest = emg(relabel_stimulus==0,:);
rest_mean = mean(emg_rest);
rest_std = std(emg_rest);
figure
hold on
area(relabel_stimulus>0, 'FaceAlpha',0.2, "EdgeAlpha", 0.6)
area(emg(:,6)>(rest_mean(6)+3*rest_std(6)), 'FaceAlpha',0.2, "EdgeAlpha", 0.6)

%% Save file as struct
%pathname = "C:\Users\sarto\OneDrive\Desktop\Tesi_Magistrale\DATA\ESP32\Franco_offset_relabel\day1\mat";
pathname = "D:\Tesi_Magistrale\DATA\Franco_offset_relabel\day1\mat";
s = struct('sample_frequency',1000,'emg',emg,'stimulus',stimulus,'repetition',repetition,'relabel_stimulus',relabel_stimulus,'relabel_repetition',relabel_repetition,'angle',sub_angle);
save(fullfile(pathname, strcat(filename,"")),'-struct','s','-v7')
%% Save file as csv
labels = ["Channel_1","Channel_2","Channel_3","Channel_4","Channel_5","Channel_6","Channel_7","Channel_8","Repetition", "Stimulus", "Relabelled_repetition","Relabelled_stimulus", "Angle"];
data = [emg(:,:), repetition, stimulus,relabel_repetition, relabel_stimulus,sub_angle];
data = [labels; data];
%savepath = "C:\Users\sarto\OneDrive\Desktop\Tesi_Magistrale\DATA\ESP32\Franco_offset_relabel\day1\csv";
savepath = "D:\Tesi_Magistrale\DATA\Franco_offset_relabel\day1\csv";
writematrix(data,fullfile(savepath,strcat(filename,"")));
