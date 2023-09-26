close all
clear all
clc
%%
pathname = "C:\Users\sarto\OneDrive\Desktop\Tesi_Magistrale\DATA\ESP32\Franco_offset_relabel\day1";
%pathname = "D:\Tesi_Magistrale\Programmi\DATA\ESP32\Giulio_offset_relabel\day1";
angle = 5;
filename = sprintf("Giulio_6rep_5sec_angle_%d_ESP32_1000Hz_relabel",angle);
full_path = fullfile(pathname,strcat(filename,".mat"));
load(full_path);
M = max(emg)
channel = 2;
diff = ((relabel_repetition>0)-(relabel_stimulus>0));
%%
figure
hold on
%area(emg(:,channel),'FaceAlpha',0.2, "EdgeAlpha", 0.6);
%area((relabel_stimulus>0)*M(channel)*1.4, 'FaceAlpha',0.2, "EdgeAlpha", 0.6);
%area((relabel_repetition>0)*M(channel)*1.2, 'FaceAlpha',0.2, "EdgeAlpha", 0.6);
area(diff*M(channel),'FaceAlpha',0.2, "EdgeAlpha", 0.6);
%%
sum(diff)
for i = 1:length(diff)
    if diff(i) == 1
        relabel_repetition(i) = 0;
    elseif diff(i) == -1
        relabel_repetition(i) = 100;
    end
end
sum((relabel_repetition>0)-(relabel_stimulus>0))
%%
x = find(relabel_repetition == 100);
if length(x)>0
    rep_value_1 = relabel_repetition(x(1)-1);
    rep_value_2 = relabel_repetition(x(end)+1);
    
    if rep_value_1 == 0 && rep_value_2 ~= 0
        relabel_repetition(x(:)) = rep_value_2;
    elseif rep_value_1 ~= 0 && rep_value_2 == 0
        relabel_repetition(x(:)) = rep_value_1;
    end
end
%%
figure
hold on
%area(emg(:,channel),'FaceAlpha',0.2, "EdgeAlpha", 0.6);
%area((relabel_stimulus>0)*M(channel)*1.4, 'FaceAlpha',0.2, "EdgeAlpha", 0.6);
%area((relabel_repetition)*M(channel)*1.2, 'FaceAlpha',0.2, "EdgeAlpha", 0.6);
area(((relabel_repetition>0)-(relabel_stimulus>0))*M(channel),'FaceAlpha',0.2, "EdgeAlpha", 0.6);
%%
s = struct('sample_frequency',1000,'emg',emg,'stimulus',stimulus,'repetition',repetition,'relabel_stimulus',relabel_stimulus,'relabel_repetition',relabel_repetition,'angle',angle);
%%
save(fullfile(pathname, strcat(filename,".mat")),'-struct','s','-v7')
% %%
% labels = ["Channel_1","Channel_2","Channel_3","Channel_4","Channel_5","Channel_6","Channel_7","Channel_8","Repetition", "Stimulus", "Relabelled_repetition","Relabelled_stimulus", "Angle"];
% data = [emg(:,:), repetition, stimulus,relabel_repetition, relabel_stimulus,angle];
% data = [labels; data];
% savepath = strcat(pathname,"\csv");
% %%
% writematrix(data,fullfile(savepath,strcat(filename,".csv")));