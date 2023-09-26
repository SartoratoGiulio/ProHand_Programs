function [emg,t,repetition,stimulus,angle] = data2emg(data, Fs)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    emg = data(:,1:8);
    repetition = data(:,9);
    stimulus = data(:,10);
    angle = data(:,11);
    l = size(emg,1);
    t = 0:1/Fs:(l-1)/Fs;
end

