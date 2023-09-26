function [used_signal, restimulus] = relabelling(signal, stimulus,varargin)
p = inputParser;
addRequired(p,'signal');
addRequired(p,'stimulus');
addOptional(p,'channel',0);
addOptional(p,'minActivationTime',3000);
addOptional(p, 'minActivation', 6);
parse(p,signal,stimulus,varargin{:});
channel = p.Results.channel;
minActivationTime = p.Results.minActivationTime;
minActivation = p.Results.minActivation;
restimulus = [];
classes = 2;
class_change = false;
max_iter = 20;
iter = 0;
if channel == 0
    signal_rescale = zeros(size(signal));
    for i = 1:8
        signal_rescale(:,i) = rescale(signal(:,i));
    end
    mean_signal = mean(signal_rescale(stimulus>0,:));
    mean_noise = mean(signal_rescale(stimulus==0,:));
    mean_ratio = mean_signal./mean_noise;
    max_avg = max(mean_ratio);
    channel = find(mean_ratio==max_avg);
end
numIntervals = 0;
change = ischange(signal(:,channel),1, 'Threshold', 1.5);
%change = islocalmax(signal(:,channel), 1);
change_idx = find(change==1);
change_values = signal(change,channel);

% I found that in some cases. where all else fails
% adding another class to the EMGM algorithm improves
% the detection of the active emg signal
while numIntervals <minActivation || max(restimulus)==2
    iter = iter+1;
    [label, model, llh] = emgm(change_values', classes);
    label = label';
    %label = kmeans(change_values,classes);
    label = label -1;
    labelled_change_idx = [change_idx label];
    % Swap the labelling if needed
    mean0 = mean(signal(labelled_change_idx(labelled_change_idx(:,2) == 0,1),channel));
    mean1 = mean(signal(labelled_change_idx(labelled_change_idx(:,2) == 1,1),channel));
    if mean0>mean1
        label = (label-1)*-1;
        labelled_change_idx = [change_idx label];
    end
    x = zeros(length(signal),1);
    for i = 1:length(labelled_change_idx)
        x(labelled_change_idx(i,1)) = labelled_change_idx(i,2);
    end
    if(size(x,1)<size(x,2))
    x = x';
    end
    t1 = 80;%samples
    t2 = 80;%samples
    strel1 = strel('line', t1, 90);
    strel2 = strel('line', t2, 90);
    % Dilation with t1
    act = imdilate(x, strel1);
    % Erosion with t1
    act = imerode(act, strel1);
    % Erosion with t2
    act = imerode(act, strel2);
    % Dilation with t2
    act = imdilate(act, strel2)';
    [restimulus, numIntervals] = deleteFalsePositives(act, minActivationTime);
    fprintf("Number of intervals: %d\n", numIntervals);
    fprintf("Channel used: %d\n", channel);
    if iter>=max_iter && ~class_change
        iter = 0;
        classes = 3;
        class_change = true;
        fprintf("Max Iteration reached! Changing classe number to 3.\n");

    elseif iter>=max_iter && class_change
        fprintf("Second Max Iteration reached! Using default stimulus\nTry changing minActivation\n");
        restimulus = stimulus;
        break
    end
end

used_signal = signal(:,channel);

end