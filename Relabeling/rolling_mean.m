function mn = rolling_mean(signal,window)
mn=[];
for i = 1+window/2:size(signal, 1)-window/2
    x = signal(i-window/2:i+window/2);
    s = sum(x)/window;
    mn = [mn s];
end
end