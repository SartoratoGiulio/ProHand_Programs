function [signalArray, numIntervals] = deleteFalsePositives(signalArray, minLength)
    % Initialize variables
    currentIntervalLength = 0;
    modified = false;
    numIntervals = 0;
    
    % Iterate through the signal array
    for i = 1:length(signalArray)
        % If the current element is 1, increase the interval length
        if signalArray(i) == 1
            currentIntervalLength = currentIntervalLength + 1;
            modified = false;
        else
            % If the current element is 0 and the interval length is shorter than minLength
            % replace the ones in the interval with zeros
            if currentIntervalLength < minLength && ~modified
                signalArray(i-currentIntervalLength:i-1) = 0;
                modified = true;
            end
            
            % Reset the interval length
            currentIntervalLength = 0;
            
            if ~modified
                numIntervals = numIntervals + 1;
            end
        end
    end
    
    % Check the last interval if the signal ends with 1s
    if currentIntervalLength < minLength && ~modified
        signalArray(end-currentIntervalLength:end) = 0;
    end
   
end

