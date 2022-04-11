function [LCA, period] = limitCycleFeatures(trajectory, startIdx, peakThreshold, LCAthresh, Decaythresh, DT, nullValue)
% LIMITCYCLEFEATURES compues limit cycle amplitude and period
%   [LCA, period] = limitCycleFeatures(trajectory, startIdx, peakThreshold, LCAthresh, Decaythresh, DT, nullValue)


    LCA = 0;
    period = 0;
    delay = startIdx;
    periodScale = DT/60;
    % OSCidxs = [];

    % temp = xpost(3, :,i);
    [lMax, maxLocs]  = findpeaks(trajectory(delay:end));
    [lMin, minLocs]  = findpeaks(-trajectory(delay:end));
    lMin = trajectory(delay+minLocs);

    if isempty(lMax) | isempty(lMin)
        LCA = nullValue;% LCArange = nan;
        period = nullValue;% periodRange = nan;
        return
    else
        peakDiff= range(lMax);
        if peakDiff<peakThreshold
                % Compute LCA and range in values
                numPks = min(numel(lMax), numel(lMin));
                amps = lMax(1:numPks) - lMin(1:numPks);
                LCA = mean(amps);
                LCArange = range(amps);
                 
                if (LCA <= LCAthresh) | (LCArange>= Decaythresh)
                    LCA = nullValue;
                    period = nullValue;
                    return
                    % LCArange = nan;
                end

                % Periods
                max2maxPeriod = diff(maxLocs)*periodScale;
                min2minPeriod = diff(minLocs)*periodScale;
                if ~isempty(max2maxPeriod) | ~isempty(min2minPeriod)
                    period = mean([max2maxPeriod(:); min2minPeriod(:)]);
                else
                    LCA = nullValue;
                    period = nullValue;
                end
                % periodRange = mean([diff(max2maxPeriod), diff(min2minPeriod)]);
        else
                LCA = nullValue; %LCArange = nan;
                period = nullValue; %periodRange = nan;
        end
    end

end