function [thetaMAP,thetaMEAN, fits, x_fit] = approximate_MAP(samples, bounds, KSbandwidth)
% APPROXIMATE_MAP uses kernel density estimation to approximate the map, mean and pdf
%   [thetaMAP,thetaMEAN, fits, x_fit] = approximate_MAP(samples, bounds) run with default bandwidth for ksdensity()
%
%   [thetaMAP,thetaMEAN, fits, x_fit] = approximate_MAP(samples, bounds, KSbandwidth) run with user specified bandwidth
%
%   See also KSDENSITY

        if nargin < 3
            runWithDefaultBand = true;
        else
            runWithDefaultBand = false;
        end

        thetaMAP = zeros(size(samples,1),1);
        thetaMEAN = zeros(size(samples,1),1);
        fits = {};
        x_fit = {};
        for i = 1:size(samples,1)
            if i <= size(bounds, 1)
                % fit kernel density to samples of ith parameter (for model parameters density is bounded)
                if runWithDefaultBand
                    [f, x] = ksdensity(samples(i,:), 'support', [bounds(i,1)-0.00000001, bounds(i,2)+0.0000000001],'BoundaryCorrection', 'reflection');
            else
                [f, x] = ksdensity(samples(i,:), 'support', [bounds(i,1)-0.00000001, bounds(i,2)+0.0000000001],'BoundaryCorrection', 'reflection', 'Bandwidth', KSbandwidth);
            end
                 
            % compute MAP point
            idxs = find(f == max(f));    
            thetaMAP(i) = x(idxs(1)); % MAP point is where the KS density equals its max val
            % compute mean
            thetaMEAN(i) = trapz(x, f.*x);
            fits{i} = f; 
            x_fit{i} = x;
        end
end
