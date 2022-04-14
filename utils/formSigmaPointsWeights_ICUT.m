function [xout,wICUT, err] = formSigmaPointsWeights_ICUT(xin, Pin, n, lambda, lb, ub)
% FORMSIGMAPOINTSWEIGHTSICUT This function created the sigma points for the interval constrained unscented transform (ICUT)
%   [xout,wICUT, err] = formSigmaPointsWeights_ICUT(xin, Pin, n, lambda, lb, ub) creates weights

% lb is the lower bound - default to 0
% ub is the upper bounds - default to Inf
% The implementation is based on the formulas given in:
% -  Vachhani et al. 2006: Robust and reliable estimation via Unscented Recursive
%                           Nonlinear Dynamic Data Reconciliation
% - Tiexiera et al. 2008: Unscented filtering for interval-constrained nonlinear systems
    
    % defualt lb and ub to 0 and Inf
    if nargin < 6
        ub = Inf*ones(n,1);
    end
    if nargin < 5
        lb = zeros(n,1);
    end
    
	[Psqrt,err] = chol(Pin);
    if (err == 0)
        S = [Psqrt, -Psqrt]; % S matrix
        Theta = zeros(size(S)); % matrix to store possible steps in each direction
        
        % Where S is 0, theta(i,j) = sqrt(n+lambda);
        Theta(S == 0) = sqrt(n + lambda); 
        
        % S is negative
        [negRowIdx, negColIdx] = find(S < 0); % get idxs of neg vals
        
        for i = 1:numel(negRowIdx) % loop over each of these :(
            % use formula for when S(i,j) is leq zero
            Theta(negRowIdx(i), negColIdx(i)) = min([sqrt(n+lambda), (lb(negRowIdx(i))-xin(negRowIdx(i)))./ S(negRowIdx(i), negColIdx(i))]);
        end

        % S is positive
        [posRowIdx, posColIdx] = find(S > 0); % get idxs of neg vals
        for i = 1:numel(posRowIdx) % loop over each of these :(
            % use formula for when S(i,j) is geq zero
            Theta(posRowIdx(i), posColIdx(i)) = min([sqrt(n+lambda), (ub(posRowIdx(i))-xin(posRowIdx(i)))./ S(posRowIdx(i), posColIdx(i))]);
        end

        thetaI = min(Theta); % the ith theta is the min of the jth column of Theta
        xout = zeros(length(xin), 2*n+1);
        xout(:, 1) = xin;
        for i = 1:n
            xout(:,i+1) = xin + thetaI(i)*Psqrt(i,:)'; % rows since Pin = Psqrt'*Psqrt
            xout(:,i+n+1) = xin - thetaI(i+n)*Psqrt(i,:)';
        end

        % weights
        Stheta = sum(thetaI);
        a = (2*lambda - 1) / (2 * (n+lambda) * (Stheta - (2*n + 1)*(sqrt(n+lambda))));
        b = (1/(2*(n+lambda))) - ((2*lambda - 1) / (2*sqrt(n+lambda) * (Stheta - (2*n + 1)*(sqrt(n+lambda)))));
        wICUT = zeros(1, 2*n+1);
        wICUT(1) = b;
        wICUT(2:end) = a*thetaI + b;
    else
        xout = xin;
        wICUT = [];
    end
end