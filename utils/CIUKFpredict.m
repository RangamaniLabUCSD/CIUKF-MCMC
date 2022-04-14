function [xout, Pout,err, wICUT] = CIUKFpredict(xin, Pin, Q, n, f, lambda, eps, k)
% CIUKFPREDICT prediction step of CIUKF
%   [xout, Pout,err, wICUT] = CIUKFpredict(xin, Pin, Q, n, f, lambda, eps, k)

    
    [X,wICUT,err] = formSigmaPointsWeights_ICUT(xin, Pin, n, lambda); % assume lb = 0, ub = infty
    if (err == 0)
        Xhat = zeros(size(X));
        for i = 1:2*n+1
            Xhat(:,i) = f(k, X(:,i));
        end
        xout = sum(Xhat .* wICUT,2);
        Pout = wICUT .* (Xhat - xout)*(Xhat - xout)';
        Pout = Pout + Q + eps*eye(n);
        Pout = 0.5 * (Pout + Pout');
    else
        xout = xin;
        Pout = Pin;
    end
end
