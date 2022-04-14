function [xout, Pout,err] = ukfPredict(xin, Pin, Q, n, f, lambda, Wm, Wc, eps)
% UKFPREDICT prediction step of UKF
%   [xout, Pout,err] = ukfPredict(xin, Pin, Q, n, f, lambda, Wm, Wc, eps)

% Original Code from: Nick Galioto. University of Michigan.  Dec. 20, 2019
    % See from <a href="matlab:web('https://github.com/ngalioto/BayesID')">BayesID </a>
    
    [X,err] = formSigmaPoints(xin, Pin, n, lambda);

    if (err == 0)
    Xhat = zeros(size(X));
    for i = 1:2*n+1
        Xhat(:,i) = f(X(:,i));
    end
    
    xout = sum(Xhat .* Wm,2);
    Pout = Wc .* (Xhat - xout)*(Xhat - xout)' + Q + eps*eye(n);
    else
        xout = xin;
        Pout = Pin;
    end
end