function [xout,err] = formSigmaPoints(xin, Pin, n, lambda)
% FORMSIGMAPOINTS creates sigma points for unscented transform
%   [xout,err] = formSigmaPoints(xin, Pin, n, lambda) creates sigma point set

% Original Code from: Nick Galioto. University of Michigan.  Dec. 20, 2019
    % See from <a href="matlab:web('https://github.com/ngalioto/BayesID')">BayesID </a>
    
	[Psqrt,err] = chol(Pin);
    if (err == 0)
        sqrtP = sqrt(n+lambda)*Psqrt;
        xout = zeros(length(xin), 2*n+1);
        xout(:, 1) = xin;
        for i = 1:n
            xout(:,i+1) = xin + sqrtP(i,:)'; % rows since Pin = Psqrt'*Psqrt
            xout(:,i+n+1) = xin - sqrtP(i,:)';
        end
    else
        xout = xin;
    end
end