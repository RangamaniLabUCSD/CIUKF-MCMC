function [Wm, Wc] = formWeights(n, lambda, alpha, beta)
% FORMWEIGHTS creates weights for UT
%   [Wm, Wc] = formWeights(n, lambda, alpha, beta) creates weights

% Original Code from: Nick Galioto. University of Michigan.  Dec. 20, 2019
    % See from <a href="matlab:web('https://github.com/ngalioto/BayesID')">BayesID </a>
    Wm = zeros(1, 2*n+1);
    Wc = zeros(1, 2*n+1);

    Wm(1) = lambda / (n+lambda);
    Wm(2:end) = 1 / (2*(n + lambda));
    Wc(1) = lambda / (n+lambda) + 1 - alpha^2 + beta;
    Wc(2:end) = 1 / (2*(n+lambda));
end