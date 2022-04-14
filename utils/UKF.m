function logpost = UKF(theta, m, C, f, h, Sigma, Gamma, y, prior,alpha,beta,kappa,eps)
% UKF unscented Kalman filter for approx marginalization
%   logpost = UKF(theta, m, C, f, h, Sigma, Gamma, y, prior,alpha,beta,kappa,eps) run UKF
%
% Original Code from: Nick Galioto. University of Michigan.  Dec. 20, 2019
    % See from <a href="matlab:web('https://github.com/ngalioto/BayesID')">BayesID </a>

    T = size(y,2);
    n = size(m,1);
    logpost = prior(theta);
    if (isfinite(logpost))  % can immediately reject if logprior is -Inf
        lambda = 1; %alpha^2 * (n+kappa) - n;
        [Wm, Wc] = formWeights(n, lambda, alpha, beta);

        for i = 1:T
            f_i = @(x) f(i, x);
            [m, C,err] = ukfPredict(m, C, Sigma, n, f_i, lambda, Wm, Wc, eps);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            [m,C,v,S,Sinv,err] = ukfUpdate(m, C, y(:,i), Gamma, n, h, lambda, Wm, Wc, eps);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            logpost = logpost - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        end 
    end
end

