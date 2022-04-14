function logpost = ciukflp(theta, m, C, f, H, Sigma, Gamma, y, prior,alpha,beta,kappa,eps)
    T = size(y,2);
    n = size(m,1);
    logpost = prior(theta);
    if (isfinite(logpost))  % can immediately reject if logprior is -Inf
        lambda = 3-n; %alpha^2 * (n+kappa) - n; % this is a nonstandard form of lambda?
        for i = 1:T
            [m, C,err, wICUT] = CIUKFpredict(m, C, Sigma, n, f, lambda, eps);
            if (err ~= 0)
                logpost = -Inf;
                disp(['Covariance is not positive definite. Filter prematurely halted at predict; k= ',num2str(i)]);
                break;
            end
            [m,C,v,S,Sinv,err] = CIUKFupdate_quadProg(m, C, y(:,i), Gamma, n, H, lambda, wICUT, eps, alpha, beta);
            if (err ~= 0)
                logpost = -Inf;
                disp(['Covariance is not positive definite. Filter prematurely halted at update; k= ',num2str(i)]);
                break;
            end
            logpost = logpost - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        end
    end
end