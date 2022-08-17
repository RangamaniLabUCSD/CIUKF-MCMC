function loglikelihood = ciukflp_quadProg(theta, x0, P0, f, H, Q, R, y, alpha,beta,kappa,eps, optOptions)
% CIUKFLP_QUADPROG run ciukf to estimate log likelihood (ASSUMES linear measurements!)
%   loglikelihood = ciukflp_quadProg(theta, x0, P0, f, H, Q, R, y, alpha,beta,kappa,eps, optOptions)

    T = size(y,2);
    n = size(x0,1);

    lambda = 1;

    num_sets = size(theta, 1);
    loglikelihood = zeros(num_sets, 1);

    parfor set = 1:num_sets
        f_set = @(i,x) f(i, x, theta(set,:));
        Sigma = Q(theta(set,:));
        Gamma = R(theta(set,:));

        m = x0; C = P0;
        for i = 1:T
            [m, C,err, wICUT] = CIUKFpredict(m, C, Sigma, n, f_set, lambda, eps, i);
            if (err ~= 0)
                loglikelihood(set) = -Inf;
                break;
            end
            [m,C,v,S,Sinv,err] = CIUKFupdate_quadProg(m, C, y(:,i), Gamma, n, H, lambda, wICUT, eps, alpha, beta, optOptions);
            if (err ~= 0)
                loglikelihood(set) = -Inf;
                break;
            end
            loglikelihood(set) = loglikelihood(set) - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        end
    end
end