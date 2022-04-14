function [xout, Pout,v,S,Sinv,err] = CIUKFupdate(xin, Pin, y, R, n, h, lambda,wICUT,eps, alpha, beta)
% CIUKFUPDATE update step of UKF
%   [xout,Pout,v,S,Sinv,err] = CIUKFupdate(xin, Pin, y, R, n, h, lambda,wICUT,eps, alpha, beta)

    
    m = size(y, 1);
    [X,err] = formSigmaPoints(xin, Pin, n, lambda);
    [Wm, Wc] = formWeights(n, lambda, alpha, beta);
    if (err == 0)
        Yhat = zeros(m,size(X,2));
        for i = 1:2*n+1
            Yhat(:,i) = h(X(:,i));
        end

        mu = sum(Wm .* Yhat,2);
        S = Wc .* (Yhat - mu)*(Yhat - mu)' + R + eps*eye(m); % Pyy_{k|k-1}
        C = Wc .* (X - xin)*(Yhat - mu)'; % Pxy_{k|k-1}

        Sinv = eye(m) / S; % (Pyy_{k|k-1})^-1
        K = C*Sinv; % Kalman gain Pxy_{k|k-1}(Pyy_{k|k-1})^-1

        % Solve optimization problem for xout (x_{k|k})
        objective = @(xout) (y-h(xout))'*inv(R)*(y-h(xout)) + (xout - xin)'*inv(Pin)*(xout - xin);
        xout0 = xin;
        lb = zeros(size(xin)); ub = Inf*ones(size(xin));
        options = optimoptions('fmincon','Display','off');
        xout = fmincon(objective, xout0, [], [], [], [], lb, ub, [], options);
        
        Pout = Pin - K*S*K' + eps*eye(n);
        Pout = 0.5*(Pout + Pout');
        v = y - mu;
    else
        xout = xin;
        Pout = Pin;
        v = 0;
        S = 0;
        Sinv = 0;
    end
end