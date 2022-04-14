function [xout, Pout,v,S,Sinv,err] = CIUKFupdate_quadProg(xin, Pin, y, R, n, H, lambda,wICUT,eps, alpha, beta, optOptions)
% CIUKFUPDATEQUADPROG update step of CIUKF with linear meas
%   [xout, Pout,v,S,Sinv,err] = CIUKFupdate_quadProg(xin, Pin, y, R, n, H, lambda,wICUT,eps, alpha, beta, optOptions)
%   Note: H is the linear measurement operator

    m = size(y, 1);
    warning('off')
    [X,err] = formSigmaPoints(xin, Pin, n, lambda);
    [Wm, Wc] = formWeights(n, lambda, alpha, beta);
    if (err == 0)
        Yhat = zeros(m,size(X,2));
        for i = 1:2*n+1
            Yhat(:,i) = H*X(:,i);
        end

        mu = sum(Wm .* Yhat,2);
        S = Wc .* (Yhat - mu)*(Yhat - mu)' + R + eps*eye(m); % Pyy_{k|k-1}
        C = Wc .* (X - xin)*(Yhat - mu)'; % Pxy_{k|k-1}

        Sinv = eye(m) / S; % (Pyy_{k|k-1})^-1
        K = C*Sinv; % Kalman gain Pxy_{k|k-1}(Pyy_{k|k-1})^-1

        % Solve optimization problem for xout (x_{k|k})
        % objective = @(xout) (y-h(xout))'*inv(R)*(y-h(xout)) + (xout - xin)'*inv(Pin)*(xout - xin);
        Rinv = inv(R); Pininv = inv(Pin);
        Hqp = 2*(H'*Rinv*H + Pininv); % Mult by two in order to cancel 1/2 in assumed form (1/2)x'*Aqp*x
        fqp = -2*(y'*Rinv*H + xin'*Pininv)'; % Note extra trapose here bc MATLAB wants for of fqp'*x but we have f*x so take fqp=f'
        xout0 = xin;
        lb = zeros(size(xin)); ub = Inf*ones(size(xin));
        
        if(isreal(Hqp) & isreal(fqp) & isreal(xout0))
            xout = quadprog(Hqp, fqp, [], [], [], [], lb, ub, xout0, optOptions);
            
            Pout = Pin - K*S*K' + eps*eye(n);
            Pout = 0.5*(Pout + Pout');
            v = y - mu;
        else
            disp('imag')
            err = 1;
            xout = xin;
            Pout = Pin;
            v = 0;
            S = 0;
            Sinv = 0;
        end
    else
        xout = xin;
        Pout = Pin;
        v = 0;
        S = 0;
        Sinv = 0;
    end
end