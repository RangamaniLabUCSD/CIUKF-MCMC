function y  = generateData(f, x0, t, H, sigmaR, odeOpts)
% GENERATEDATA generates noisy samples
%   y  = generateData(f, x0, t, H, sigmaR, odeOpts)

    [~, x] = ode15s(@(t,x)f(x), t, x0, odeOpts);
    y = H*x' + sigmaR.*randn(size(H*x'));
    y = y(:,2:end);
end