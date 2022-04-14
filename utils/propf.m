function xout = propf(xin, f, dt, k, jac)
% PROPF wrapper for discrete dynamics propagotor using odes15s
%   xout = propf(xin, f, dt, k, jac)

    t = k*dt;
    xout = xin;
    odeOpts = odeset('Jacobian', jac);
    [~, xout] = ode15s(@(t,x)f(x), [t, t+dt], xout, odeOpts);
    xout = xout(end,:)';
end