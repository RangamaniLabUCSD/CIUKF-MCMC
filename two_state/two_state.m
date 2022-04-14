function dxdt = two_state(t, x, params)
% TWOSTATE This function evaluates the differential equation for the two compartment model presented in Villaverde et al. 2019 IEEE Control Sys. Lett.
%    dxdt = two_state(t, x, params)
% x = [x1, x2, x3, x4]' where x(x) = u(t) and x4 = t
% u(t) = a*t + b so x2dot = a

% default parameters
% Parameter vector: [k1e, k12, k21, b];
    default_params = [1, 1, 1, 2];       
    if nargin < 2
	    params = default_params;
    end
    
    % unpack parameters (for clarity)
    k1e = params(1); % k1e + k12
    k12 = params(2); 
    k21 = params(3);
    b   = params(4);

    % input slope
    a1 = 1;
    u = stepFuncInput(t);

    dxdt = [-(k1e + k12)*x(1) + k21*x(2) + b*u;  
            k12*x(1) - k21*x(2)];
            % stepFuncInput(x(3), x(4), 1, 1, a1, 1);
            % 1]; % time dependent
    
    % % unpack parameters (for clarity)
    % k1e = params(1);
    % k12 = params(2);
    % k21 = params(3);
    % b   = params(4);

    % % input slope
    % a = 1;

    % dxdt = [-(k1e + k12)*x(1) + k21*x(2) + b*x(3);  
    %         k12*x(1) - k21*x(2);
    %         stepFuncInput(x(3), x(4), 1, 1, a, 1);
    %         1]; % time dependent
end

% function dudt = stepFuncInput(u, t,ton, toff, uon, uoff)
%     if t < ton
%         dudt = uoff;
%     elseif t >= ton
%         dudt = -u; 
%     elseif t >= ton && u <= 0.5
%         dudt = 0;
%     else
%         dudt = 0;
%     end
% end

function u = stepFuncInput(t)
    u_init = 0; slope = 2;
    ton = 1;
    if t < ton
        u = u_init + (slope*t);
    else
        u = 1.5*exp(1-t);
    end
end
