function dxdt = phosphatase_kinase(t, x, p, f_Ca)
    % This function evaluates the ODE for the coupled Phosphatase-Kinase system
    %   that is modeled in Pi and Lisman 2008. J Neusoci

    % Inputs:
    %   - time: t
    %   - State vector: x  = [pK, P, A] e.g. concentrations (uM) of phosphorylated-CaMKII, phosphatase, and membrane bound AMPAR level (unitless)
    %   - Parameter vector: p = [k1, k2, k3, k4, k5, k6, k7, k8, c1, c2, c3, c4, Km1, Km2, Km3, Km4, Km5, K0, P0, Ktot, Ptot, Atot]
    %           default: p_nominal (defined below) is the nominal parameter set that was used in Pi and Lisman.
    %           units, definitions and nominal values:
    %           Kinase
    %           - k1: Kinase autophos rate (k1 in Pi and Lisman) - 2 1/s
    %           - k2: Kinase dephos by phosphatase rate (k2 in Pi and Lisman) - 15 1/s
    %           - k3: Basal kinase activity rate (k3 in Pi and Lisman) - 1 1/s
    %           - k4: Ca-dep phos rate (k4 in Pi and Lisman) - 120 1/s
    %           Phosphatase:
    %           - k5: Phosphatase autodephos rate (k11 in Pi and Lisman) - 2 1/s
    %           - k6: Phosphatase phos by Kinase (CaMKII or other) rate (k12 in Pi and Lisman) - 15 1/s
    %           - k7: Basal phosphatase activity rate (k13 in Pi and Lisman) - 1 1/s
    %           - k8: Ca-dep dephos rate (k14 in Pi and Lisman) - 80 1/s
    %           AMPAR trafficking:
    %           - c1: AMPAR membrane insertion rate scaling (k21 = c1*pK + c3 in Pi and Lisman) - 1
    %           - c2: AMPAR membrane removal rate scaling (k22 = c2*P + c4 in Pi and Lisman) - 1
    %           - c3: AMPAR membrane insertion rate const (k21 = c1*pK + c3 in Pi and Lisman) - 6 1/s
    %           - c4: AMPAR membrane removal rate const (k22 = c2*P + c4 in Pi and Lisman) - 8 1/s
    %           Kinase:
    %           - Km1: Eq constant for kinase (Km1 in Pi and Lisman) - 10 uM
    %           - Km2: Eq constant for phosphatase (Km2 in Pi and Lisman) - 0.3 uM
    %           - Km3: Eq constant for Ca2+ (Km in Pi and Lisman) - 4 uM
    %           Phosphatase:
    %           - Km4: Eq constant for kinase (Km11 in Pi and Lisman) - 10 uM
    %           - Km5: Eq constant for phosphatase (Km12 in Pi and Lisman) - 1 uM
    %           Basal concentrations:
    %           - K0: basal conc active kinase  - 0.5 u
    %           - P0: basal conc active phosphatase - 0.5 uM
    %           Total Concentrations:
    %           - Ktot: total amount of Kinase (Ktot = K + pK) - 20 uM
    %           - Ptot: total amount of Phosphatase (Ptot = P + pP) - 20 uM
    %           - Atot: total amount of AMPAR (Atot = A + Aint) - 1
    %   - Calcium input function: f_Ca = @(t) f(t) is a time dependent function for the calcium input
    %           default: defaults to the constant resting concentration of 0.1 uM

    % Outputs:
    %   - dxdt: time derivate of the state vector

    % Default inputs
    if nargin < 4
        f_Ca = @(t) 0.1;
    end
    if nargin < 3 % nominal parameters
        % p = [k1, k2, k3, k4, k5, k6, k7, k8, c1, c2, c3, c4, Km1, Km2, Km3, Km4, Km5, K0, P0, Ktot, Ptot, Atot]
        p = [2, 15, 1, 120, 2, 15, 1, 80, 1, 1, 6, 8, 10, 0.3, 4, 10, 1, 0.5, 0.5, 20, 20, 1]; 
    end

    % Unpack parameter vector:
    k1 = p(1); k2 = p(2); k3 = p(3); k4 = p(4); k5 = p(5); k6 = p(6); k7 = p(7); k8 = p(8);
    c1 = p(9); c2 = p(10); c3 = p(11); c4 = p(12);
    Km1 = p(13); Km2 = p(14); Km3 = p(15); Km4 = p(16); Km5 = p(17);
    K0 = p(18); P0 = p(19); Ktot = p(20); Ptot = p(21); Atot = p(22);

    % Unpack states
    pK = x(1); P = x(2); A = x(3);

    % Conservation relations
    K = Ktot - pK; % kinase
    pP = Ptot - P; % phosphatase
    Aint = Atot - A; % AMPAR

    % Calcium level
    Ca = f_Ca(t);

    % Kinase ODE
    dpKdt = k1*pK*(K/(Km1 + K)) - k2*(P+P0)*(pK/(Km2+pK)) + k3*K0 + k4*K*((Ca^4)/((Km3^4) + (Ca^4)));

    % Phosphatase ODE
    dPdt = k5*P*(pP/(Km4+pP)) - k6*(pK+K0)*(P/(Km5+P)) + k7*P0 + k8*pP*((Ca^3)/((Km3^3) + (Ca^3)));

    % AMPAR ODE
    k21 = c1*pK + c3;
    k22 = c2*P + c4;
    dAdt = k21*Aint - k22*A;
    
    % collect rates
    dxdt = zeros(3,1);
    dxdt(1) = dpKdt; dxdt(2) = dPdt; dxdt(3) = dAdt;
end

% % INPUT FOR SIAN %%
% dx1/dt=(k1*x1*((Ktot - x1)/(Km1+(Ktot - x1))))-(k2*(x2+P0)*(x1/(Km2+x1)))+(k3*K0)+(k4*(Ktot - x1)*((u(t)^4)/((u(t)^4)+(Km3^4))));
% dx2/dt=(k5*x2*((Ptot - x2)/(Km4+(Ptot - x2))))-(k6*(x1+K0)*(x2/(Km5+x2)))+(k7*P0)+(k8*(Ptot - x2)*((u(t)^3)/((u(t)^3))+(Km3^3)));
% dx3/dt= (c1*x1 + c3)*(Atot - x3) - (c2*x2 + c4)*x3;
% y1=x2;
% y2=x2;
% y3=x3

