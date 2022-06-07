function Jac = phosphatase_kinase_Jacobian(t, x, p, f_Ca)
    % This function evaluates the Jacobian of th RHS of ODE for the coupled Phosphatase-Kinase system
    %   that is modeled in Pi and Lisman 2008. J Neusoci

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

    Ca_term1 = (Ca^4) / (Km3^4 + Ca^4);
    Ca_term2 = (Ca^3) / (Km3^3 + Ca^3);

    % Non-zero Jacobian entries
    J11 = ((k1*(Km1*(Ktot-2*pK) - K^2)) / ((Km1 + K)^2)) - ((Km2*k2*(P+P0)) / ((Km2 + pK)^2)) - (k4*Ca_term1);
    J12 = (-k2*pK) / (Km2 + pK);

    J21 = (-k6*P) / (Km5 + P);
    J22 = ((-k5*P) / (Km4 + pP)) + ((k5*P*pP) / ((Km4 + pP)^2)) - ((k6*(pK + K0)*Km5) / ((Km5 + P)^2)) - (k8*Ca_term2);
    
    J31 = c1*Aint;
    J32 = -c2*A;
    J33 = -c1*pK - c3 - c2*P - c4;

    Jac = [J11, J12, 0;
           J21, J22, 0;
           J31, J32, J33];
end