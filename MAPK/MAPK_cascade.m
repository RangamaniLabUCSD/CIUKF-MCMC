function dxdt = MAPK_cascade(x, p)
% MAPK_CASCASE ODEs for a three tier MAPK signalling cascade
%   dxdt = MAPK_cascade(x, p)
%
% This function provides the ODEs for a three tier MAPK signalling cascade presented in
%   DYVIPC: an integrated analysis and visualization framework to probe multi-dimensional biological networks
%   Nguyen et al. Sci Rep. 2015

% The system has three species (pS1, pS2, pS3) = (x(1), x(2), x(3)) 
% that can be in unphosphorylated or phosporylated states. The state variables are the 
% phosporylated version of the thress species and the unphosphorylated concentrations are fixed. It is constructed such that 
%   there are both positive and negative feedback loops.

% This system is multistable such that different parameter choices lead to different dynamis, e.g. osillaitons
%   different fixed points, etc... There are 14 model parameters. 

% The authors founds that different parameter combinations produce different model behaviors:
% Osciallations: S1t = 100, S2t = 100, S3t = 100, k1 = 0.1, k2 = 0.01, k3 = 0.01, k4 = 0.01, k5 = 0.01,
%                k6 = 0.01, n1 = 10, K1 = 1, n2 = 15, K2 = 8, α = 10
% Bistable fixed point: S1t = 0.22, S2t = 10, S3t = 53, k1 = 0.0012, k2 = 0.006, k3 = 0.049, k4 = 0.084, 
%                k5 = 0.043, k6 = 0.066, n1 = 5, K1 = 9.5, n2 = 10, K2 = 15, α = 95
% Mixed bistability: S1t = 20, S2t = 50, S3t = 30, k1 = 0.001, k2 = 0.08, k3 = 0.001, k4 = 0.08, 
%                k5 = 0.001, k6 = 0.05, n1 = 10, K1 = 0.66, n2 = 5, K2 = 0.8, α = 96

% We chosed the default parameters to be those that produce oscillations. The parameter vector is defined as:
%   p = [S1t, S2t, S3t, k1, k2, k3, k4, k5, k6, n1, K1, n2, K2, alpha]

    if nargin < 2 % No specied params? Use defaults to produce osciallatory behavior
        p = [100, 100,100, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 10, 1, 15, 8, 10];
    end
    
    % Unpack parameters for clarity
    S1t   =   p(1);
    S2t   =   p(2);
    S3t   =   p(3);
    k1    =   p(4);
    k2    =   p(5);
    k3    =   p(6);
    k4    =   p(7);
    k5    =   p(8);
    k6    =   p(9);
    n1    =   p(10);
    K1    =   p(11);
    n2    =   p(12);
    K2    =   p(13);
    alph =   p(14);

    % conservation law
    S1 = S1t - x(1);
    S2 = S2t - x(2);
    S3 = S3t - x(3);

    % Fluxes
    v1 = k1 * S1 * ((K1^n1) / (K1^n1 + x(3)^n1));
    v2 = k2 * x(1);
    v3 = k3 * S2 * x(1) * (1 + ((alph * x(3)^n2) / (K2^n2 + x(3)^n2)));
    v4 = k4 * x(2);
    v5 = k5 * S3 * x(2);
    v6 = k6 * x(3);

    % differential equations
    dxdt = zeros(3,1);
    dxdt(1) = v1 - v2;
    dxdt(2) = v3 - v4;
    dxdt(3) = v5 - v6;
end