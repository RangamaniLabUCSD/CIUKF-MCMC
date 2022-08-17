% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;


% folder to save results
savedir = './LTP/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL SETUP and OVERHEAD %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time 
tend = 9;
t0 = 0;
dt = 0.25;
t = t0:dt:tend; 
DT = 0.01; tfine = t0:DT:tend;
if dt < DT, DT = dt; end

% MODEL PARAMETERS
ptrueFull = [2, 15, 1, 120, 2, 15, 1, 80, 1, 1, 6, 8, 10, 0.3, 4, 10, 1, 0.5, 0.5, 20, 1, 20];

paramNames = {'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'c1', 'c2', 'c3', 'c4', 'Km1', ...
        'Km2', 'Km3', 'Km4', 'Km5', 'K0', 'P0', 'Ktot', 'Atot', 'Ptot'};
paramNamesTex = {'$k_1$', '$k_2$', '$k_3$', '$k_4$', '$k_5$', '$k_6$', '$k_7$', '$k_8$', ...
        '$c_1$', '$c_2$', '$c_3$', '$c_4$', '$K_{m1}$', '$K_{m2}$', '$K_{m3}$', '$K_{m4}$',...
        '$K_{m5}$', '$K_0$', '$P_0$', 'K_{tot}', 'A_{tot}', 'P_{tot}'};
state_names = {'x1', 'x2', 'x3'};

% Fix subset of params based on sensitivity analysis
% Fixed params include:
%   - Rates: k1, k3, k4, k5, k7, k8
%   - Km1, Km2, Km4, Km5
%   - c1, c2, c3, c4
% Free params are:
%   - k2, k6, Km3, K0, P0, Ktot, Ptot, Atot 
fixedParamIndex = [1, 3, 4, 5, 7, 8, 9, 10, 13, 14, 16, 17];
freeParamIndex = setdiff(1:numel(ptrueFull), fixedParamIndex);
ptrue = ptrueFull(freeParamIndex);
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);

% Intial Condition
x0 = [0.0228, 0.0017, 0.4294]';

% Model dimensions
d = 3; m = 3;
pdyn = numel(ptrue);
pvar = 2*d; % # process noise + # meas noise

Ca_basal = 0.1; Ca_peak = 4; % calcium input
start = 1; stop = 3;
f_Ca = @(t) stepFunc(t, start, stop, Ca_basal, Ca_peak);

kinpho = @(t, x, theta) synaptic_plasticity(t, x, [thetaFull(theta), f_Ca(t)]);
Jac = @(t,x, theta) synaptic_plasticity_Jacobian(t, x, [thetaFull(theta), f_Ca(t)]);

odeOpts = odeset('Jacobian', @(t, x) Jac(t,x,theta));
[~, yTrue] = ode15s(@(t,x) kinpho(t,x,ptrue), tfine, x0);

% discrete time dynamics operator (psi)
f = @(k, x, theta) propf(x, @(t, x) kinpho(t,x,theta), dt, k, @(t,x) Jac(t,x,theta));
H = eye(d); h = @(x) H*x; % measurement operator

% noise cov matrices
Q = @(theta) diag(theta(end-(2*d-1):end-d)); % process noise
R = @(theta) diag(theta(end-(d-1):end)); % measurement nosie 
P0 = 1e-16*eye(d); % assume we have a known initial condition, but must be non-zero for UKF

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate noisy samples %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize noise
sigmaR = 0.1*std(yTrue)';
y = generateData(@(t, x) kinpho(t, x, ptrue), x0, t, H, sigmaR); % Use ode45 to solve the system and get 'training data'
y=abs(y); % if less than zero, set to abs values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETER BOUNDS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bounds set to be one order of magnitude smaller and 
%   one order of mag larger than the nominal values
bounds = zeros(pdyn, 2);
bounds(:,1) = 0.1*ptrue;
bounds(:,2) = 10*ptrue; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STRUCTS FOR UQLAB %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Priors
priorOptions.Name = 'Prior distribution MAPK No-Feedback';

% Model Parameters
for i = 1:pdyn
    priorOptions.Marginals(i).Name =       paramNames{i};
    priorOptions.Marginals(i).Type =       'Uniform';
    priorOptions.Marginals(i).Parameters = bounds(i,:);
end
data_std = std(y');

% Process Noise covariance terms
priorOptions.Marginals(pdyn+1).Name =       'Q1'; 
priorOptions.Marginals(pdyn+1).Type =       'Gaussian';
priorOptions.Marginals(pdyn+1).Parameters = [0, data_std(1)/3];
priorOptions.Marginals(pdyn+1).Bounds =     [0, data_std(1)];

priorOptions.Marginals(pdyn+2).Name =       'Q2';       
priorOptions.Marginals(pdyn+2).Type =       'Gaussian'; 
priorOptions.Marginals(pdyn+2).Parameters = [0, data_std(2)/3];    
priorOptions.Marginals(pdyn+2).Bounds =     [0, data_std(2)];   

priorOptions.Marginals(pdyn+3).Name =       'Q3'; 
priorOptions.Marginals(pdyn+3).Type =       'Gaussian';
priorOptions.Marginals(pdyn+3).Parameters = [0, data_std(3)/3];
priorOptions.Marginals(pdyn+3).Bounds =     [0, data_std(3)];

% Measurement Noise covariance term
priorOptions.Marginals(pdyn+4).Name =       'R1'; 
priorOptions.Marginals(pdyn+4).Type =       'Gaussian';
priorOptions.Marginals(pdyn+4).Parameters = [0, data_std(1)/3];
priorOptions.Marginals(pdyn+4).Bounds =     [0, data_std(1)];

priorOptions.Marginals(pdyn+5).Name =       'R2'; 
priorOptions.Marginals(pdyn+5).Type =       'Gaussian';
priorOptions.Marginals(pdyn+5).Parameters = [0, data_std(2)/3];
priorOptions.Marginals(pdyn+5).Bounds =     [0, data_std(2)];

priorOptions.Marginals(pdyn+6).Name =       'R3'; 
priorOptions.Marginals(pdyn+6).Type =       'Gaussian';
priorOptions.Marginals(pdyn+6).Parameters = [0, data_std(3)/3];
priorOptions.Marginals(pdyn+6).Bounds =     [0, data_std(3)];

% Create prior dist input
priorDistribution = uq_createInput(priorOptions);

% UKF parameters
alph = 1e-3; beta = 1; kappa = 0;
eps = 1e-10;

% Custom logLikelihood
optOptions = optimoptions('quadprog', 'Display','off', 'Algorithm', 'trust-region-reflective'); % options for optimizer
logLikelihood = @(theta, y) ciukflp_quadProg(theta, x0, P0, f, H,...
    Q, R, y, alph,beta,kappa,eps, optOptions);

if isempty(gcp('nocreate'))
   parpool(6)
end

% Data into UQLab STRUCT
data.y = y; % Noisy synthetic data
data.Name = 'Noisy Measurements';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN1 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RUN WITH AIES %
%Inverse problem solver
Solver.Type = 'MCMC'; % Markov Chain Monte Carlo Solver
Solver.MCMC.Sampler = 'AIES'; % Affine Invariant Ensemble Algorithm
Solver.MCMC.NChains = 150; % Number of chains for AIES
Solver.MCMC.Steps = 8000; % Steps per chain

% Everything in a STRUCT for the software
BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'Kinase Phosphatase Stochastic';
BayesOpts.Prior = priorDistribution;
BayesOpts.Data = data;
BayesOpts.LogLikelihood = logLikelihood;
BayesOpts.Solver = Solver;

BayesianAnalysis = uq_createAnalysis(BayesOpts);
save([savedir, 'results_synaptic_plasticity.mat'], 'BayesianAnalysis')

%%%% Functions %%%%
function thetaFull = fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull)
    thetaFull = zeros(size(ptrueFull));
    thetaFull(fixedParamIndex) = ptrueFull(fixedParamIndex);
    thetaFull(freeParamIndex) = theta(1:numel(freeParamIndex));
end

function Ca = stepFunc(t, tauOn, tauOff, low, high)
    Ca = low + (high-low)*heavyside(t-tauOn) - (high-low)*heavyside(t-tauOff);
end

function y = heavyside(x)
    y = 0*x;
    y(find(x>0)) = 1;
end

function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode15s(f, t, x0);
    y = H*x' + sigmaR.*randn(size(H*x'));
    y = y(:,2:end);
end


function xout = propf(xin, f, dt, k, jac)
    t = k*dt;
    xout = xin;
    odeOpts = odeset('Jacobian', jac);
    [~, xout] = ode15s(@(t,x)f(t, x), [t, t+dt], xout, odeOpts);
    xout = xout(end,:)';
end

function logpost = ciukflp_quadProg(theta, x0, P0, f, H, Q, R, y, alpha,beta,kappa,eps, optOptions)
    T = size(y,2);
    n = size(x0,1);

    lambda = 1;

    num_sets = size(theta, 1);
    logpost = zeros(num_sets, 1);

    parfor set = 1:num_sets
    % for set = 1:num_sets
        f_set = @(i,x) f(i, x, theta(set,:));
        Sigma = Q(theta(set,:));
        Gamma = R(theta(set,:));

        m = x0; C = P0;
        for i = 1:T
            [m, C,err, wICUT] = CIUKFpredict(m, C, Sigma, n, f_set, lambda, eps, i);
            % figure(1); plot(i, m(1), 'b.'); hold on
            if (err ~= 0)
                logpost(set) = -Inf;
                break;
            end
            [m,C,v,S,Sinv,err] = CIUKFupdate_quadProg(m, C, y(:,i), Gamma, n, H, lambda, wICUT, eps, alpha, beta, optOptions);
            % figure(1); plot(i, m(1), 'k.'); hold on
            if (err ~= 0)
                logpost(set) = -Inf;
                break;
            end
            logpost(set) = logpost(set) - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        end
    end
end
