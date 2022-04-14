% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../')
addpath('../../../')
addpath('../../BayesID-master/utils')
plottingPreferencesNJL;
% Add model
addpath('../models/')

% folder to save results
savedir = './MAPK/BISTABLE/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

% UKF parameters
alph = 1e-3; beta = 1; kappa = 0;
eps = 1e-10;

% time 
tend = 1800; %tend = 3600; 
t0 = 0; dt = 60;
t = t0:dt:tend; DT = 10;
tfine = t0:DT:tend;
if dt < DT, DT = dt; end

% dynamics model
d = 3; % dimension of the state
P0 = 1e-16*eye(d); % known initial condition, but must be non-zero for UKF
pdyn = 4; pvar = 2*d;

% measurement model
H = eye(d); h = @(x, theta) H*x; % identity measurements
m = size(H,1);

Q = @(theta) diag(theta(end-5:end-3)); % process noise
R = @(theta) diag(theta(end-2:end)); % measurement nosie theta(end) is the cov

% MODEL PARAMETERS
% Bistable fixed point: S1t = 0.22, S2t = 10, S3t = 53, k1 = 0.0012, k2 = 0.006, k3 = 0.049, k4 = 0.084, 
%   k5 = 0.043, k6 = 0.066, n1 = 5, K1 = 9.5, n2 = 10, K2 = 15, α = 95
ptrueFull = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];
freeParamIndex = [5, 7, 8, 9]; % estimate k2, k4, k5, k6
fixedParamIndex = setdiff(1:numel(ptrueFull), freeParamIndex);
ptrue = ptrueFull(freeParamIndex);

% Bounds of parameter values
paramBounds = [0 100; 0 100; 0 100; 0 0.1; 0 0.1; 0 0.05; 0 0.1; 0 0.05; 0 0.1; 5 10; 0 10; 5 10; 0 20; 0 100];
bounds = paramBounds(freeParamIndex,:);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramNames = paramNames(freeParamIndex);
state_names = {'x1', 'x2', 'x3'};

x0low   = [0.0015; 3.6678; 28.7307]; % IC for low SS

thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);
MAPK = @(x, theta) MAPK_cascade(x, thetaFull(theta));
Jac = @(x, theta) MAPK_Jacobian(x, thetaFull(theta));

MAPK_col = @(x, theta) MAPK_cascade(x, thetaFull(theta'));

odeOptsTrue = odeset('Jacobian', @(t, x) Jac(x,ptrue));
[~, yTruelow] = ode15s(@(t,x) MAPK(x, ptrue), tfine, x0low, odeOptsTrue);

% discrete time dynamics operator (psi)
f = @(k, x, theta)propf(x,@(x) MAPK(x,theta), dt, k, @(t, x) Jac(x,theta));

%%%% LOW STEADY STATE! %%%%%
% Initialize noise
% original multiplier for results_LOWSS_run1.mat
% c = 0.1; % multiplier
% high noise multiplier for results_LOWSS_NOISY1.mat
c = 0.5; % multiplier
sigmaR = c*std(yTruelow)'; % measurement noise

% Generate noisy samples
y = generateData(@(x) MAPK(x, ptrue), x0low, t, H, sigmaR, odeOptsTrue); % Use ode45 to solve the system and get 'training data'
y=abs(y); % if less than zero, set to abs values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STRUCTS FOR UQLAB %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Priors
priorOptions.Name = 'Prior distribution MAPK';

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

% Custom logLikelihood
optOptions = optimoptions('quadprog', 'Display','off', 'Algorithm', 'trust-region-reflective'); % options for optimizer
logLikelihood = @(theta, y) ciukflp_quadProg(theta, x0low, P0, f, H,...
    Q, R, y, alph,beta,kappa,eps, optOptions);

if isempty(gcp('nocreate'))
   parpool(6);
end

% Data into UQLab STRUCT
data.y = y; % Noisy synthetic data
data.Name = 'Noisy Measurements';

% RUN WITH AIES %
%%%% FIRST RUN %%%%%
% Inverse problem solver
Solver.Type = 'MCMC'; % Markov Chain Monte Carlo Solver
Solver.MCMC.Sampler = 'AIES'; % Affine Invariant Ensemble Algorithm
Solver.MCMC.NChains = 150; % Number of chains for AIES
Solver.MCMC.Steps = 3500; % Steps per chain

% Everything in a STRUCT for the software
BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'MAPK LOWSS';
BayesOpts.Prior = priorDistribution;
BayesOpts.Data = data;
BayesOpts.LogLikelihood = logLikelihood;
BayesOpts.Solver = Solver;

% % RUN QULab code
BayesianAnalysis = uq_createAnalysis(BayesOpts);
save([savedir, 'results_LOWSS_noisy1.mat'], 'BayesianAnalysis')

% RUN WITH AIES %
%%%% ADDITIONAL RUN %%%%%
% load previous run
prev = load([savedir, 'results_AIES_run1.mat']);
lastPoints = prev.BayesianAnalysis.Results.Sample(end,:,:);

% Inverse problem solver
Solver.Type = 'MCMC'; % Markov Chain Monte Carlo Solver
Solver.MCMC.Sampler = 'AIES'; % Affine Invariant Ensemble Algorithm
Solver.MCMC.NChains = 150; % Number of chains for AIES
Solver.MCMC.Steps = 1500; % Steps per chain

% Restart from last run
Solver.Seed = lastPoints;

% Everything in a STRUCT for the software
BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'MAPK';
BayesOpts.Prior = priorDistribution;
BayesOpts.Data = prev.BayesianAnalysis.Data; % ensure we use exactly the same data as the previous run
BayesOpts.LogLikelihood = logLikelihood;
BayesOpts.Solver = Solver;

% RUN QULab code
BayesianAnalysis = uq_createAnalysis(BayesOpts);
save([savedir, 'results_AIES_noisy2.mat'], 'BayesianAnalysis')