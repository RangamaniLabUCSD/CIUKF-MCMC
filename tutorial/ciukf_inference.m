% Nathaniel Linden
% UCSD MAE
% Script for the CIUKF-MCMC tutorial
% This script performs MCMC with CIUKF-MCMC

clear all; close all; clc
addpath('../utils/')   % add paths to utilities and the MAPK model
addpath('../MAPK/')
savedir = './figures/';   % define path to save outputs
rng(100,'twister') % set random number seed for reproducibility
uqlab

% Set up overhead for simulating the model
% This include time limites, time resolution, and the functions with the ODEs and Jacobian matrix

% timing
t0 = 0; tend = 1800;
dt = 60; DT = 20;   % define two time resolutions, fine and coarse
t = t0:dt:tend; tfine = t0:DT:tend;

% parameters and initial conditions
theta_bistable = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];
x0  = [0.0015; 3.6678; 28.7307]; % low

freeParamIndex = [5, 7, 8, 9]; % estimate k2, k4, k5, k6
fixedParamIndex = setdiff(1:numel(theta_bistable), freeParamIndex);
ptrue = theta_bistable(freeParamIndex);

% Bounds of parameter values
paramBounds = [0 100; 0 100; 0 100; 0 0.1; 0 0.1; 0 0.05; 0 0.1; 0 0.05; 0 0.1; 5 10; 0 10; 5 10; 0 20; 0 100];

bounds = paramBounds(freeParamIndex,:);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramNames = paramNames(freeParamIndex);
state_names = {'x1', 'x2', 'x3'};

% load noisy data from demoModel.m
y = load('data.mat');

%% Setup for CIUKF
% dynamics model
d = 3; % dimension of the state
P0 = 1e-16*eye(d); % known initial condition, but must be non-zero for UKF
pdyn = 4; pvar = 2*d;
Q = @(theta) diag(theta(end-5:end-3)); % process noise

% measurement model
H = eye(d); h = @(x, theta) H*x; % identity measurements
m = size(H,1);
R = @(theta) diag(theta(end-2:end)); % measurement nosie theta(end) is the cov

% Model overhead
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, theta_bistable);
MAPK = @(x, theta) MAPK_cascade(x, thetaFull(theta));
Jac = @(x, theta) MAPK_Jacobian(x, thetaFull(theta)); % use analytical (computed by hand) Jacobian for stability
odeOptsTrue = odeset('Jacobian', @(t, x) Jac(x,ptrue));

% discrete time dynamics operator (psi) using ode15s()
f = @(k, x, theta)propf(x,@(x) MAPK(x,theta), dt, k, @(t, x) Jac(x,theta));

% CIUKF parameters
alph = 1e-3; beta = 1; kappa = 0;
eps = 1e-10;

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

% compute std of data
data_std = std(y.data);
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
logLikelihood = @(theta, y) ciukflp_quadProg(theta, x0, P0, f, H,...
    Q, R, y, alph,beta,kappa,eps, optOptions);

% control the number of workers for the parallel for loop
if isempty(gcp('nocreate'))
   parpool(4);
end

% Data into UQLab STRUCT
data.y = y.data; % Noisy synthetic data
data.Name = 'Noisy Measurements';

%% RUN CIUKF-MCMC WITH AIES %%
% Inverse problem solver
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AIES'; % Affine Invariant Ensemble Algorithm

% fewer chains and steps for demo
Solver.MCMC.NChains = 20; % Number of chains for AIES
Solver.MCMC.Steps = 35; % Steps per chain

% Uncoment to reproduce results in the manuscript
% Warning: This computation will take upwards of 24hrs
% Solver.MCMC.NChains = 150; % Number of chains for AIES
% Solver.MCMC.Steps = 7000; % Steps per chain

% STRUCT for the UQLab 
BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'MAPK Example';
BayesOpts.Prior = priorDistribution;
BayesOpts.Data = data;
BayesOpts.LogLikelihood = logLikelihood;
BayesOpts.Solver = Solver;

%% RUN QULab code
BayesianAnalysis = uq_createAnalysis(BayesOpts);

%% Plot the traces of MCMC trajectories
% use uq_display(); this plots each parameter in its own figure
close all
uq_display(BayesianAnalysis, 'trace','all')

saveas(1, [savedir, paramNames{1}, '.png'])
saveas(2, [savedir, paramNames{2}, '.png'])
saveas(3, [savedir, paramNames{3}, '.png'])
saveas(4, [savedir, paramNames{4}, '.png'])

%% IACT analysis
IACT = computeIACT(BayesianAnalysis.Results.Sample);
burnIn = floor(5*IACT);
uq_postProcessInversionMCMC(BayesianAnalysis, 'burnIn', burnIn, 'pointEstimate', 'MAP');

save([savedir, 'CIUKF_MCMC_results.mat'], 'BayesianAnalysis')