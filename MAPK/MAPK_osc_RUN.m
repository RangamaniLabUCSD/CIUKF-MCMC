% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;


% folder to save results
savedir = './MAPK_osc/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

% UKF parameters
alph = 1e-3; beta = 1; kappa = 0;
eps = 1e-10;

% time 
tend = 3600; 
t0 = 0; dt = 120;
t = t0:dt:tend; 
DT = 30; % finer time scale for plotting
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
ptrueFull = [100,100,100, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 10, 1, 15, 8, 10];
freeParamIndex = [5, 6, 7, 8]; % estimate k2, k3, k4, k5
fixedParamIndex = setdiff(1:numel(ptrueFull), freeParamIndex);
ptrue = ptrueFull(freeParamIndex);

% Bounds of parameter values
paramBounds = [0 100; 0 100; 0 100; 0 0.1; 0 0.1; 0 0.05; 0 0.1; 1e-5 0.05; 0 0.1; 5 10; 0 10; 5 10; 0 20; 0 100];
bounds = paramBounds(freeParamIndex,:);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramNames = paramNames(freeParamIndex);
state_names = {'x1', 'x2', 'x3'};
ptrue = ptrueFull(freeParamIndex);
x0 = [10; 80; 80];

% Functions for the ODE and Jacobian
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);
MAPK = @(x, theta) MAPK_cascade(x, thetaFull(theta));
Jac = @(x, theta) MAPK_Jacobian(x, thetaFull(theta));

[~, yTrue] = ode15s(@(t,x) MAPK(x, ptrue), tfine, x0);

% discrete time dynamics operator (psi)
f = @(k, x, theta)propf(x,@(t, x) MAPK(x,theta), dt, k, @(t, x) Jac(x,theta));

% Initialize noise
c = 0.1; % multiplier
sigmaR = c*std(yTrue)'; % measurement noise

% Generate noisy samples
y = generateData(@(x) MAPK(x, ptrue), x0, t, H, sigmaR); % Use ode45 to solve the system and get 'training data'
y=abs(y); % if less than zero, set to abs values

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

% Custom logLikelihood
optOptions = optimoptions('quadprog', 'Display','off', 'Algorithm', 'trust-region-reflective'); % options for optimizer
logLikelihood = @(theta, y) ciukflp_quadProg(theta, x0, P0, f, H,...
    Q, R, y, alph,beta,kappa,eps, optOptions);

% Run with multiple threads
% Change size of parpool acording to the machine
if isempty(gcp('nocreate'))
   parpool(6)
end

% Data into UQLab STRUCT
data.y = y; % Noisy synthetic data
data.Name = 'Noisy Measurements';

% RUN WITH AIES %
% Inverse problem solver
Solver.Type = 'MCMC'; % Markov Chain Monte Carlo Solver
Solver.MCMC.Sampler = 'AIES'; % Affine Invariant Ensemble Algorithm
Solver.MCMC.NChains = 150; % Number of chains for AIES
Solver.MCMC.Steps = 6000; % Steps per chain

% Everything in a STRUCT for the software
BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'MAPK';
BayesOpts.Prior = priorDistribution;
BayesOpts.Data = data;
BayesOpts.LogLikelihood = logLikelihood;
BayesOpts.Solver = Solver;

% RUN QULab code
BayesianAnalysis = uq_createAnalysis(BayesOpts);
save([savedir, 'results_lessData_run1.mat'], 'BayesianAnalysis')

%%%% Functions %%%%
function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode15s(@(t,x)f(x), t, x0);
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