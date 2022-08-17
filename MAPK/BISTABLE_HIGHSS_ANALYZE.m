% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;

% folder to save results
savedir = './BISTABLE/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

% UKF parameters
alph = 1e-3; beta = 1; kappa = 0;
eps = 1e-10;

% time 
tend = 1800;
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
ptrueFull = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 9.5, 5, 15, 10, 95];
freeParamIndex = [5, 7, 8, 9]; % estimate k2, k4, k5, k6
fixedParamIndex = setdiff(1:numel(ptrueFull), freeParamIndex);
ptrue = ptrueFull(freeParamIndex);

% Bounds of parameter values
paramBounds = [0 100; 0 100; 0 100; 0 0.1; 0 0.1; 0 0.05; 0 0.1; 0 0.05; 0 0.1; 0 10; 5 10; 0 20; 5 10; 0 100];
bounds = paramBounds(freeParamIndex,:);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','K1','n1','K2','n2','alpha'};
paramNames = paramNames(freeParamIndex);
state_names = {'x1', 'x2', 'x3'};

x0high  = [0.1245; 2.4870; 31.2623]; % IC for high SS

thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);
MAPK = @(t, x, theta) MAPK(t, x, thetaFull(theta));
Jac = @(t, x, theta) MAPK_Jacobian(t, x, thetaFull(theta));

MAPK_col = @(t, x, theta) MAPK(t, x, thetaFull(theta'));

odeOptsTrue = odeset('Jacobian', @(t, x) Jac(t, x,ptrue));
[~, yTruehigh] = ode15s(@(t,x) MAPK(t, x, ptrue), tfine, x0high, odeOptsTrue);


% load the MCMC samples and all of the info on the run
run1 = load([savedir, 'results_HIGHSS_noisy1.mat']);
run2 = []; %load([savedir, 'results_HIGHSS_noisy2.mat']); % only one run 


% process samples
postSamples_1 = run1.BayesianAnalysis.Results.Sample;
postSamples_2 = [];% run2.BayesianAnalysis.Results.Sample;

fprintf('IACT with no burn in for the first run: \n');
IACTfull_run1 = computeIACT(postSamples_1);
fprintf([num2str(IACTfull_run1), '\n']);

% compute burnin as a multiple (5-10x) of the ICAT
burnin = floor(IACTfull_run1*7);

fprintf(['IACT with ', num2str(burnin), ' samples of burn-in for the first run: \n'])
fprintf([num2str(computeIACT(postSamples_1(burnin:end,:,:))), '\n']);

fprintf('IACT for the second run: \n')
fprintf([num2str(computeIACT(postSamples_2)), '\n']);

% concatenate converged samples from run 1 and run 2
posteriorSamples = cat(1, postSamples_1(burnin:end,:,:), postSamples_2);
posteriorSamplesNoBurn = cat(1, postSamples_1, postSamples_2);

fprintf('IACT for all posterior samples: \n')
fprintf([num2str(computeIACT(posteriorSamples)), '\n']);

% flatten ensemble into 2D matrix
posteriorSamples2d = flattenEnsemble(posteriorSamples);

% approximate MAP and MEAN by fitting a kernel density estimator
[thetaMap, thetaMean, fits, x_fit] = approximate_MAP(posteriorSamples2d, bounds, 0.001);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting posteiors and priors %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nsampls = 30000; % number of samples to use for plotting
[xpost, index] = runEnsembleSimulation(@(t, x,theta) MAPK_col(x, theta), @(t,x,theta) Jac(x,theta), ...
        posteriorSamples2d, tfine, x0high, nsampls);
[xprior, index] = runEnsembleSimulation(@(t, x,theta) MAPK_col(x, theta),  @(t,x,theta) Jac(x,theta),...
        run1.BayesianAnalysis.Results.PostProc.PriorSample', tfine, x0high, 1000);

% dynanics post w/ 95% credible
filename = 'HIGHSS_xpost_';
plotDynamicsPosterior(posteriorSamples2d, nsampls, t, tfine, run1.BayesianAnalysis.Data.y, x0high, @(t, x,theta) MAPK_col(x, theta), ...
        savedir, filename, ptrue', thetaMap,thetaMean, [1,1,1], state_names, [0.025, 0.975], xpost);

% parameter 1D marginal posteriors
fileprefix = 'HIGHSS_param_';
priors = {};
for i = 1:numel(paramNames)
    priors{i} = makedist('Uniform',  bounds(i,1), bounds(i,2));  
end
plotMarginalPosterior(posteriorSamples2d, ptrue, paramNames, paramNamesTex, savedir,fileprefix, thetaMap, thetaMean, fits, x_fit, priors, bounds)
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting traces of the Ensembles %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
names = paramNames; names{5}='Q1'; names{6}='Q2'; names{7}='Q3';
names{8}='R1';names{9}='R2';names{10}='R3';
plotEnsemble(posteriorSamplesNoBurn, savedir, 'highSS_MCMC_', names, 10, burnin)