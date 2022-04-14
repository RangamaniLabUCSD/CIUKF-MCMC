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
paramNamesTex = {'$S_{1t}$','$S_{2t}$','$S_{3t}$','$k_1$', '$k_2$','$k_3$','$k_4$',...
        '$k_5$','$k_6$','$n_1$','$K_1$','$n_2$','$K_2$','$\alpha$'};
paramNames = paramNames(freeParamIndex);
paramNamesTex = paramNamesTex(freeParamIndex);
state_names = {'x1', 'x2', 'x3'};

x0low   = [0.0015; 3.6678; 28.7307]; % IC for low SS

thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);
MAPK = @(x, theta) MAPK_cascade(x, thetaFull(theta));
Jac = @(x, theta) MAPK_Jacobian(x, thetaFull(theta));

MAPK_col = @(x, theta) MAPK_cascade(x, thetaFull(theta'));

odeOptsTrue = odeset('Jacobian', @(t, x) Jac(x,ptrue));
[~, yTruehigh] = ode15s(@(t,x) MAPK(x, ptrue), tfine, x0low, odeOptsTrue);


% load the MCMC samples and all of the info on the run
run1 = load([savedir, 'results_LOWSS_noisy1.mat']);
run2 = load([savedir, 'results_LOWSS_noisy2.mat']);

% process samples
postSamples_1 = run1.BayesianAnalysis.Results.Sample;
postSamples_2 = run2.BayesianAnalysis.Results.Sample;

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
        posteriorSamples2d, tfine, x0low, nsampls);
[xprior, index] = runEnsembleSimulation(@(t, x,theta) MAPK_col(x, theta),  @(t,x,theta) Jac(x,theta),...
        run1.BayesianAnalysis.Results.PostProc.PriorSample', tfine, x0low, 1000);

% dynanics post w/ 95% credible
filename = 'LOWSS_xpost_';
plotDynamicsPosterior(posteriorSamples2d, nsampls, t, tfine, run1.BayesianAnalysis.Data.y, x0low, @(t, x,theta) MAPK_col(x, theta), ...
        savedir, filename, ptrue', thetaMap,thetaMean, [1,1,1], state_names, [0.025, 0.975], xpost);
        
% parameter 1D marginal posteriors
fileprefix = 'lowSS_param_';
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
plotEnsemble(posteriorSamplesNoBurn, savedir, 'lowSS_MCMC_', names, 10, burnin)