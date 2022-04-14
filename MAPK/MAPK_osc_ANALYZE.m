% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')

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
tend = 5400;
t0 = 0; dt = 120; DT = 20;
t = t0:dt:tend;
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


% load the MCMC samples and all of the info on the run
run1 = load([savedir, 'results_lessData_run1.mat']);

% process samples
postSamples_1 = run1.BayesianAnalysis.Results.Sample;

fprintf('IACT with no burn in for the first run: \n');
IACTfull_run1 = computeIACT(postSamples_1);
fprintf([num2str(IACTfull_run1), '\n']);

% compute burnin as a multiple (5-10x) of the ICAT
burnin = floor(IACTfull_run1*7);

% concatenate converged samples from run 1
posteriorSamples = postSamples_1(burnin:end,:,:);
posteriorSamplesNoBurn =  postSamples_1;

fprintf('IACT for all posterior samples: \n')
fprintf([num2str(computeIACT(posteriorSamples)), '\n']);

% flatten ensemble into 2D matrix
posteriorSamples2d = flattenEnsemble(posteriorSamples);

% approximate MAP and MEAN by fitting a kernel density estimator
[thetaMap, thetaMean, fits, x_fit] = approximate_MAP(posteriorSamples2d, bounds, 0.001);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting posteiors and priors %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = [run1.BayesianAnalysis.Data.y, nan*ones(3,15)];
nsampls = 30000; % number of samples to use for plotting
[xpost, index] = runEnsembleSimulation(@(t, x,theta) MAPK(x, theta), @(t,x,theta) Jac(x, theta), posteriorSamples2d, tfine, x0, nsampls);
[xprior, index] = runEnsembleSimulation(@(t, x,theta) MAPK(x, theta),  @(t,x,theta) Jac(x, theta),...
        run1.BayesianAnalysis.Results.PostProc.PriorSample', tfine, x0, 1000);

% dynanics post w/ 95% credible
filename = 'xpost';
plotDynamicsPosterior(posteriorSamples2d, nsampls, t, tfine, y, x0, @(t, x,theta) MAPK(x, theta), ...
        savedir, filename, ptrue', thetaMap,thetaMean, [1,1,1], state_names, [0.025, 0.975], xpost);


% parameter 1D marginal posteriors
filename = ['param_'];
priors = {};
for i = 1:numel(paramNames)
    priors{i} = makedist('Uniform',  bounds(i,1), bounds(i,2));  
end
plotMarginalPosterior(posteriorSamples2d, ptrue, paramNames, paramNames, savedir, filename, thetaMap, thetaMean, fits, x_fit, priors, bounds)
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot traces of the Ensembles %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
names = paramNames; names{5}='Q1'; names{6}='Q2'; names{7}='Q3';
names{8}='R1';names{9}='R2';names{10}='R3';
plotEnsemble(posteriorSamplesNoBurn, savedir, 'osc_MCMC_', names, 10, burnin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot sample traces of x3(t) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nsamp = 50;
downsamp = 1;
figure(1)
for i = 1:nsamp
        figure(1); plot(tfine(1:downsamp:end),  xpost(3,1:downsamp:end,i), ...
                'Color', [0,0,1,0.1], 'LineWidth', 1); hold on
end
plot(tfine(1:downsamp:end), yTrue(1:downsamp:end,3), 'g', 'LineWidth',2)

fname = [savedir, 'x3samples.tex'];
datPath = [savedir,'x3samples_data/'];
relDatPath = 'x3samples_data';
saveas(gcf, [savedir, 'x3samples.png'])
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot histograms of the Limit Cycle features %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
startIdx = 50;
peakThreshold = 17;
LCAthresh = 1.0;
Decaythresh = 5.0;

LCAs = zeros(nsampls,1);
periods = zeros(nsampls,1);
parfor i = 1:nsampls    
    [LCAs(i), periods(i)] = limitCycleFeatures(xpost(3,:,i), startIdx, peakThreshold, LCAthresh, Decaythresh, DT, nan);
end
[trueLCA, truePeriod] = limitCycleFeatures(yTrue(:,3), startIdx, peakThreshold, LCAthresh, Decaythresh, DT, nan);
% states of limit cycles
percentOsc = sum(~isnan(LCAs))/nsampls;
percentDecay = sum(isnan(LCAs))/nsampls;

figure;
bar([percentOsc, percentDecay])
fname = [savedir, 'oscStats.tex'];
datPath = [savedir,'oscStats_data/'];
relDatPath = 'oscStats_data';
saveas(gcf, [savedir, 'oscStates.png'])
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);

figure; 
histogram(LCAs, 'Normalization','Probability');
ax = gca; hold on
plot([trueLCA trueLCA], ax.YLim, 'b--')
xlabel('Limit Cycle Amplitude'); 
ylabel('Probability')

fname = [savedir, 'LCA.tex'];
datPath = [savedir,'LCA_data/'];
relDatPath = 'LCA_data';
saveas(gcf, [savedir, 'LCA.png'])
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

figure; 
histogram(periods,  'Normalization','Probability');
ax = gca; hold on
plot([truePeriod truePeriod], ax.YLim, 'b--')
xlabel('Limit Cycle Period'); 
ylabel('Probability')

fname = [savedir, 'period.tex'];
datPath = [savedir,'period_data/'];
relDatPath = 'period_data';
saveas(gcf, [savedir, 'period.png'])
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all