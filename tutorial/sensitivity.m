% Nathaniel Linden
% UCSD MAE

% Script for the CIUKF-MCMC tutorial
% This script runs global sensitivity analysis of the MAPK model with UQLab
clear all; close all; clc

%% Problem set-up
addpath('../utils/'); addpath('../MAPK/');
% folder to save plots
savedir = './figures/';

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

% Time spacing
t0 = 0; tend = 5400;
dt = 60; DT = 20;
t = t0:dt:tend; 
tfine = t0:DT:tend;

% Specifications for MAPK oscilltions
theta_bistable = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];
x0   = [0.0015; 3.6678; 28.7307]; % low

% Fix non-identifiable params and exclude from analysis
fixedParamIndex = [1,2,3,10,11,12,13]; nparams = 14;
freeParamIndex = setdiff(1:nparams, fixedParamIndex);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramNamesTex = {'$S_{1t}$','$S_{2t}$','$S_{3t}$','$k_1$', '$k_2$','$k_3$','$k_4$','$k_5$','$k_6$','$n_1$','$K_1$','$n_2$','$K_2$','$\alpha$'};
paramNames = paramNames(freeParamIndex);
paramNamesTex = paramNamesTex(freeParamIndex);

paramBounds = [0 100; 0 100; 0 100; 0 0.1; 0 0.1; 0 0.05; 0 0.1; 0 0.05; 0 0.1; 5 10; 0 10; 5 10; 0 20; 0 100];

% Functions for the ODE/jacobian with some fixed params
thetaFull = @(theta, ptrueFull) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);
MAPK = @(x, theta, ptrueFull) MAPK_cascade(x, thetaFull(theta, ptrueFull));
Jac = @(x, theta, ptrueFull) MAPK_Jacobian(x, thetaFull(theta,ptrueFull));

% Struct for parameter options
IOopts.marginals = [];
for i = 1:numel(freeParamIndex)
    pidx = freeParamIndex(i);
    IOopts.marginals(i).Name       = paramNames{i};
    IOopts.marginals(i).Type       = 'Uniform';
    IOopts.marginals(i).Parameters = paramBounds(pidx,:);
end
% We want the lower bound for k5 to be slightly larger than 0 eg 1e-5
IOopts.marginals(5).Parameters(1) = 1e-5;
input = uq_createInput(IOopts); % create input structure

% options for Sobol sensitivity analysis
SobolSensOpts.Type = 'Sensitivity';
SobolSensOpts.Method = 'Sobol';
SobolSensOpts.Input = input;
SobolSensOpts.Sobol.Sampling = 'sobol';
SobolSensOpts.Sobol.SampleSize = 1000;

% Cell array of QoI functions
qoiFuncs = {@(sol) finalValQOI(sol, 1),@(sol) finalValQOI(sol, 2), @(sol) finalValQOI(sol, 3)};
qoiNames = {'x1-ss', 'x2-ss', 'x3-ss'};
% Function to eval MAPK model
Y = @(X) numericalModel(X, x0, @(t,x,theta) MAPK(x,theta, theta_bistable), t, ...
            @(t,x,theta) Jac(x,theta, theta_bistable), qoiFuncs);

% Pass everything to UQLab
modelOpts.mHandle = Y;
modelOpts.isVectorized = false;
Model = uq_createModel(modelOpts);

%% Run Global sensitivity analysis
% Warning this can take a long time!

fprintf('Running analysis for Oscillations\n');
Analysis       = SobolSensOpts;
Analysis.Model = Model;
SensitivtyResults = uq_createAnalysis(Analysis);

%% Analysis
% the function ../utils/plotGSAResults.m performs the analysis show here
% The following steps are repeated for each QoI:
%       1) Sort params based on sensitivity measure
%       2) Plot a bar chart
%       3) Choose a cutoff value
%       4) Find parameters above cutoff


% extract sensitivity measures
totalInd      = SensitivtyResults.Results.Total;
firstOrderInd = SensitivtyResults.Results.FirstOrder;

plottingPreferencesNJL;
figure('Renderer', 'painters', 'position',[0,0,1000,500]); % everything as subplots
numQoi = numel(qoiNames);

% loop over each QoI
for qoi = 1:numQoi
    % total order
    [sorted, idx] = sort(totalInd(:,qoi),1,'descend');
    subplot(2, numQoi, qoi); bar(sorted);
    xticks(1:numel(paramNames));
    xticklabels(paramNamesTex(idx));
    ylabel('Total Order Index');
    title(qoiNames{qoi})
   
    % first order
    [sorted, idx] = sort(firstOrderInd(:,qoi),1,'descend');
    subplot(2, numQoi, qoi+numQoi); bar(sorted);
    xticks(1:numel(paramNames));
    xticklabels(paramNamesTex(idx));
    ylabel('First Order Index');
    title(qoiNames{qoi})
end
saveas(gcf, [savedir, 'mapk_GSA.png'])
close all
%% Functions %%%%
function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode15s(@(t,x)f(x), t, x0);
    y = H*x' + sigmaR.*randn(size(H*x'));
    y = y(:,2:end);
end

function Y = numericalModel(theta, x0, f, t, jac, qoiFuncs)
    % We want a function of the form Y = M(theta) that maps from the inputs (parameters)
    %   to the output quantities of interest
    % This is a numerical model!
    % Inputs:
    % - theta:    the model inputs, in the notation of UQLab theta is X
    % - x0:       initial condition for the ODE
    % - f:        ODE function should be @f(t, x, theta)
    % - t:        time (range of vector) to solve ode
    % - jac:      function for jacobian should be @f(t, x, theta)
    % - qoiFuncs: cell array of function handles for the qoi functions
    %               qoi funcs must map from xout --> qoi
    %
    % Outputs:
    % - Y:    output vector of quanities of iterest

    f = @(t,x) f(t,x,theta);
    jac = @(t,x) jac(t,x,theta);
    sol = solve(x0, f, t, jac); % solve ODE using ode15s

    Y = zeros(1, numel(qoiFuncs));

    for i = 1:numel(qoiFuncs)
        Y(i) = feval(qoiFuncs{i}, sol);
	if isnan(Y(i))
		disp(i);
		disp(theta);
		disp(Y(i));
	end
    end
end

function xout = solve(x0, f, t, jac)
    % sovle the ODE system using ODE15s
    odeOpts = odeset('Jacobian', @(t,x) jac(t,x));
    [~, xout] = ode15s(@(t,x) f(t,x), t, x0, odeOpts);
end

function qoi = meanQOI(sol, colIdx, startIdx, endIdx)
    if nargin < 4 % endIdx = end
        qoi = mean(sol(startIdx:end,colIdx));
    else % endIdx is different than end
        qoi = mean(sol(startIdx:endIdx,colIdx));
    end
end

function qoi = lcaQOI(sol, colIdx, startIdx, DT)
    startIdx = 50;
    peakThreshold = 17;
    LCAthresh = 1.0;
    Decaythresh = 5.0;

    [qoi, ~] = limitCycleFeatures(sol(:,colIdx), startIdx, peakThreshold, LCAthresh, Decaythresh, DT, 0);
end

function qoi = periodQOI(sol, colIdx, startIdx, DT)
    startIdx = 50;
    peakThreshold = 17;
    LCAthresh = 1.0;
    Decaythresh = 5.0;

    [~, qoi] = limitCycleFeatures(sol(:,colIdx), startIdx, peakThreshold, LCAthresh, Decaythresh, DT, 0);
end

function qoi = finalValQOI(sol, colIdx)
    qoi = sol(end,colIdx);
end
