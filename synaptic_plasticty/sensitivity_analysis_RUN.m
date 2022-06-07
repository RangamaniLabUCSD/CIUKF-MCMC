% Nathaniel Linden
% UCSD MAE
% Script to run GSA for the synaptic plasticity model
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;

% folder to save results
savedir = './sensitivity_analysis/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

% time 
tend = 12;
t0 = 0;
dt = 0.25;
t = t0:dt:tend; 
DT = 0.01; 
tfine = t0:DT:tend;
if dt < DT, DT = dt; end

% MODEL PARAMETERS
ptrueFull = [2, 15, 1, 120, 2, 15, 1, 80, 1, 1, 6, 8, 10, 0.3, 4, 10, 1, 0.5, 0.5, 20, 20, 1]; 
paramNames = {'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'c1', 'c2', 'c3', 'c4', 'Km1', ...
        'Km2', 'Km3', 'Km4', 'Km5', 'K0', 'P0', 'Ktot', 'Ptot', 'Atot'};
fixedParamIndex = []; % no fixed params - all params are ID
freeParamIndex = setdiff(1:numel(ptrueFull), fixedParamIndex);
ptrue = ptrueFull(freeParamIndex);
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);

% Intial Condition
x0 = [0.0228, 0.0017, 0.4294]';

Ca_basal = 0.1; % calcium input
start = 3; stop = 5;
f_Ca_LTP = @(t) stepFunc(t, start, stop, Ca_basal, 4.0);
f_Ca_LTD = @(t) stepFunc(t, start, stop, Ca_basal, 2.2);

kinpho_LTP = @(t, x, theta) phosphatase_kinase(t, x, thetaFull(theta), f_Ca_LTP);
Jac_LTP = @(t,x, theta) phosphatase_kinase_Jacobian(t, x, thetaFull(theta), f_Ca_LTP);

kinpho_LTD = @(t, x, theta) phosphatase_kinase(t, x, thetaFull(theta), f_Ca_LTD);
Jac_LTD = @(t,x, theta) phosphatase_kinase_Jacobian(t, x, thetaFull(theta), f_Ca_LTD);

% Struct for parameter options
IOopts.marginals = [];
for i = 1:numel(ptrue)
    pidx = freeParamIndex(i);
    IOopts.marginals(i).Name       = paramNames{pidx};
    IOopts.marginals(i).Type       = 'Uniform';
    IOopts.marginals(i).Parameters = [0.1 10]*ptrue(i);
end

% total concentrations cannot be smaller than the initial condition!
if IOopts.marginals(end).Parameters(1) < x0(end)
    IOopts.marginals(end).Parameters(1) = x0(end)
end
if IOopts.marginals(end-1).Parameters(1) < x0(end-1)
    IOopts.marginals(end-1).Parameters(1) = x0(end-1)
end
if IOopts.marginals(end-2).Parameters(1) < x0(end-2)
    IOopts.marginals(end-2).Parameters(1) = x0(end-2)
end

input = uq_createInput(IOopts);

% options for Sobol sensitivity analysis
SobolSensOpts.Type = 'Sensitivity';
SobolSensOpts.Method = 'Sobol';
SobolSensOpts.Input = input;
SobolSensOpts.Sobol.Sampling = 'sobol';
SobolSensOpts.Sobol.SampleSize = 35000;


qoiFuncs = {@(sol) finalValQOI(sol, 1),@(sol) finalValQOI(sol, 2), @(sol) finalValQOI(sol, 3), @(sol) finalValQOI(sol, 3)/x0(end)};
Yltp = @(X) numericalModel(X, x0, @(t,x,theta) kinpho_LTD(t, x,theta), t, ...
            @(t,x,theta) Jac_LTD(t, x,theta), qoiFuncs);
Yltd = @(X) numericalModel(X, x0, @(t,x,theta) kinpho_LTD(t, x,theta), t, ...
            @(t,x,theta) Jac_LTD(t, x,theta), qoiFuncs);

% Create UQlab models
modelOptsLTP.mHandle      = Yltp;
modelOptsLTP.isVectorized = false;
ltpModel                  = uq_createModel(modelOptsLTP);

fprintf('Running analysis for LTP\n');
ltpAnalysis       = SobolSensOpts;
ltpAnalysis.Model = ltpModel;
ltpSensitivtyResults = uq_createAnalysis(ltpAnalysis);

save([savedir, 'ltpGSA.mat'], 'ltpSensitivtyResults');

% Create UQlab models
modelOptsLTD.mHandle      = Yltd;
modelOptsLTD.isVectorized = false;
ltdModel                  = uq_createModel(modelOptsLTD);

fprintf('Running analysis for LTD\n');
ltdAnalysis       = SobolSensOpts;
ltdAnalysis.Model = ltdModel;
ltdSensitivtyResults = uq_createAnalysis(ltdAnalysis);

save([savedir, 'ltdGSA.mat'], 'ltdSensitivtyResults');

%%%% Functions %%%%
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
    end
end

function xout = solve(x0, f, t, jac)
    % sovle the ODE system using ODE15s
    odeOpts = odeset('Jacobian', @(t,x) jac(t,x));
    [~, xout] = ode23s(@(t,x) f(t,x), t, x0, odeOpts);
end

function qoi = meanQOI(sol, colIdx, startIdx, endIdx)
    if nargin < 4 % endIdx = end
        qoi = mean(sol(startIdx:end,colIdx));
    else % endIdx is different than end
        qoi = mean(sol(startIdx:endIdx,colIdx));
    end
end

function qoi = finalValQOI(sol, colIdx)
    qoi = sol(end,colIdx);
end

function Ca = stepFunc(t, tauOn, tauOff, low, high)
    Ca = low + (high-low)*heavyside(t-tauOn) - (high-low)*heavyside(t-tauOff);
end

function y = heavyside(x)
    y = 0*x;
    y(find(x>0)) = 1;
end
