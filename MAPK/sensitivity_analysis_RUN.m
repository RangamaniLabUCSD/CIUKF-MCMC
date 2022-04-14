% Nathaniel Linden
% UCSD MAE
% Script to run GSA for the MAPK model
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;

% folder to save results
savedir = './MAPK/sensitivity_analysis/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')
uqlab

% Time spacing
t0 = 0; 
dt = 60;
DT = 10;

% Exclude non-identifiable or exponents from analysis
fixedParamIndex = [1,2,3,10,11,12,13]; nparams = 14;
freeParamIndex = setdiff(1:nparams, fixedParamIndex);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramBounds = [0 100; 0 100; 0 100; 0 0.1; 0 0.1; 0 0.05; 0 0.1; 0 0.05; 0 0.1; 5 10; 0 10; 5 10; 0 20; 0 100];

% Functions for the ODE
thetaFull = @(theta, ptrueFull) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);
MAPK = @(x, theta, ptrueFull) MAPK_cascade(x, thetaFull(theta, ptrueFull));
Jac = @(x, theta, ptrueFull) MAPK_Jacobian(x, thetaFull(theta,ptrueFull));

% Struct for parameter options
IOopts.marginals = [];
for i = 1:numel(freeParamIndex)
    pidx = freeParamIndex(i);
    IOopts.marginals(i).Name       = paramNames{pidx};
    IOopts.marginals(i).Type       = 'Uniform';
    IOopts.marginals(i).Parameters = paramBounds(pidx,:);
end
input = uq_createInput(IOopts);

% options for Sobol sensitivity analysis
SobolSensOpts.Type = 'Sensitivity';
SobolSensOpts.Method = 'Sobol';
SobolSensOpts.Input = input;
SobolSensOpts.Sobol.Sampling = 'sobol';
SobolSensOpts.Sobol.SampleSize = 5000;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bistable Case %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
tend = 1800;
t = t0:dt:tend;
tfine = t0:DT:tend;

ptrueBistable = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];

% Different SS are dicated by the initial condition
x0Low   = [0.1245; 2.4870; 31.2623];
x0High  = [0.0015; 3.6678; 28.7307]; 

qoiFuncs = {@(sol) finalValQOI(sol, 1),@(sol) finalValQOI(sol, 2), @(sol) finalValQOI(sol, 3)};
Ylow = @(X) numericalModel(X, x0Low, @(t,x,theta) MAPK(x,theta, ptrueBistable), t, ...
            @(t,x,theta) Jac(x,theta, ptrueBistable), qoiFuncs);
Yhigh = @(X) numericalModel(X, x0High, @(t,x,theta) MAPK(x,theta, ptrueBistable), t, ...
            @(t,x,theta) Jac(x,theta, ptrueBistable), qoiFuncs);


% Create UQlab models
modelOptsLow.mHandle      = Ylow;
modelOptsLow.isVectorized = false;
lowModel                  = uq_createModel(modelOptsLow);

fprintf('Running analysis for Low steady-state\n');
lowAnalysis       = SobolSensOpts;
lowAnalysis.Model = lowModel;
lowSensitivtyResults = uq_createAnalysis(lowAnalysis);
save([savedir, 'lowSSGSA.mat'], 'lowSensitivtyResults');

modelOptsHigh.mHandle = Yhigh;
modelOptsHigh.isVectorized = false;
highModel = uq_createModel(modelOptsHigh);

fprintf('Running analysis for High steady-state\n');
highAnalysis       = SobolSensOpts;
highAnalysis.Model = highModel;
highSensitivtyResults = uq_createAnalysis(highAnalysis);

save([savedir, 'highSSGSA.mat'], 'highSensitivtyResults');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Oscillatory Case %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We want the lower bound for k5 to be slightly larger than 0 eg 1e-5
IOopts.marginals(5).Parameters(1) = 1e-5;
input = uq_createInput(IOopts);
SobolSensOpts.Input = input;

tend = 5400;
t = t0:dt:tend; 
DT = 20;
tfine = t0:DT:tend;

ptrueOSC = [100,100,100, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 10, 1, 15, 8, 10];

x0 = [10; 80; 80];
qoiFuncs = {@(sol) lcaQOI(sol, 3, 50, DT), @(sol) periodQOI(sol, 3, 50, DT), @(sol) meanQOI(sol, 3, 50)};
Yosc = @(X) numericalModel(X, x0, @(t,x,theta) MAPK(x,theta, ptrueOSC), tfine, @(t,x,theta) Jac(x,theta, ptrueOSC), qoiFuncs);

modelOptsOsc.mHandle = Yosc;
modelOptsOsc.isVectorized = false;
oscModel = uq_createModel(modelOptsOsc);

fprintf('Running analysis for Oscillations\n');
oscAnalysis       = SobolSensOpts;
oscAnalysis.Model = oscModel;
oscSensitivtyResults = uq_createAnalysis(oscAnalysis);

save([savedir, 'oscGSA.mat'], 'oscSensitivtyResults');
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
