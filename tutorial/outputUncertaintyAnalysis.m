% Nathaniel Linden
% UCSD MAE
% Script for the CIUKF-MCMC tutorial
% This script runs output uncertainty analysis for the example


clear all; close all; clc
addpath('../utils/')   % add paths to utilities and the MAPK model
addpath('../MAPK/')
savedir = './figures/';   % define path to save outputs
rng(100,'twister') % set random number seed for reproducibility
uqlab
plottingPreferences;
% Set up overhead for simulating the model
% This include time limites, time resolution, and the functions with the ODEs and Jacobian matrix

% timing
t0 = 0; tend = 1800;
dt = 60; DT = 20;   % define two time resolutions, fine and coarse
t = t0:dt:tend; tfine = t0:DT:tend;

% parameters and initial conditions
theta_bistable = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];
x0   = [0.0015; 3.6678; 28.7307]; % low

freeParamIndex = [5, 7, 8, 9]; % estimate k2, k4, k5, k6
fixedParamIndex = setdiff(1:numel(theta_bistable), freeParamIndex);
ptrue = theta_bistable(freeParamIndex);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramNames = paramNames(freeParamIndex);
state_names = {'x1', 'x2', 'x3'};

% load results
load([savedir, 'CIUKF_MCMC_results.mat']);


% Model overhead
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, theta_bistable);
MAPK = @(x, theta) MAPK_cascade(x, thetaFull(theta));
Jac = @(x, theta) MAPK_Jacobian(x, thetaFull(theta)); % use analytical (computed by hand) Jacobian for stability
odeOptsTrue = odeset('Jacobian', @(t, x) Jac(x,ptrue));

% Run the ensemble simulation
% only want model params so take (:,1:4,:)
% Note: we can randomly downsample BayesianAnalysis.Results.PostProc.PostSample(:,1:4,:) 
%       if it is large
postSamples2d = flattenEnsemble(BayesianAnalysis.Results.PostProc.PostSample(:,1:4,:));
[xpost, index] = runEnsembleSimulation(@(t,x,theta) MAPK(x,theta), @(t,x,theta) Jac(x, theta), postSamples2d, t, x0, size(postSamples2d,2));

% Extract MAP point
MAP = BayesianAnalysis.Results.PostProc.PointEstimate.X{1};

%% Plot the ensemble for x_3(t)
quantl = quantile(xpost, 0.025, 3); quantu = quantile(xpost, 0.975, 3);
lower = [t'; flipud(t')]; upper = [quantl, fliplr(quantu)];

xtrue = solve(x0, @(t,x,theta) MAPK_cascade(x, theta), tfine, @(t,x,theta) MAPK_Jacobian(x,theta), theta_bistable);
xmap = solve(x0, @(t,x,theta) MAPK(x, theta), tfine, @(t,x,theta) Jac(x,theta), ...
       MAP(1:4));


green = [75 93 22] / 255;
% shaded 95 credible interval
xposts = fill(lower/60, upper(3,:), green, 'FaceAlpha', 0.1, ...
            'DisplayName', '$95 \%$ credible', 'EdgeColor', green); hold on

% ensemble
for i = 1:size(postSamples2d,2)
    plot(t/60, xpost(3,:,i),'Color', [0,0,1,0.01], 'HandleVisibility', 'off'); hold on
end
plot(nan, nan,'Color', [0,0,1,], 'DisplayName','Simulated Trajectory')

% truth
plot(tfine/60, xtrue(:,3), 'Color', 'k', 'DisplayName', 'True parameters', 'LineWidth', 2)

% map
plot(tfine/60, xmap(:,3), 'Color', 'r', 'DisplayName', 'MAP parameters', 'LineWidth', 2)

xlabel('Time (min)')
ylabel('Concentration of $x_3$')
legend('FontSize', 20)

saveas(gcf, [savedir,'outputUQ.png'])
% 
%% Functions
function xout = solve(x0, f, t, jac, theta)
    % solve the ODE system using ODE15s
    odeOpts = odeset('Jacobian', @(t,x) jac(t,x, theta));
    [~, xout] = ode15s(@(t,x) f(t,x, theta), t, x0, odeOpts);
end

function plottingPreferences()
    N = 18;
    set(0,'DefaultLineLineWidth',2)
    
    set(0,'defaultAxesFontSize',N)
    set(0, 'defaultLegendFontSize', N)
    set(0, 'defaultColorbarFontSize', N);
    
    set(0,'defaulttextinterpreter','latex')
    set(0, 'defaultAxesTickLabelInterpreter', 'latex')
    set(0, 'defaultLegendInterpreter', 'latex')
end