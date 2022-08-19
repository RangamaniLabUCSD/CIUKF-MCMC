% Nathaniel Linden
% UCSD MAE
% Script for the CIUKF-MCMC tutorial
% This script runs a demo simulation for the MAPK example

clear all; close all; clc
addpath('../utils/')   % add paths to utilities and the MAPK model
addpath('../MAPK/')
savedir = './figures/';   % define path to save outputs
rng(100,'twister') % set random number seed for reproducibility

% Set up overhead for simulating the model
% This include time limites, time resolution, and the functions with the ODEs and Jacobian matrix

% timing
t0 = 0; tend = 1800;
dt = 60; DT = 20;   % define two time resolutions, fine and coarse
t = t0:dt:tend; tfine = t0:DT:tend;

% ode and Jacobian functions
MAPK = @(x, theta) MAPK_cascade(x, theta);
Jac = @(x, theta) MAPK_Jacobian(x, theta);

% parameters and initial conditions
theta_bistable = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];
x0   = [0.0015; 3.6678; 28.7307]; % low
% Simulate the model with ODE15s
% Specification of the Jacobian matrix speeds up the simulation time and improves accuracy
xout = solve(x0, @(t,x,theta) MAPK(x, theta), tfine, @(t,x,theta) Jac(x,theta), theta_bistable);

% Noisy sampling
% We will take one sample of each state variable every 60 seconds to simulate experimental measurements
% Furthermore we will add normally distributed noise with mean zero and a variance of 10 percent of the standard deviation of the data
sigma_noise = 0.5 * std(xout)';
odeOpts = odeset('Jacobian', @(t, x) Jac(x,theta_bistable));
data = generateData(@(x) MAPK(x, theta_bistable), x0, t, eye(3), sigma_noise, odeOpts); % Use ode15s to solve the system and get 'training data'
data=abs(data); % if less than zero, set to abs values
save('data.mat', 'data');

% Plot
plottingPreferences;
plot(tfine/60, xout(:,3), 'b'); hold on
plot(t(2:end)/60, data(3,:), 'k.', 'MarkerSize', 15)
xlabel('Time (min)'); ylabel('Concentration of $x_3$')
legend('Simulated Trajectory', 'Synthetic Data')
saveas(gcf, [savedir, 'mapk_example.png'])

%%%% Function definitions %%%%
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