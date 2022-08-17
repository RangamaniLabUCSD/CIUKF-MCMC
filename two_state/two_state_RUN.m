% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')

propCMult = 4; 

rng(100, 'twister'); % set random number gen. for reproducibility
if isempty(gcp('nocreate'))
    parpool(2)
end

% UKF parameters
alpha = 1e-3; beta = 2; kappa = 0; eps = 1e-10;

% Model parameters
% Parameters are [k1e, k12, k21, b]
ptrue = [1, 1, 1, 2];

% time 
% Same time domain and grid spacing as in SBINN training
tend = 2;
t0 = 0;
dt = 0.1; dtfine = 0.05; dtxfine = 0.0025;
t = t0:dt:tend; tfine = t0:dtfine:tend; txfine = t0:dtxfine:tend;
DT = 0.005;
if dt < DT, DT = dt; end

% dynamics model
d = 2; % dimension of the state
x0 = [0.5, 0.5]'; % initial condition
P0 = 1e-16*eye(d); % known initial condition, but must be non-zero for UKF
pdyn = 4; pvar = 3;
Q = @(theta) diag([theta(pdyn+1:end-1)]);
Qfixed = @(theta, sigmaQ) diag([sigmaQ, sigmaQ]);

% measurement model
H = [1 0; 0 1];
h = @(x, theta) H*x;
m = size(H,1);
R = @(theta) theta(end)*eye(m); % measurement nosie theta(end) is the cov
Rfixed = @(theta, sigmaR) sigmaR*eye(m);

% discrete time dynamics operator (psi)
u = @(t) stepFuncInput(t);
f = @(t_idx, x, theta)propf(t_idx, x, @(t,x)two_state(t, x, [theta, u(t)]), dt/DT, dt, DT);

% general other constants and parameters
num_samples = 3e4;
% optimization options
options = optimoptions('fmincon', 'UseParallel', true,'MaxFunctionEvaluations', 1e4, 'Display','final');

% unif dist bonnds
bounds(:,1) = zeros(pdyn, 1);
bounds(:,2) = [5;5;5;10];

% log priors
logprior = @(theta)log(uniformPrior(theta, bounds, pdyn, 0));
logprior_free = @(theta)log(uniformPrior(theta, bounds, pdyn, pvar));

state_colors = [75 93 22; 173 3 222] / 255;

paramNames = {'k1e', 'k12', 'k21', 'b'};
paramNamesTex = {'$k_{1e}$', '$k_{12}$', '$k_{21}$', '$b$'};
priors = {};
for i = 1:pdyn
    priors{i} = makedist('Uniform', bounds(i,1), bounds(i,2));
end
samplePnts = 0.995*bounds(:,1):0.001:bounds(:,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    Experiment 1: noise level                                     %%%%
%%%%    - free noise                                                  %%%%
%%%%    - Full-state measurements                                     %%%%
%%%%    - vary measurement noise level                                %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savedir = ['./data_noise_exps/'];
mkdir(savedir); 

noiseLevels = [0.0037, 0.015, 0.0375, 0.075, 0.1125, 0.15];
theta_init = [unifrnd(bounds(:,1), bounds(:,2), pdyn, 1)];

for i = 1:numel(noiseLevels)
    measCov = noiseLevels(i);
    disp(['Noise ', num2str(measCov)])
    filename = ['sigmaR_',num2str(measCov), '_'];
    [y] = generateData(@(t, x) two_state(t, x, ptrue), x0, t, H, measCov);

    objective_ukf = @(theta)-UKF(theta, x0, P0, @(t, x) f(t, x,theta), h, ...
        Q(theta), R(theta), y, logprior_free, alpha, beta, kappa, eps);
    theta_init_free = [theta_init; 4*measCov; 4*measCov; measCov];
    [theta0] = fmincon(objective_ukf, theta_init_free, [],[],[],[],[bounds(:,1); zeros(pvar,1)],[bounds(:,2); Inf*ones(pvar,1)], [],options);
    % theta0 = theta_init_free;
    propC = diag([(propCMult/12) * (bounds(:,2)-bounds(:,1)).^2; 0.01*theta0(end-2); 0.01*theta0(end-1); 0.0025*theta0(end)]);
   % propC = 1e-3*eye(numel(theta0));
    [samples, acc] = UKFMCMC(y,theta0, propC, num_samples, x0, P0, f, h,Q,R,logprior_free);
    acc
    save([filename, 'free_fullStateSamples.mat'], 'samples');
    save([filename, 'free_fullStateAcc.mat'], 'acc'); acc
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    Experiment 2: temporal resolution                             %%%%
%%%%    - free noise                                                  %%%%
%%%%    - Full-state measurements                                     %%%%
%%%%    - same nosie, vary measurement resolution                     %%%%
%%%%    - fix forward euler res (DT) to a small value                 %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savedir = ['./data_sampling_exps/'];
mkdir(savedir); 

%measCov = 0.0375;
measCov = [0.0375, 0.1125];
dts = [0.2, 0.1, 0.05];%, 0.001]; % measurement noise!
theta_init = [unifrnd(bounds(:,1), bounds(:,2), pdyn, 1)];
%theta_init_free = [theta_init; 4*measCov; 4*measCov; measCov];


for j = 1:numel(measCov)
    for i = 1:numel(dts)
        noise = measCov(j);
        dt = dts(i)
        disp(['dt ', num2str(dt)])
        filename = ['dt_',num2str(dt), '_', num2str(noise), '_'];
        t = t0:dt:tend;
        f = @(t_idx, x, theta)propf(t_idx, x, @(t,x)two_state(t, x, theta), dt/DT, dt, DT);

        [y] = generateData(@(t, x) two_state(t, x, ptrue), x0, t, H, noise);
	
	    theta_init_free = [theta_init; 4*noise; 4*noise; noise];
        
         objective_ukf = @(theta)-UKF(theta, x0, P0, @(t, x) f(t, x,theta), h, ...
             Q(theta), R(theta), y, logprior_free, alpha, beta, kappa, eps);

         theta0 = fmincon(objective_ukf, theta_init_free, [],[],[],[],[bounds(:,1); zeros(pvar,1)],[bounds(:,2); Inf*ones(pvar,1)], [],options);
         propC = diag([(propCMult/12) * (bounds(:,2)-bounds(:,1)).^2; 0.01*theta0(end-2); 0.01*theta0(end-1); 0.0025*theta0(end)]);
         [samples, acc] = UKFMCMC(y,theta0, propC, num_samples, x0, P0, f, h,Q,R,logprior_free);
         save([filename, 'free_fullStateAcc.mat'], 'acc'); acc
         save([filename, 'free_fullStateSamples.mat'], 'samples');
    end
end

%%%% Functions %%%%
function pTheta = uniformPrior(theta, bounds, pdyn, pvar)
    noiseVari = 10;
    modelParams = 1;
    for i = 1:pdyn
        modelParams = modelParams * unifpdf(theta(i),bounds(i,1),bounds(i,2));
    end

    if pvar
        noiseParams = rhnpdf(theta(pdyn+1:end), ...
            zeros(size(theta(pdyn+1:end))), noiseVari*eye(numel(theta(pdyn+1:end))));
        pTheta = modelParams * noiseParams;
    else
        pTheta = modelParams;
    end
end

function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode15s(@(t,x) f(t, x), t, x0);
    y = H*x' + sigmaR*randn(size(H*x'));
    y = y(:,2:end);
end

function xout = propf(t_idx, xin, f, N, dt, DT)
    t = t_idx*dt;
    thresh = 1e-5*ones(2,1);
    xout = xin;

    [~, xout] = ode15s(@(t,x)f(t, x), [t, t+dt], xin);
    % end
    xout = xout(end,:)';
end

function u = stepFuncInput(t)
    u_init = 0; slope = 2;
    ton = 1;
    if t < ton
        u = u_init + (slope*t);
    else
        u = 1.5*exp(1-t);
    end
end
