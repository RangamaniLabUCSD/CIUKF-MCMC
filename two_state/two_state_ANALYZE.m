% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils')
plottingPreferencesNJL;
propCMult = 4;
rng(100, 'twister'); % set random number gen. seed to 1
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
f = @(t_idx, x, theta)propf(t_idx, x, @(t,x)two_state(t, x, theta), dt/DT, dt, DT);

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

    load([savedir, filename, 'free_fullStateSamples.mat'])
    
    burnIn = 10000;
    plotChains(samples, savedir, [filename, 'chain'], paramNames, 1, burnIn)
    
    samples = samples(:,burnIn:end);
    [thetaMap, thetaMean, fits, x_fit] = approximate_MAP(samples, bounds); % compute the MAP point

    [xpost, index] = runEnsembleSimulation(@(t,x,theta) two_state(t,x,theta), [], samples, t, x0, 5000);
    fname = [filename, 'dynamicsPost'];

    plotDynamicsPosterior(samples, 5000, t, t, y, x0,  @(t,x,theta) two_state(t,x,theta),savedir, ...
        fname, ptrue, thetaMap,thetaMean, [1,1], {'x1','x2'}, [0.025 0.975], xpost,[1;1])

    fname = [filename, 'param'];
    paramNames = {'k1e', 'k12', 'k21', 'b'};
    plotMarginalPosterior(samples, ptrue, paramNames,paramNamesTex, savedir, fname, thetaMap, thetaMean, fits, x_fit, priors, bounds);
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

        load([savedir, filename, 'free_fullStateSamples.mat'])
        burnIn = 10000;

        plotChains(samples, savedir, [filename, 'chain'], paramNames, 1, burnIn)

        samples = samples(:,burnIn:end);
        [thetaMap, thetaMean, fits, x_fit] = approximate_MAP(samples, bounds); % compute the MAP point

        [xpost, index] = runEnsembleSimulation(@(t,x,theta) two_state(t,x,theta), [], samples, t, x0, 5000);
        fname = [filename, 'dynamicsPost'];
        plotDynamicsPosterior(samples, 5000, t, t, y, x0,  ...
            @(t,x,theta) two_state(t,x,theta),savedir, fname, ptrue, thetaMap, thetaMean, ...
            [1,1], {'x1','x2'}, [0.025 0.975], xpost, [1;1]);
    
        fname = [filename, 'param'];
        paramNames = {'k1e', 'k12', 'k21', 'b'};
        plotMarginalPosterior(samples, ptrue, paramNames,paramNamesTex, savedir, fname, thetaMap, thetaMean, fits, x_fit, priors, ...
            bounds);
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
        % noiseParams = mvnpdf(theta(pdyn+1:end), ...
            % zeros(size(theta(pdyn+1:end))), noiseVari*eye(numel(theta(pdyn+1:end))));
        pTheta = modelParams * noiseParams;
    else
        pTheta = modelParams;
    end
end

function plotChains(samples, savedir, filename, paramNames, subsamp, burnin)
    if ~subsamp
        subsamp = 1;
    end

    nparams = numel(paramNames);
    nsteps = size(samples,2);
    steps = 1:nsteps;
    for prm = 1:nparams
        figure
        plot(steps(1:subsamp:end), samples(prm, 1:subsamp:end), 'LineWidth', 1,...
            'Color', [0.7, 0.7, 0.7]);
        hold on
        ax = gca;
        plot([burnin, burnin], ax.YLim)
        xlabel('MCMC Steps');
        ylabel(paramNames{prm});

        fname = [savedir, filename, paramNames{prm}];
        datPath = [savedir, filename, paramNames{prm},'_data/'];
        relDatPath = [filename, paramNames{prm},'_data/'];

        saveas(gcf, [fname, '.png']);
        % cleanfigure; 
        % matlab2tikz([fname, '.tex'], 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
        %     'externalData', true);
    end
end

function plot2Dmarginals(samples, paramNamesTex, paramNames, filename, nParam)
    cmap = colorcet('L12');
    for i = 1:nParam
        for j = 1:nParam
            if i~=j
                figure('position', [0,0,400,400])
                [N, Xedges, Yedges] = histcounts2(samples(i,:), samples(j,:), 'Normalization', 'pdf');
                Xcenters = Xedges(2:end) - ((Xedges(2) - Xedges(1))/2);
                Ycenters = Yedges(2:end) - ((Yedges(2) - Yedges(1))/2);
                contourf(Xcenters, Ycenters, N',15)
                colormap(cmap); colorbar
                xlabel(paramNamesTex{i}); ylabel(paramNamesTex{j});
                fname = [filename,'_',paramNames{i},'_',paramNames{j},'.png'];
                saveas(gcf, fname);
                close gcf
            end
        end
    end
end

function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode45(@(t,x) f(t, x), t, x0);
    y = H*x' + sigmaR*randn(size(H*x'));
    y = y(:,2:end);
end

function xout = propf(t_idx, xin, f, N, dt, DT)
    t = t_idx*dt;
    thresh = 1e-5*ones(2,1);
    xout = xin;
    % for i = 1:N
        % xout = xout + f((t+(i-1)*DT), xout)*DT;
        % J = numjac(f, t+(i-1)*DT, xout, f(t+(i-1)*DT, xout), thresh, [], 0);
        % xout = xout + (DT*eye(2))/(eye(2) - DT*J) * f((t+(i-1)*DT), xout);
    [~, xout] = ode15s(@(t,x)f(t, x), [t, t+dt], xin);
    % end
    xout = xout(end,:)';
end
