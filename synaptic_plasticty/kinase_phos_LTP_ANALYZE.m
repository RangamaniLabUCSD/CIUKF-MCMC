% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;

% folder to save results
savedir = './LTP/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL SETUP and OVERHEAD %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time
tend = 9;
t0 = 0;
dt = 0.25;
t = t0:dt:tend;
DT = 0.01; tfine = t0:DT:tend;
if dt < DT, DT = dt; end

% MODEL PARAMETERS
ptrueFull = [2, 15, 1, 120, 2, 15, 1, 80, 1, 1, 6, 8, 10, 0.3, 4, 10, 1, 0.5, 0.5, 20, 20, 1]; 

paramNames = {'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'c1', 'c2', 'c3', 'c4', 'Km1', ...
        'Km2', 'Km3', 'Km4', 'Km5', 'K0', 'P0', 'Ktot', 'Ptot', 'Atot'};
paramNamesTex = {'$k_1$', '$k_2$', '$k_3$', '$k_4$', '$k_5$', '$k_6$', '$k_7$', '$k_8$', ...
        '$c_1$', '$c_2$', '$c_3$', '$c_4$', '$K_{m1}$', '$K_{m2}$', '$K_{m3}$', '$K_{m4}$',...
        '$K_{m5}$', '$K_0$', '$P_0$', 'K_{tot}', 'P_{tot}', 'A_{tot}'};
state_names = {'x1', 'x2', 'x3'};

% Fix subset of params based on sensitivity analysis
% Fixed params include:
%   - Rates: k1, k3, k4, k5, k7, k8
%   - Km1, Km2, Km4, Km5
%   - c1, c2, c3, c4
% Free params are:
%   - k2, k6, K0, P0, Ktot, Ptot, Atot 
fixedParamIndex = [1, 3, 4, 5, 7, 8, 9, 10,11,12, 13, 14, 16, 15, 17];
freeParamIndex = setdiff(1:numel(ptrueFull), fixedParamIndex);
paramNames = paramNames(freeParamIndex);
paramNamesTex = paramNamesTex(freeParamIndex);
ptrue = ptrueFull(freeParamIndex);
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);

% Intial Condition
x0 = [0.0228, 0.0017, 0.4294]';

% Model dimensions
d = 3; m = 3;
pdyn = numel(ptrue);
pvar = 2 * d; % # process noise + # meas noise

Ca_basal = 0.1; Ca_peak = 4; % calcium input
start = 1; stop = 3;
f_Ca = @(t) stepFunc(t, start, stop, Ca_basal, Ca_peak);

kinpho = @(t, x, theta) phosphatase_kinase(t, x, thetaFull(theta), f_Ca);
Jac = @(t, x, theta) phosphatase_kinase_Jacobian(t, x, thetaFull(theta), f_Ca);
[~, yTrue] = ode15s(@(t, x) kinpho(t, x, ptrue), tfine, x0);

% PARAMETER BOUNDS %
% bounds set to be one order of magnitude smaller and
%   one order of mag larger than the nominal values
bounds = zeros(pdyn, 2);
bounds(:, 1) = 0.1 * ptrue;
bounds(:, 2) = 10 * ptrue;

% load the MCMC samples and all of the info on the run
run1 = load([savedir, 'results_AIES.mat']);
run2 = load([savedir, 'results_AIES_run2.mat']);
run3 = load([savedir, 'results_AIES_run3.mat']);

% process samples
postSamples_1 = run1.BayesianAnalysis.Results.Sample;
postSamples_2 = run2.BayesianAnalysis.Results.Sample;
postSamples_3 = run3.BayesianAnalysis.Results.Sample;

posteriorSamplesNoBurn = cat(1, postSamples_1, postSamples_2, postSamples_3);

fprintf('IACT with no burn in for all runs: \n');
IACTfull_all = computeIACT(posteriorSamplesNoBurn);
fprintf([num2str(IACTfull_all), '\n']);

% compute burnin as a multiple (5-10x) of the ICAT
burnin = floor(IACTfull_all * 7);

fprintf('IACT with no burn in for the first run: \n');
IACTfull_run1 = computeIACT(postSamples_1);
fprintf([num2str(IACTfull_run1), '\n']);

fprintf('IACT for the second run: \n')
fprintf([num2str(computeIACT(postSamples_2)), '\n']);

fprintf('IACT for the third run: \n')
fprintf([num2str(computeIACT(postSamples_3)), '\n']);

% concatenate converged samples from run 1 and run 2
posteriorSamples = posteriorSamplesNoBurn(burnin:end, :, :);

fprintf('IACT for all posterior samples (with burn-in removed): \n')
fprintf([num2str(computeIACT(posteriorSamples)), '\n']);

% flatten ensemble into 2D matrix
posteriorSamples2d = flattenEnsemble(posteriorSamples);

% approximate MAP and MEAN by fitting a kernel density estimator
[thetaMap, thetaMean, fits, x_fit] = approximate_MAP(posteriorSamples2d, bounds);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting posteiors and priors %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nsampls = 100; % number of samples to use for plotting

[xpost, index] = runEnsembleSimulation(@(t, x, theta) kinpho(t, x, theta), Jac, posteriorSamples2d, tfine, x0, nsampls);
[xprior, index] = runEnsembleSimulation(@(t, x, theta) kinpho(t, x, theta), Jac, ...
    run1.BayesianAnalysis.Results.PostProc.PriorSample', tfine, x0, 1000);

% dynanics post w/ 95% credible
plotDynamicsPosterior(posteriorSamples2d, nsampls, t, tfine, run1.BayesianAnalysis.Data.y, x0, ...
    @(t, x, theta) kinpho(t, x, theta), savedir, 'xpost', ptrue', thetaMap, thetaMean, [1, 1, 1], ...
    state_names, [0.025, 0.975], xpost, [1; 1; x0(end)]);

% parameter 1D marginal posteriors
priors = {};

for i = 1:numel(paramNames)
    priors{i} = makedist('Uniform', bounds(i, 1), bounds(i, 2));
end

plotMarginalPosterior(posteriorSamples2d, ptrue, paramNames, paramNamesTex, savedir, 'param_', thetaMap, thetaMean, fits, x_fit, priors, bounds);
close all;

%% Plot posterior with a different Ca input
Ca_basal = 0.1; Ca_peak = 2.2; % calcium input
start = 1; stop = 3;
f_Ca_ltd = @(t) stepFunc(t, start, stop, Ca_basal, Ca_peak);

kinpho_ltd = @(t, x, theta) phosphatase_kinase(t, x, thetaFull(theta), f_Ca_ltd);
Jac_ltd = @(t, x, theta) phosphatase_kinase_Jacobian(t, x, thetaFull(theta), f_Ca_ltd);
[~, yTrueLTD] = ode15s(@(t, x) kinpho_ltd(t, x, ptrue), tfine, x0);
[xpostLTD, index] = runEnsembleSimulation(@(t, x, theta) kinpho_ltd(t, x, theta), Jac_ltd, posteriorSamples2d, tfine, x0, nsampls);
[xpriorLTD, index] = runEnsembleSimulation(@(t, x, theta) kinpho_ltd(t, x, theta), Jac_ltd, ...
    run1.BayesianAnalysis.Results.PostProc.PriorSample', tfine, x0, 1000);

plotDynamicsPosterior(posteriorSamples2d, nsampls, t, tfine, [], x0, @(t, x, theta) kinpho(t, x, theta), ...
    savedir, 'xpostLTDtest', ptrue', thetaMap, thetaMean, [0, 0, 0], state_names, [0.025, 0.975], xpostLTD, [1; 1; x0(end)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots of ensemble with LTP inducing input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
% Posterior
nsamp = 100;
downsamp = 6;
% LTP inducing Ca input
figure(1)
for i = 1:nsamp
    if xpost(3,end,i)/x0(end) > 1 % This is LTP case
        figure(1); plot(tfine(1:downsamp:end),  xpost(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,1,0.15], 'LineWidth', 1); hold on
    else % This is other/LTD case
        figure(1); plot(tfine(1:downsamp:end),  xpost(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,0,0.15], 'LineWidth', 1); hold on
    end
end
plot(tfine(1:downsamp:end), yTrue(1:downsamp:end,3)/x0(3), 'g', 'LineWidth',2)

saveas(gcf, [savedir, 'postSamps_ltp.png'])
% fname = [savedir, 'postSamps_ltp.tex'];
% datPath = [savedir,'postSamps_ltp_data/'];
% relDatPath = 'postSamps_ltp_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

[N, edges] = histcounts(xpost(3, end, :)/x0(end),100,'Normalization','probability');
centers = edges(2:end) - ((edges(2) - edges(1))/2);
fill([1.6, 1.8, 1.8, 1.],[0 0 0.2 0.2],'r'); hold on
fill([0.3, 0.6, 0.6, 0.3],[0 0 0.2 0.2],'g');
fill([0.95, 1.05, 1.05, 0.95],[0 0 0.2 0.2],'b');
bar(centers, N,1, 'b'); hold on
% bar(centers(centers<1), N(centers <1),1, 'b'); hold on
% bar(centers(centers>1), N(centers >1),1, 'k'); 
ax = gca;
plot([yTrue(end,3) yTrue(end,3)]/x0(end), ax.YLim, 'k', 'LineWidth',2)
plot([ax.XLim(1)+0.01, 0.99],ones(1,2)*ax.YLim(end), 'k', 'LineWidth',3)
plot([1.01, ax.XLim(end)-0.01], ones(1,2)*ax.YLim(end), 'b', 'LineWidth',3)

saveas(gcf, [savedir, 'SShist_ltp.png'])
% fname = [savedir, 'SShist_ltp.tex'];
% datPath = [savedir,'SShist_ltp_data/'];
% relDatPath = 'SShist_ltp_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

% estimate proportion of samples that predict LTP vs. LTD
numLTP_ltp = sum((xpost(3, end, :)/x0(end)) > 1);
numLTD_ltp = sum((xpost(3, end, :)/x0(end)) < 1);
fprintf('Calculating LTP and LTD rates for LTP inducing input...\n');
fprintf([num2str(100*(numLTP_ltp/nsampls)), ' percent (', num2str(numLTP_ltp),' total) of the simulations reached an elevated EPSP state.\n'])
fprintf([num2str(100*(numLTD_ltp/nsampls)), ' percent (', num2str(numLTD_ltp),' total) of the simulations reached an depressed EPSP state.\n'])

close all
% Prior
nsamp = 100;
downsamp = 6;
% LTP inducing Ca input
figure(1)
for i = 1:nsamp
    if xprior(3,end,i)/x0(end) > 1 % This is LTP case
        figure(1); plot(tfine(1:downsamp:end),  xprior(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,1,0.15], 'LineWidth', 1); hold on
    else % This is other/LTD case
        figure(1); plot(tfine(1:downsamp:end),  xprior(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,0,0.15], 'LineWidth', 1); hold on
    end
end
plot(tfine(1:downsamp:end), yTrue(1:downsamp:end,3)/x0(3), 'g', 'LineWidth',2)

saveas(gcf, [savedir, 'priorSamps_ltp.png'])
% fname = [savedir, 'priorSamps_ltp.tex'];
% datPath = [savedir,'priorSamps_ltp_data/'];
% relDatPath = 'priorSamps_ltp_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

[N, edges] = histcounts(xprior(3, end, :)/x0(end),100,'Normalization','probability');
centers = edges(2:end) - ((edges(2) - edges(1))/2);
fill([1.6, 1.8, 1.8, 1.],[0 0 0.2 0.2],'r'); hold on
fill([0.3, 0.6, 0.6, 0.3],[0 0 0.2 0.2],'g');
fill([0.95, 1.05, 1.05, 0.95],[0 0 0.2 0.2],'b');
bar(centers, N,1, 'b'); hold on
% bar(centers(centers<1), N(centers <1),1, 'b'); hold on
% bar(centers(centers>1), N(centers >1),1, 'k'); 
ax = gca;
plot([yTrue(end,3) yTrue(end,3)]/x0(end), ax.YLim, 'k', 'LineWidth',2)
plot([ax.XLim(1)+0.01, 0.99],ones(1,2)*ax.YLim(end), 'k', 'LineWidth',3)
plot([1.01, ax.XLim(end)-0.01], ones(1,2)*ax.YLim(end), 'b', 'LineWidth',3)

saveas(gcf, [savedir, 'priorSShist_ltp.png'])
% fname = [savedir, 'priorSShist_ltp.tex'];
% datPath = [savedir,'priorSShist_ltp_data/'];
% relDatPath = 'priorSShist_ltp_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

% estimate proportion of samples that predict LTP vs. LTD
PRIORnumLTP_ltp = sum((xprior(3, end, :)/x0(end)) > 1);
PRIORnumLTD_ltp = sum((xprior(3, end, :)/x0(end)) < 1);
fprintf('Calculating LTP and LTD rates for LTP inducing input...\n');
fprintf([num2str(100*(PRIORnumLTP_ltp/nsampls)), ' percent (', num2str(PRIORnumLTP_ltp),' total) of the simulations reached an elevated EPSP state.\n'])
fprintf([num2str(100*(PRIORnumLTD_ltp/nsampls)), ' percent (', num2str(PRIORnumLTD_ltp),' total) of the simulations reached an depressed EPSP state.\n'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots of ensemble with LTD inducing input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
for i = 1:nsamp
    if xpostLTD(3,end,i)/x0(end) > 1 % This is LTP case
        figure(1); plot(tfine(1:downsamp:end),  xpostLTD(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,1,0.15], 'LineWidth', 1); hold on
    else % This is other/LTD case
        figure(1); plot(tfine(1:downsamp:end),  xpostLTD(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,0,0.15], 'LineWidth', 1); hold on
    end
end
plot(tfine(1:downsamp:end), yTrueLTD(1:downsamp:end,3)/x0(3), 'g', 'LineWidth',2)

saveas(gcf, [savedir, 'postSamps_ltd.png'])
% fname = [savedir, 'postSamps_ltd.tex'];
% datPath = [savedir,'postSamps_ltd_data/'];
% relDatPath = 'postSamps_ltd_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);

close all
[N, edges] = histcounts(xpostLTD(3, end, :)/x0(end),100,'Normalization','probability');
centers = edges(2:end) - ((edges(2) - edges(1))/2);
fill([1.6, 1.8, 1.8, 1.6],[0 0 0.2 0.2],'r'); hold on
fill([0.3, 0.6, 0.6, 0.3],[0 0 0.2 0.2],'g');
fill([0.95, 1.05, 1.05, 0.95],[0 0 0.2 0.2],'b');
bar(centers, N,1, 'b'); hold on
% bar(centers(centers<1), N(centers <1),1, 'b'); hold on
% bar(centers(centers>1), N(centers >1),1, 'k'); 
ax = gca;
plot([yTrueLTD(end,3) yTrueLTD(end,3)]/x0(end), ax.YLim, 'k', 'LineWidth',2)
plot([ax.XLim(1)+0.01, 0.99],ones(1,2)*ax.YLim(end), 'k', 'LineWidth',3)
plot([1.01, ax.XLim(end)-0.01], ones(1,2)*ax.YLim(end), 'b', 'LineWidth',3)

saveas(gcf, [savedir, 'SShist_ltd.png'])
% fname = [savedir, 'SShist_ltd.tex'];
% datPath = [savedir,'SShist_ltd_data/'];
% relDatPath = 'SShist_ltd_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

% estimate proportion of samples that predict LTP vs. LTD
numLTP_ltd = sum((xpostLTD(3, end, :)/x0(end)) > 1);
numLTD_ltd = sum((xpostLTD(3, end, :)/x0(end)) < 1);
fprintf('Calculating LTP and LTD rates for LTD inducing input ...\n');
fprintf([num2str(100*(numLTP_ltd/nsampls)), ' percent (', num2str(numLTP_ltd),' total) of the simulations reached an elevated EPSP state.\n'])
fprintf([num2str(100*(numLTD_ltd/nsampls)), ' percent (', num2str(numLTD_ltd),' total) of the simulations reached an depressed EPSP state.\n'])

close all
% Prior
nsamp = 100;
downsamp = 6;
% LTP inducing Ca input
figure(1)
for i = 1:nsamp
    if xpriorLTD(3,end,i)/x0(end) > 1 % This is LTP case
        figure(1); plot(tfine(1:downsamp:end),  xpriorLTD(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,1,0.15], 'LineWidth', 1); hold on
    else % This is other/LTD case
        figure(1); plot(tfine(1:downsamp:end),  xpriorLTD(3,1:downsamp:end,i)/x0(end), ...
            'Color', [0,0,0,0.15], 'LineWidth', 1); hold on
    end
end
plot(tfine(1:downsamp:end), yTrueLTD(1:downsamp:end,3)/x0(3), 'g', 'LineWidth',2)

saveas(gcf, [savedir, 'priorSamps_ltd.png']);
% fname = [savedir, 'priorSamps_ltd.tex'];
% datPath = [savedir,'priorSamps_ltd_data/'];
% relDatPath = 'priorSamps_ltd_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);

close all
[N, edges] = histcounts(xpriorLTD(3, end, :)/x0(end),100,'Normalization','probability');
centers = edges(2:end) - ((edges(2) - edges(1))/2);
fill([1.6, 1.8, 1.8, 1.],[0 0 0.2 0.2],'r'); hold on
fill([0.3, 0.6, 0.6, 0.3],[0 0 0.2 0.2],'g');
fill([0.95, 1.05, 1.05, 0.95],[0 0 0.2 0.2],'b');
bar(centers, N,1, 'b'); hold on
% bar(centers(centers<1), N(centers <1),1, 'b'); hold on
% bar(centers(centers>1), N(centers >1),1, 'k'); 
ax = gca;
plot([yTrue(end,3) yTrue(end,3)]/x0(end), ax.YLim, 'k', 'LineWidth',2)
plot([ax.XLim(1)+0.01, 0.99],ones(1,2)*ax.YLim(end), 'k', 'LineWidth',3)
plot([1.01, ax.XLim(end)-0.01], ones(1,2)*ax.YLim(end), 'b', 'LineWidth',3)

saveas(gcf, [savedir, 'priorSShist_ltd.png']);
% fname = [savedir, 'priorSShist_ltd.tex'];
% datPath = [savedir,'priorSShist_ltd_data/'];
% relDatPath = 'priorSShist_ltd_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close all

% estimate proportion of samples that predict LTP vs. LTD
PRIORnumLTP_ltd = sum((xpriorLTD(3, end, :)/x0(end)) > 1);
PRIORnumLTD_ltd = sum((xpriorLTD(3, end, :)/x0(end)) < 1);
fprintf('Calculating LTP and LTD rates for LTP inducing input...\n');
fprintf([num2str(100*(PRIORnumLTP_ltd/nsampls)), ' percent (', num2str(PRIORnumLTP_ltd),' total) of the simulations reached an elevated EPSP state.\n'])
fprintf([num2str(100*(PRIORnumLTD_ltd/nsampls)), ' percent (', num2str(PRIORnumLTD_ltd),' total) of the simulations reached an depressed EPSP state.\n'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bar chart of EPSP state %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
bar([numLTD_ltp, numLTP_ltp]/nsampls);

saveas(1, [savedir, 'Ssprob_ltp.png']);
% fname = [savedir, 'SSprob_ltp.tex'];
% datPath = [savedir,'SSprob_ltp_data/'];
% relDatPath = 'SSprob_ltp_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);

figure(2);
bar([numLTD_ltd, numLTP_ltd]/nsampls);

saveas(2, [savedir, 'SSprob_ltd.png']);
% fname = [savedir, 'SSprob_ltd.tex'];
% datPath = [savedir,'SSprob_ltd_data/'];
% relDatPath = 'SSprob_ltd_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);

close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting traces of the Ensembles %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
names = paramNames; names{8}='Q1'; names{9}='Q2'; names{10}='Q3';
names{11}='R1';names{12}='R2';names{13}='R3';
plotEnsemble(posteriorSamplesNoBurn, savedir, 'MCMC_', names, 10, burnin)

%%%% Functions %%%%
function Ca = stepFunc(t, tauOn, tauOff, low, high)
    Ca = low + (high - low) * heavyside(t - tauOn) - (high - low) * heavyside(t - tauOff);
end

function y = heavyside(x)
    y = 0 * x;
    y(find(x > 0)) = 1;
end
