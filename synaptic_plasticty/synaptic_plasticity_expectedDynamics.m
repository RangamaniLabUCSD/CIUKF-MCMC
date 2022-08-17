% Nathaniel Linden
% UCSD MAE
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;


% folder to save results
savedir = './expectedDynamics/';
mkdir(savedir); 

% Set seed and initialize uqlab
rng(100,'twister')

% time 
tend = 9;
t0 = 0;
dt = 0.25;
t = t0:dt:tend; 
DT = 0.01; 
tfine = t0:DT:tend;
if dt < DT, DT = dt; end

% MODEL PARAMETERS
ptrueFull = [2, 15, 1, 120, 2, 15, 1, 80, 1, 1, 6, 8, 10, 0.3, 4, 10, 1, 0.5, 0.5, 20, 1, 20];
paramNames = {'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'c1', 'c2', 'c3', 'c4', 'Km1', ...
        'Km2', 'Km3', 'Km4', 'Km5', 'K0', 'P0', 'Ktot', 'Atot', 'Ptot'};
state_names = {'x1', 'x2', 'x3'};

fixedParamIndex = []; % no fixed params - all params are ID
freeParamIndex = setdiff(1:numel(ptrueFull), fixedParamIndex);
ptrue = ptrueFull(freeParamIndex);
thetaFull = @(theta) fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull);

% Intial Condition
x0 = [0.0228, 0.0017, 0.4294]';

Ca_basal = 0.1; % calcium input
start = 1; stop = 3;
f_Ca_LTP = @(t) stepFunc(t, start, stop, Ca_basal, 4.0);
f_Ca_LTD = @(t) stepFunc(t, start, stop, Ca_basal, 2.2);

kinpho_LTP = @(t, x, theta) synaptic_plasticity(t, x, [thetaFull(theta), f_Ca_LTP(t)]);
Jac_LTP = @(t,x, theta) synaptic_plasticity_Jacobian(t, x, [thetaFull(theta), f_Ca_LTP(t)]);

kinpho_LTD = @(t, x, theta) synaptic_plasticity(t, x, [thetaFull(theta), f_Ca_LTD(t)]);
Jac_LTD = @(t,x, theta) synaptic_plasticity_Jacobian(t, x, [thetaFull(theta), f_Ca_LTD(t)]);

xoutLTP = solve(x0, @(t,x) kinpho_LTP(t,x,ptrue), tfine, @(t,x) Jac_LTP(t,x,ptrue));
xoutLTD = solve(x0, @(t,x) kinpho_LTD(t,x,ptrue), tfine, @(t,x) Jac_LTD(t,x,ptrue));

sigmaR = 0.1*std(xoutLTP)';
ltpData = generateData(@(t, x) kinpho_LTP(t, x, ptrue), x0, t, eye(3), sigmaR);

sigmaR = 0.1*std(xoutLTD)';
ltdData = generateData(@(t, x) kinpho_LTD(t, x, ptrue), x0, t, eye(3), sigmaR);
ltdData=abs(ltdData); % if less than zero, set to abs values
ltpData=abs(ltpData);

figure(1)
plot(tfine, xoutLTP(:,1), 'k'); hold on
plot(tfine, xoutLTD(:,1), 'b');
plot(t, [x0(1), ltpData(1,:)],'.')
plot(t, [x0(1), ltdData(1,:)],'.')

saveas(gcf, [savedir, 'Pexpected.png']);
% fname = [savedir, 'Pexpected.tex'];
% datPath = [savedir,'Pexpected_data/'];
% relDatPath = 'Pexpected_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 1

figure(2)
plot(tfine, xoutLTP(:,2), 'k'); hold on
plot(tfine, xoutLTD(:,2), 'b');
plot(t, [x0(2), ltpData(2,:)],'.')
plot(t, [x0(2), ltdData(2,:)],'.')

saveas(gcf, [savedir, 'pKexpected.png']);
% fname = [savedir, 'pKexpected.tex'];
% datPath = [savedir,'pKexpected_data/'];
% relDatPath = 'pKexpected_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 2

figure(3)
plot(tfine, xoutLTP(:,3)/x0(3), 'k'); hold on
plot(tfine, xoutLTD(:,3)/x0(3), 'b');
plot(t, [x0(3), ltpData(3,:)]/x0(3),'.')
plot(t, [x0(3), ltdData(3,:)]/x0(3),'.')

saveas(gcf, [savedir, 'Aexpected.png']);
% fname = [savedir, 'Aexpected.tex'];
% datPath = [savedir,'Aexpected_data/'];
% relDatPath = 'Aexpected_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 3


%%%% Functions %%%%
function xout = solve(x0, f, t, jac)
    % sovle the ODE system using ODE15s
    odeOpts = odeset('Jacobian', @(t,x) jac(t,x));
    [~, xout] = ode15s(@(t,x) f(t,x), t, x0, odeOpts);
end
function Ca = stepFunc(t, tauOn, tauOff, low, high)
    Ca = low + (high-low)*heavyside(t-tauOn) - (high-low)*heavyside(t-tauOff);
end

function y = heavyside(x)
    y = 0*x;
    y(find(x>0)) = 1;
end

% Need generate data that assumes dx/dt = f(t,x)
function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode15s(@(t,x)f(t, x), t, x0);
    y = H*x' + sigmaR.*randn(size(H*x'));
    y = y(:,2:end);
end