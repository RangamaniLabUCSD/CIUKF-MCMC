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
tend = 5400;
t0 = 0; dt = 60; DT = 20;
t = t0:dt:tend;
tfine = t0:DT:tend;
if dt < DT, DT = dt; end

% Functions for the ODE
MAPK = @(x, theta) MAPK_cascade(x, theta);
Jac = @(x, theta) MAPK_Jacobian(x, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bistable Case %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
tend = 1800;
t = t0:dt:tend;
tfine = t0:DT:tend;

ptrueBistable = [0.22,10,53, 0.0012, 0.006, 0.049, 0.084, 0.043, 0.066, 5, 9.5, 10, 15, 95];

% Different SS are dicated by the initial condition
x0High   = [0.1245; 2.4870; 31.2623];
x0Low  = [0.0015; 3.6678; 28.7307]; 

xoutLow = solve(x0Low, @(t,x) MAPK(x, ptrueBistable), tfine, @(t,x) Jac(x,ptrueBistable));
xoutHigh = solve(x0High, @(t,x) MAPK(x, ptrueBistable), tfine, @(t,x) Jac(x,ptrueBistable));

highData = load('./BISTABLE/results_HIGHSS_noisy1.mat').BayesianAnalysis.Data.y;
lowData = load('./BISTABLE/results_LOWSS_noisy1.mat').BayesianAnalysis.Data.y;
oscData = load('./MAPK_osc/results_lessData_run1.mat').BayesianAnalysis.Data.y;


% bistable
figure(1)
plot(tfine, xoutLow(:,1), 'k'); hold on
plot(tfine, xoutHigh(:,1), 'b');
plot(t, [x0Low(1), lowData(1,:)],'.')
plot(t, [x0High(1), highData(1,:)],'.')

saveas(gcf, [savedir, x1expectedBI.png]);
% fname = [savedir, 'x1expectedBI.tex'];
% datPath = [savedir,'x1expectedBI_data/'];
% relDatPath = 'x1expectedBI_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 1

figure(2)
plot(tfine, xoutLow(:,2), 'k'); hold on
plot(tfine, xoutHigh(:,2), 'b');
plot(t, [x0Low(2), lowData(2,:)],'.')
plot(t, [x0High(2), highData(2,:)],'.')

saveas(gcf, [savedir, x2expectedBI.png]);
% fname = [savedir, 'x2expectedBI.tex'];
% datPath = [savedir,'x2expectedBI_data/'];
% relDatPath = 'x2expectedBI_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 2

figure(3)
plot(tfine, xoutLow(:,3), 'k'); hold on
plot(tfine, xoutHigh(:,3), 'b');
plot(t, [x0Low(3), lowData(3,:)],'.')
plot(t, [x0High(3), highData(3,:)],'.')

saveas(gcf, [savedir, x3expectedBI.png]);
% fname = [savedir, 'x3expectedBI.tex'];
% datPath = [savedir,'x3expectedBI_data/'];
% relDatPath = 'x3expectedBI_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Oscillatory Case %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
tend = 3600;
dt=120;
t = t0:dt:tend; 
DT = 20;
tfine = t0:DT:tend;

ptrueOSC = [100,100,100, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 10, 1, 15, 8, 10];

x0 = [10; 80; 80];

xoutOsc = solve(x0, @(t,x) MAPK(x, ptrueOSC), tfine, @(t,x) Jac(x,ptrueOSC));

figure(1)
plot(tfine, xoutOsc(:,1), 'b'); hold on
plot(t, [x0(1), oscData(1,:)],'.')
saveas(gcf, [savedir, x1expectedOSC.png]);
% fname = [savedir, 'x1expectedOSC.tex'];
% datPath = [savedir,'x1expectedOSC_data/'];
% relDatPath = 'x1expectedOSC_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 1

figure(2)
plot(tfine, xoutOsc(:,2), 'b'); hold on
plot(t, [x0(2), oscData(2,:)],'.')
saveas(gcf, [savedir, x2expectedOSC.png]);
% fname = [savedir, 'x2expectedOSC.tex'];
% datPath = [savedir,'x2expectedOSC_data/'];
% relDatPath = 'x2expectedOSC_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 2

figure(3)
plot(tfine, xoutOsc(:,3), 'b'); hold on
plot(t, [x0(3), oscData(3,:)],'.')
saveas(gcf, [savedir, x3expectedOSC.png]);
% fname = [savedir, 'x3expectedOSC.tex'];
% datPath = [savedir,'x3expectedOSC_data/'];
% relDatPath = 'x3expectedOSC_data';
% matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
%     'externalData', true);
close 3

%%%% Functions %%%%
function xout = solve(x0, f, t, jac)
    % sovle the ODE system using ODE15s
    odeOpts = odeset('Jacobian', @(t,x) jac(t,x));
    [~, xout] = ode15s(@(t,x) f(t,x), t, x0, odeOpts);
end