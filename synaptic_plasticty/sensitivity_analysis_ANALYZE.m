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

% Exclude non-identifiable or exponents from analysis
fixedParamIndex = []; % no fixed params - all params are ID
paramNames = {'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'c1', 'c2', 'c3', 'c4', 'Km1', ...
        'Km2', 'Km3', 'Km4', 'Km5', 'K0', 'P0', 'Ktot', 'Ptot', 'Atot'};
freeParamIndex = setdiff(1:numel(paramNames), fixedParamIndex);
paramNames = paramNames(freeParamIndex);

% LTP induced steady-state
load([savedir, 'ltpGSA.mat']);

% return
qoiNames = {'P-ss', 'pK-ss', 'A-ss', 'EPSP-ss'};
plotGSAResults(ltpSensitivtyResults, savedir, '_ltpGSA', paramNames, qoiNames)
close all

% LTD induced steady-state
load([savedir, 'ltdGSA.mat']);
qoiNames = {'P-ss', 'pK-ss', 'A-ss', 'EPSP-ss'};
plotGSAResults(ltdSensitivtyResults, savedir, '_ltdGSA', paramNames, qoiNames)