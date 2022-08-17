% Nathaniel Linden
% UCSD MAE
% Script to run GSA for the MAPK model
clear all; close all; clc

addpath('../utils/')
plottingPreferencesNJL;

% folder to save results
savedir = './sensitivity_analysis/';
mkdir(savedir); 


% Exclude non-identifiable or exponents from analysis
fixedParamIndex = [1,2,3,10,11,12,13]; nparams = 14;
freeParamIndex = setdiff(1:nparams, fixedParamIndex);
paramNames = {'S1t','S2t','S3t','k1', 'k2','k3','k4','k5','k6','n1','K1','n2','K2','alpha'};
paramNames = paramNames(freeParamIndex);

% Low steady-state
load([savedir, 'lowSSGSA.mat']);
qoiNames = {'x1-ss', 'x2-ss', 'x3-ss'};
plotGSAResults(lowSensitivtyResults, savedir, '_lowSSGSA', paramNames, qoiNames)
close all

% High steady-state
load([savedir, 'highSSGSA.mat']);
qoiNames = {'x1-ss', 'x2-ss', 'x3-ss'};
plotGSAResults(lowSensitivtyResults, savedir, '_highSSGSA', paramNames, qoiNames)

% Oscillations
load([savedir, 'oscGSA.mat']);
qoiNames = {'LCA', 'period', 'x3-mean'};
plotGSAResults(oscSensitivtyResults, savedir, '_oscGSA', paramNames, qoiNames)