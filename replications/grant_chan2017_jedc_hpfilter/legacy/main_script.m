% This is the main run file for estimating the new unobserved components
% model with a second-order Markov process for the trend in Grant and 
% Chan (2017). It also computes the corresponding marginal likelihood.
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
model = 1;      % 1: UCUR-2M; 2: HP; 3: HP-AR
cp_ml = 1;      % 1: compute marginal likelihood 
nsims = 100000;
burnin = 10000;
M = 50000;      % number of replications in ML estimation

%% load data
load 'USGDP.csv';  % 1947Q1-2014Q4
y = 100*log(USGDP);
T = length(y);
tid = linspace(1947,2014.75,T)';

switch model
    case 1 
        UCUR_2M;        
    case 2        
        HP;
    case 3
        HP_AR;
end    