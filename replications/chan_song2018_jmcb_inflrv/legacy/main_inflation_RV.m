% This is the main run file for estimating the unobserved compoents models
% in Chan and Yong (2018). 
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. and Song, Y. (2018). Measuring Inflation Expectations 
% Uncertainty Using High-Frequency Data, Journal of Money, Credit and 
% Banking, 50(6), 1139-1166.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
% 1: UCSV-RV; 2: UCSV; 3: UCSV-RV-BE; 4: UCSV-RV-h; 5: UCSV-RV-SV;
% 6: UCSV-RV-MA; 7: UCSV-RV-liq; 8: UCSV-RV-D;
model = 1;

% MCMC setting
MCMC.Nuse = 50000;   % # of posterior draws
MCMC.Burnin = 10000; % # of burn-in
load data06_20160608 
y = CPI; 
x = ei510;
z = ei510RV;
logz = log(z);
T = 156;
tid = linspace(2003,2015+11/12,T)';

switch model
    case 1
        model_name = 'UCSV-RV';
        UCSV_RV;
    case 2
        model_name = 'UCSV';
        UCSV;
    case 3
        model_name = 'UCSV-RV-BE';
        UCSV_RV_BE;
    case 4
        model_name = 'UCSV-RV-h';
        UCSV_RV_h;
    case 5
       model_name = 'UCSV-RV-SV';
       UCSV_RV_SV;
    case 6
        model_name = 'UCSV-RV-MA';
        UCSV_RV_MA;
    case 7
        model_name = 'UCSV-RV-liq';
        UCSV_RV_liq;
    case 8
        model_name = 'UCSV-RV-D';
        UCSV_RV_D;
end
