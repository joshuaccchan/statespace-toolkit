% =======================================================================
% This is the main run file for estimating the moving average stochastic
% volatility models in Chan (2013).
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models
% with Application to Inflation Forecast, Journal of Econometrics,
% 176 (2), 162-172.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.
% =======================================================================

clear; clc;
% 1:UC-MA; 2:UCSV-MA; 3:AR(1)-MA; 4:AR(2)-MA;
model = 4;
nloop = 11000;
burnin = 1000;

%% load data
load 'USCPI_Q.csv'; % 1947Q2 to 2011Q3
y0 = USCPI_Q(1); 
y = USCPI_Q(2:end); 
T = length(y);
tid = (1947.5:.25:2011.5)';
options = optimset('Display', 'off') ;
warning off all;

switch model
    case 1 
        UC_MA;        
    case 2        
        UCSV_MA; 
    case 3
        AR1MA;
    case 4
        AR2MA;   
end    