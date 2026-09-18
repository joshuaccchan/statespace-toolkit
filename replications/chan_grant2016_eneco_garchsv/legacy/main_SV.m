% This is the main run file for estimating the seven SV models in Chan and 
% Grant (2016). It also computes the corresponding marginal likelihood.
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
% 1: SV; 2: SV-2; 3: SV-J; 4: SV-M; 5: SV-MA; 6: SV-t; 7: SV-L
model = 1;
cp_ml = 1;      % 1: compute marginal likelihood
nloop = 11000;
burnin = 1000;
M = 10000;      % number of replications in ML estimation
series = 5;     % choice of data series (1 to 9)
nlag = 20;      % number of lags for computing the Ljung-Box Q-stat

%% load data
load 'data4.csv';
data4 = 100*diff(log(data4(:,:)));
id = find(data4(:,series)~=0);
y = data4(id,series);
T = length(y);    
tid = linspace(2007,2013,T)';
    
switch model
    case 1 
        SV;        
    case 2        
        SV_2;
    case 3
        SV_J;
    case 4
        SV_M;
    case 5
        SV_MA;
    case 6
        SV_t;
    case 7
        SV_L;
end    


