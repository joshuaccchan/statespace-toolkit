% This is the main run file for estimating the unobserved components models
% in Grant and Chan (2016). It also computes the corresponding marginal
% likelihood.
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
% 1: DT; 2: UC0; 3: UCUR; 4: DT-t0; 5: UCUR-t0; 6: UCUR-(t0,t1); 
% 7: bivariate UCUR with one break + unemployment
% 8: bivariate UCUR with one break + inflation
model = 3;
cp_ml = 1;      % 1: compute marginal likelihood 
nsims = 100000;
burnin = 10000;
M = 50000;      % number of replications in ML estimation

%% load data
load 'USGDP.csv';  % 1947Q1-2014Q4
if model <= 6  % univariate models
    y = 100*log(USGDP);
    T = length(y);
    tid = linspace(1947,2014.75,T)';
        % break dates
    t0 = 105;       % 1973:Q1
    % t0 = 237;     % 2006:Q1 
    % t0 = 241;     % 2007:Q1 
    % t0 = 245;     % 2008:Q1 
    % t0 = 249;     % 2009:Q1
    t1 = 241;
else % bivariate model
    load 'USURATE.csv'; % 1948Q1-2014Q4    
    load 'USCPI.csv';   % 1947Q2-2014Q4    
    if model == 7
            % use unemployment
        T = length(USURATE);
        y = reshape([100*log(USGDP(5:end)) USURATE]',T*2,1);
    else
            % use inflation 
        T = length(USCPI(4:end));
        y = reshape([100*log(USGDP(5:end)) USCPI(4:end)]',T*2,1);
    end
    t0 = 237;   % 2007:Q1 (start from 1948Q1)
    tid = linspace(1948,2014.75,T)';    
end

switch model
    case 1 
        DT;        
    case 2        
        UC0;        
    case 3
        UCUR;        
    case 4
        DT_break;      
    case 5
        UCUR_break;        
    case 6
        UCUR_break2;
    case {7, 8}
        biUCUR_break;
end    