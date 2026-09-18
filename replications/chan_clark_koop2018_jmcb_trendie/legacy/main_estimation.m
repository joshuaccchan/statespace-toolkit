% This is the main run file for estimating the trend inflation models in 
% Chan, Clark and Koop (2018)
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
model = 1;     % 1: M1; 2: M2; 3: M3
dataset = 1;   % 1: PCE + PTR; 2: CPI + BC
nsim = 30000;
burnin = 1000;

switch dataset
    case 1
        % PCE + PTR; 1960Q1-2016Q1
    data1 = xlsread('cck1_data.xlsx', 'B54:B278');
    data2 = xlsread('cck1_data.xlsx', 'C54:C278'); 
    t0 = [1960 2016]; 
    case 2
        % CPI + Blue Chip 6-10 year ahead (CPI); 1979Q4-2016Q1
    data1 = xlsread('cck1_data.xlsx', 'D133:D278');
    data2 = xlsread('cck1_data.xlsx', 'E133:E278');    
    t0 = [1980 2016]; 
end
pi0 = data1(1,1);    
pi = data1(2:end,1); 
Einf = data2;
z0 = Einf(1);
z = Einf(2:end);
T = length(pi);
q = 1;    

switch model
    case 1
        model_name = 'M1';
        M1;        
    case 2
        model_name = 'M2';
        isM2 = true;
        M2;        
    case 3
        model_name = 'M3';
        isM2 = false;
        M2;     
end
