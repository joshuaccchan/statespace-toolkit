% This is the main run file for estimating the seven SV models in Chan and 
% Grant (2016). It also computes two versions of the DIC.
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
% 1: SV; 2: SV-2; 3: SV-J; 4: SV-M; 5: SV-MA; 6: SV-L; 7: SV-t
model = 1;
DIC_choice = 1;   % 1: observed-data DIC; 2: conditional DIC
nloop = 11000;
burnin = 1000;
R = 10;           % number of parallel chains

%% load data
load 'SP500.csv'; % 2007 to 2012  
id = find(SP500~=0);
y = SP500(id);
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
        SV_L;
    case 7        
        SV_t;
end    


