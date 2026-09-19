%% GOLDEN-RUN VARIANT PATCH (2026-09-19): identical to legacy/MASV_matlab/main_UCMA.m except that
%% a numeric capture is appended at the end, which prints the results the script computes
%% and saves their posterior means and 5% and 95% quantiles to golden_capture.mat.
%% The computation is untouched.
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

%% ===== GOLDEN CAPTURE, appended by the savegolden patch; the computation above is untouched =====
golden_named = {};
golden_draws = [who('store_*'); {'stheta', 'sbeta', 'spsi', 'sh'}'];
golden = struct('matlab', version);
for golden_v = golden_named(:)'
    if exist(golden_v{1}, 'var')
        golden.(golden_v{1}) = eval(golden_v{1});
        fprintf('%s = %s\n', golden_v{1}, mat2str(golden.(golden_v{1}), 8));
    end
end
for golden_v = golden_draws(:)'
    if exist(golden_v{1}, 'var')
        golden_x = eval(golden_v{1});
        if isnumeric(golden_x) && ismatrix(golden_x) && size(golden_x,1) > 1
            golden.([golden_v{1} '_mean']) = mean(golden_x, 1);
            golden.([golden_v{1} '_q05']) = quantile(golden_x, .05, 1);
            golden.([golden_v{1} '_q95']) = quantile(golden_x, .95, 1);
            if size(golden_x, 2) <= 12
                fprintf('%s posterior means: %s\n', golden_v{1}, mat2str(mean(golden_x, 1), 6));
            end
        end
    end
end
save('golden_capture.mat', 'golden');
