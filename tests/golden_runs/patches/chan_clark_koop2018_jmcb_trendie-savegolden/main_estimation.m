%% GOLDEN-RUN VARIANT PATCH (2026-09-19): identical to legacy/main_estimation.m except that
%% a numeric capture is appended at the end, which prints the results the script computes
%% and saves their posterior means and 5% and 95% quantiles to golden_capture.mat.
%% The computation is untouched.
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

%% ===== GOLDEN CAPTURE, appended by the savegolden patch; the computation above is untouched =====
golden_named = {'model'};
golden_draws = [who('store_*'); {}'];
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
