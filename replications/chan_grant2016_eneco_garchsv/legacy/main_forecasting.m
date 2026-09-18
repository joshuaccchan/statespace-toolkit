% This is the main run file for forecasting using the seven SV and GARCH
% models in Chan and Grant (2016). 
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
%
% Sample period: 03 Jan 1997 - 6 Feb 2015
% Evaluation period: Jan 2000 - end of sample
% Forecast horizon: 1

clear; clc;
% 1: SV; 2: SV-2; 3: SV-J; 4: SV-M; 5: SV-MA; 6: SV-t; 7: SV-L
% 8: GARCH; 9: GARCH-2; 10: GARCH-J; 11: GARCH-M; 12: GARCH-MA; 13: GARCH-t; 
% 14: GARCH-GJR
model = 1;
nsims = 10000;
burnin = 1000;
series = 2; % choice of data series (1 to 9)
load 'data4.csv';
data4 = 100*diff(log(data4(:,:)));
id = find(data4(:,series)~=0);
y = data4(id,series);
T = length(y);
     
    % recursive forecast exercise
T0 = 159; % start from 2000 Jan
yhat1  = zeros(T-1-T0+1,3);   % h=1; [observed y, point forecast, log prelike]

    % prior
phih0 = .97; Vphih = .1^2;
mu0 = 0; Vmu = 10;
muh0 = 1; Vmuh = 10;
nuh = 5; Sh = .2^2*(nuh-1);
gam0 = log([exp(1) .1 .8]'); Vgam = diag([10 1 1]);
rhoh0 = 0; Vrhoh = 1;
delta0 = [0 log(10)]'; Vdelta = diag([10 1]);
lam0 = 0; Vlam = 100;
alp0 = 0; Valp = 100; 
psi0 = 0; Vpsi = 1;
nuub = 100; % upperbound for nu
rho0 = 0; Vrho = 1;

disp('Starting the recursive forecasting exercise.... ');
disp(' ' );
start_time = clock; 
for t = T0:T-1
    disp([ num2str(t-T) ' more loops to go... ' ] );
    yt = y(1:t,:);
    Tt = size(yt,1);     
        % run the forecast model here 
    switch model
        case 1
            forecast_SV;
        case 2
            forecast_SV2;
        case 3
            forecast_SV_J;   
        case 4
            forecast_SV_m;
        case 5
            forecast_SV_ma;
        case 6
            forecast_SVt;
        case 7
            forecast_SVL;
        case 8
             forecast_GARCH;
        case 9
            forecast_GARCH2;
        case 10 
            forecast_GARCH_J;
        case 11 
            forecast_GARCH_m;
        case 12
            forecast_GARCH_ma;
        case 13 
            forecast_GARCHt;
        case 14
            forecast_GARCH_gjr;
    end    
    yhat1(t-T0+1,:)  = [y(t+1) mean(tempyhat1(:,1)) log(mean(exp(tempyhat1(:,2))))]; 
end

disp( ['The forecasting exercise takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

logscore = sum(yhat1(:,3))';
fprintf(['The log predictive score for ' model_name ' is %.1f '], logscore);
fprintf('\n'); 

