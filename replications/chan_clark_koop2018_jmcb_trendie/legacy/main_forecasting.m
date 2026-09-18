% This is the main run file for the forecasting exercise in 
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
dataset = 2;   % 1: PCE + PTR; 2: CPI + BC
nsim = 30000;
burnin = 1000;
if dataset == 1
        % PCE + PTR; 1960Q1-2016Q1
    data1 = xlsread('cck1_data.xlsx', 'B54:B278');
    data2 = xlsread('cck1_data.xlsx', 'C54:C278'); 
    t0 = [1960 2016]; 
elseif dataset == 2
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

    % recursive forecast exercise
T0 = 20;
yhat1  = zeros(T-1-T0+1,3);   % h=1; [observed y, point forecast, log prelike]
yhat2 = zeros(T-2-T0+1,3);    % h=2; 
yhat4  = zeros(T-4-T0+1,3);   % h=4; 
yhat8  = zeros(T-8-T0+1,3);   % h=8; 
yhat12 = zeros(T-12-T0+1,3);  % h=12;
yhat16 = zeros(T-16-T0+1,3);  % h=16;
yhat20 = zeros(T-20-T0+1,3);  % h=20;
yhat40 = zeros(T-40-T0+1,3);  % h=40;

    % prior
Vpistar = 100;  % implicitly pistar0 = 0;
Vb = 100;       % implicitly b0 = 0;
psi0 = zeros(q,1); Vpsi = .25^2*speye(q);
mud0 = [0 1]'; Vmud = [.1^2 .1^2]';
rhod0 = [.95 .95]'; Vrhod = [.1^2 .1^2]';
nud0 = 5*ones(2,1); Sd0 = [.01 .001]'.*(nud0-1);
nub0 = 5; Sb0 = .001*(nub0-1);
nuw0 = 5; Sw0 = .01*(nuw0-1);
nuv0 = 5; Sv0 = .01*(nuv0-1);
nun0 = 5; Sn0 = .01*(nun0-1);
lamv0 = log(1); Vlamv = 100;
lamn0 = log(1); Vlamn = 100;
d00 = [0 1]'; Vd = [1^2 .5^2]';   
c_psi = 1/(normcdf((1-psi0)/sqrt(Vpsi))-normcdf((-1-psi0)/sqrt(Vpsi)));
lpsipri = @(x) log(c_psi) -.5*log(2*3.14159*Vpsi) -.5*(x-psi0).^2/Vpsi;

options = optimset('Display', 'off') ;
warning off all;

switch model
    case 1 
        model_name = 'M1';
    case 2    
        model_name = 'M2';
    case 3
        model_name = 'M3';
end 

disp(['Starting the recursive forecasting exercise for ' model_name '....']);
disp(' ' );

start_time = clock; 
for t = T0:T-1
    disp([ num2str(t-T) ' more loops to go... ' ] );
    pit = pi(1:t);
    zt = z(1:t);
    Tt = length(pit); 
    
        %% run the forecast model here
    switch model
        case 1
            forecast_M1;
        case 2
            isM2 = true;            
            forecast_M2;
        case 3
            isM2 = false;
            forecast_M2;
    end
    
    yhat1(t-T0+1,:)  = [pi(t+1) mean(tmpyhat1(:,1)) log(mean(tmpyhat1(:,2)))];
    if t<=T-40
        yhat40(t-T0+1,:) = [mean(pi(t+21:t+40)) mean(tmpyhat40(:,1)) log(mean(tmpyhat40(:,2)))];
    end
    if t<=T-20
        yhat20(t-T0+1,:) = [mean(pi(t+17:t+20)) mean(tmpyhat20(:,1)) log(mean(tmpyhat20(:,2)))];
    end
    if t<=T-16
        yhat16(t-T0+1,:) = [mean(pi(t+13:t+16)) mean(tmpyhat16(:,1)) log(mean(tmpyhat16(:,2)))];
    end
    if t<=T-12
        yhat12(t-T0+1,:) = [mean(pi(t+9:t+12)) mean(tmpyhat12(:,1)) log(mean(tmpyhat12(:,2)))];
    end
    if t<=T-8
        yhat8(t-T0+1,:) = [mean(pi(t+5:t+8)) mean(tmpyhat8(:,1)) log(mean(tmpyhat8(:,2)))];
    end
    if t<=T-4
        yhat4(t-T0+1,:) = [mean(pi(t+1:t+4)) mean(tmpyhat4(:,1)) log(mean(tmpyhat4(:,2)))];
    end
    if t<=T-2
        yhat2(t-T0+1,:) = [mean(pi(t+1:t+2)) mean(tmpyhat2(:,1)) log(mean(tmpyhat2(:,2)))];
    end    
    
end

disp( ['Forecasting takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

switch dataset
    case 1
        % forecast evaluation from 1975Q1 to 2016Q1
        tid = (1975:.25:2016)';
        forecasts = [yhat1(40:end,:) yhat2(39:end,:) yhat4(37:end,:) yhat8(33:end,:) ...
            yhat12(29:end,:) yhat16(25:end,:) yhat20(21:end,:) yhat40(1:end,:)];
    case 2
        % forecast evaluation from 1995Q1 to 2016Q1
        tid = (1995:.25:2016)';        
        forecasts = [yhat1(41:end,:) yhat2(40:end,:) yhat4(38:end,:) yhat8(34:end,:) ...
            yhat12(30:end,:) yhat16(26:end,:) yhat20(22:end,:) yhat40(2:end,:)];        
end
RMSFE = zeros(8,1); % 8 forecast horizons
log_predlike = zeros(8,1);
for i_horizon = 1:8
    col = (i_horizon-1)*3+1;
    RMSFE(i_horizon) = sqrt(mean((forecasts(:,col)-forecasts(:,col+1)).^2));
    log_predlike(i_horizon) = sum(forecasts(:,col+2));    
end

clc;
fprintf(['Forecasting results for ' model_name ':'])
fprintf('\n'); 
fprintf('            | RMSFE, log predictive likelihood\n'); 
fprintf('1Q-ahead    | %.2f, %.2f\n', RMSFE(1), log_predlike(1)); 
fprintf('2Q-ahead    | %.2f, %.2f\n', RMSFE(2), log_predlike(2)); 
fprintf('4Q-ahead    | %.2f, %.2f\n', RMSFE(3), log_predlike(3)); 
fprintf('8Q-ahead    | %.2f, %.2f\n', RMSFE(4), log_predlike(4)); 
fprintf('12Q-ahead   | %.2f, %.2f\n', RMSFE(5), log_predlike(5)); 
fprintf('16Q-ahead   | %.2f, %.2f\n', RMSFE(6), log_predlike(6)); 
fprintf('20Q-ahead   | %.2f, %.2f\n', RMSFE(7), log_predlike(7)); 
fprintf('6-10Y-ahead | %.2f, %.2f\n', RMSFE(8), log_predlike(8)); 

figure;
for i_horizon =1:8
    col = (i_horizon-1)*3+1;
    subplot(4,2,i_horizon); plot(tid,[forecasts(:,col) forecasts(:,col+1)]); 
    xlim([tid(1) tid(end)]); box off;    
    switch i_horizon
        case 1
            title_text = '1Q-ahead forecasts';
        case 2
            title_text = '2Q-ahead forecasts';
        case 3
            title_text = '4Q-ahead forecasts';
        case 4
            title_text = '8Q-ahead forecasts';
        case 5
            title_text = '12Q-ahead forecasts'; 
        case 6
            title_text = '16Q-ahead forecasts'; 
        case 7
            title_text = '20Q-ahead forecasts'; 
        case 8
            title_text = '6-10Y-ahead forecasts'; 
            legend('actual obs.', 'forecasts');
            
    end   
    title(title_text);
end
set(gcf,'Position',[100 100 800 800]);

