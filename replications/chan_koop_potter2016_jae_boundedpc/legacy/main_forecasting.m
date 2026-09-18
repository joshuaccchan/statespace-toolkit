% This is the main run file for the forecasting exercise in 
% Chan, Koop and Potter (2016)
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C., Koop, G. and Potter, S.M. (2016). A Bounded Model of Time 
% Variation in Trend Inflation, NAIRU and the Phillips Curve, Journal of 
% Applied Econometrics, 31(3), 551-565.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
model = 1;     % 1: Bi-UC; 2: Bi-UC-const-lambda; 3: Bi-UC-const-rhopi; 
               % 4: Bi-UC-const-lambda-rhopi;
nsim = 40000;
burnin = 1000;

load 'USdata.csv';
y0 = USdata(2,1);
y = USdata(3:end,1); 
u0 = USdata(1:2,2);
u = USdata(3:end,2);
T = length(y);

    % recursive forecast exercise
T0 = 91; % starts from 1971Q1
yhat1  = zeros(T-1-T0+1,7);   % h=1; [observations, point forecasts, log prelike]
yhat4  = zeros(T-4-T0+1,7);   % h=4; 
yhat8  = zeros(T-8-T0+1,7);   % h=8; 
yhat12 = zeros(T-12-T0+1,7);  % h=12; 
yhat16 = zeros(T-16-T0+1,7);  % h=16; 

    % prior
api = 0; bpi = 5;
au = 4; bu = 7;
taupi0 = 0; invVtaupi = 1/5; sqrtinvVtaupi = sqrt(invVtaupi);
tauu0 = [5;5]; invVtauu = 1/5; sqrtinvVtauu = sqrt(invVtauu);
invVrhopi = 1; sqrtinvVrhopi = sqrt(invVrhopi);
invVrhou = speye(2)/5;
invVlam = 1; sqrtinvVlam = sqrt(invVlam);
invVh = 1/5; 
nutaupi0 = 20; Staupi0 = 2*.02*(nutaupi0/2-1);
nutauu0 = 20; Stauu0 = 2*.01*(nutauu0/2-1);
nurhopi0 = 20; Srhopi0 = 2*.002*(nurhopi0/2-1);
nulam0 = 20; Slam0 = 2*.002*(nulam0/2-1);
nuh0 = 20; Sh0 = 2*.1*(nuh0/2-1);
nuu0 = 20; Su0 = 2*.1*(nuu0/2-1);

switch model
    case 1 
        model_name = 'Bi-UC';
    case 2    
        model_name = 'Bi-UC-const-lambda';
    case 3
        model_name = 'Bi-UC-const-rhopi';
    case 4
        model_name = 'Bi-UC-const-lambda-rhopi';
end 

disp(['Starting the recursive forecasting exercise for ' model_name '....']);
disp(' ' );

start_time = clock; 
for t = T0:T-1
    disp([ num2str(t-T) ' more loops to go... ' ] );
    yt = y(1:t);
    ut = u(1:t);
    Tt = length(yt);
    
        %% run the forecast model here
    switch model
        case 1
            tvp_lam = 1;
            tvp_rhopi = 1;
        case 2
            tvp_lam = 0;
            tvp_rhopi = 1;
        case 3          
            tvp_lam = 1;
            tvp_rhopi = 0;
        case 4
            tvp_lam = 0;
            tvp_rhopi = 0;
    end
    forecast_biUC;
    
    yhat1(t-T0+1,:)  = [y(t+1) u(t+1) mean(tmpyhat1(:,1:2)) log(mean(tmpyhat1(:,3:end)))]; 
    if  t<=T-4
        yhat4(t-T0+1,:) = [y(t+4) u(t+4) mean(tmpyhat4(:,1:2)) log(mean(tmpyhat4(:,3:end)))]; 
    end
    if  t<=T-8
        yhat8(t-T0+1,:) = [y(t+8) u(t+8) mean(tmpyhat8(:,1:2)) log(mean(tmpyhat8(:,3:end)))]; 
    end
    if  t<=T-12
        yhat12(t-T0+1,:) = [y(t+12) u(t+12) mean(tmpyhat12(:,1:2)) log(mean(tmpyhat12(:,3:end)))]; 
    end
    if  t<=T-16
        yhat16(t-T0+1,:) = [y(t+16) u(t+16) mean(tmpyhat16(:,1:2)) log(mean(tmpyhat16(:,3:end)))]; 
    end    
end

disp( ['Forecasting takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

% forecast evaluation from 1975Q1 to 2013Q1
tid = (1975:.25:2013)';
forecasts = [yhat1(16:end,:) yhat4(13:end,:) yhat8(9:end,:) ...
    yhat12(5:end,:) yhat16(1:end,:) ];
RMSFE = zeros(5,2); % 5 forecast horizons, 2 variables (CPI, urate)
log_predlike = zeros(5,3); % 2 individual forecasts + 1 joint forecasts
for i_horizon = 1:5
    col = (i_horizon-1)*7+1;
    RMSFE(i_horizon,1) = sqrt(mean((forecasts(:,col)-forecasts(:,col+2)).^2)); % CPI
    RMSFE(i_horizon,2) = sqrt(mean((forecasts(:,col+1)-forecasts(:,col+3)).^2)); % urate
    log_predlike(i_horizon,:) = sum(forecasts(:,col+4:col+6));
end
 
clc;
fprintf(['RMSFEs for ' model_name ':'])
fprintf('\n'); 
fprintf('            | CPI,  Urate\n'); 
fprintf('1Q-ahead    | %.2f, %.2f\n', RMSFE(1,:));
fprintf('4Q-ahead    | %.2f, %.2f\n', RMSFE(2,:));
fprintf('8Q-ahead    | %.2f, %.2f\n', RMSFE(3,:));
fprintf('12Q-ahead   | %.2f, %.2f\n', RMSFE(4,:)); 
fprintf('16Q-ahead   | %.2f, %.2f\n', RMSFE(5,:)); 
fprintf('\n');

fprintf(['log-predictive likelihoods for ' model_name ':'])
fprintf('\n'); 
fprintf('            | Joint,   CPI,   Urate, \n'); 
fprintf('1Q-ahead    | %.0f, %.0f, %.0f\n', log_predlike(1,:)); 
fprintf('4Q-ahead    | %.0f, %.0f, %.0f\n', log_predlike(2,:)); 
fprintf('8Q-ahead    | %.0f, %.0f, %.0f\n', log_predlike(3,:)); 
fprintf('12Q-ahead   | %.0f, %.0f, %.0f\n', log_predlike(4,:)); 
fprintf('16Q-ahead   | %.0f, %.0f, %.0f\n', log_predlike(5,:)); 

figure;
for i_horizon =1:5
    col = (i_horizon-1)*7+1;
    subplot(3,2,i_horizon); plot(tid,[forecasts(:,col) forecasts(:,col+2)]); 
    xlim([tid(1) tid(end)]); box off;    
    switch i_horizon
        case 1
            title_text = '1Q-ahead forecasts';
        case 2            
            title_text = '4Q-ahead forecasts';
        case 3
            title_text = '8Q-ahead forecasts';
        case 4
            title_text = '12Q-ahead forecasts'; 
        case 5
            title_text = '16Q-ahead forecasts'; 
            legend('CPI', 'forecasts');
    end   
    title(title_text);
end
set(gcf,'Position',[100 100 800 800]);

figure;
for i_horizon =1:5
    col = (i_horizon-1)*7+1;
    subplot(3,2,i_horizon); plot(tid,[forecasts(:,col+1) forecasts(:,col+3)]); 
    xlim([tid(1) tid(end)]); box off;    
    switch i_horizon
        case 1
            title_text = '1Q-ahead forecasts';
        case 2            
            title_text = '4Q-ahead forecasts';
        case 3
            title_text = '8Q-ahead forecasts';
        case 4
            title_text = '12Q-ahead forecasts'; 
        case 5
            title_text = '16Q-ahead forecasts'; 
            legend('Urate', 'forecasts');
    end   
    title(title_text);
end
set(gcf,'Position',[100 100 800 800]);