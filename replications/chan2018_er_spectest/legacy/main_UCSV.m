% This is the main run file for testing if stochastic volatility is needed
% in an unobserved components model for modeling inflation in G7 countries.
% See the first application in Chan (2018)
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter 
% Models with Stochastic Volatility, Econometric Reviews, 37(8), 807-823
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
R = 10;
nloop = 101000;
burnin = 1000;

data = xlsread('OECD_G7CPI.xls', 'D51:J266'); 
country_id = 7; %1: Canada; 2: France; 3: Germany; 4: Italy; 5: Japan; 6: UK; 7: US

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;    

store_lBF = zeros(R,3);
for bigloop = 1:R
    disp(  [ num2str( R-bigloop+1) ' big loops to go... ' ] )
    y = log(data(2:end,country_id)./data(1:end-1,country_id))*400;       
    UCSV_gam;
    theta_qan = quantile(store_theta,[.05 .95]);
    store_lBF(bigloop,:) = lBF;        
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );
BFhat = mean(store_lBF)';
BFstd = std(store_lBF)/sqrt(R);

% report estimates
fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('omega2_g    | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 

% report log BFs
fprintf('\n'); 
fprintf('log Bayes Factor | estimate (NSE):\n'); 
fprintf('log BF_uh        | %.1f (%.2f)\n', BFhat(1), BFstd(1)); 
fprintf('log BF_ug        | %.1f (%.2f)\n', BFhat(2), BFstd(2)); 
fprintf('log BF_u,gh      | %.1f (%.2f)\n', BFhat(3), BFstd(3)); 

% plot some figures
tid = linspace(1955,2013.75,T)';
figure;
subplot(1,2,1);
hold on
    plot(omh_grid,pomhhat,'LineWidth',1,'Color','blue');
    plot(omh_grid,priden_omh,'--','LineWidth',1,'Color','red');
hold off
title('\omega_h');
subplot(1,2,2);
hold on
    plot(omg_grid,pomghat,'LineWidth',1,'Color','blue'); 
    plot(omg_grid,priden_omg,'--','LineWidth',1,'Color','red');
hold off
box off; legend('posterior','prior',1);
title('\omega_g');
set(gcf,'Position',[100 100 800 300]);

% plot h and g
figure
subplot(1,2,1); 
hold on 
    plotCI(tid,hCI(:,1),hCI(:,2)); 
    plot(tid,hhat,'LineWidth',1,'Color','blue');     
hold off
xlim([tid(1)-.5,tid(end)+.5]); box off;
title('exp(h_t/2)');
subplot(1,2,2); 
hold on 
    plotCI(tid,gCI(:,1),gCI(:,2)); 
    plot(tid,ghat,'LineWidth',1,'Color','blue');     
hold off
xlim([tid(1)-.5,tid(end)+.5]); box off;
title('exp(g_t/2)');
set(gcf,'Position',[100 100 800 300]);
