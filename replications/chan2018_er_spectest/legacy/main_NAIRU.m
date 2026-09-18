% This is the main run file for testing if the NAIRU is time varying
% in a bivariate unobserved components model for inflation and
% unemployment.
% See the second application in Chan (2018)
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
R = 3;
nloop = 51000;
burnin = 1000;

load 'USdata.csv';
y0 = USdata(3,1);
y = USdata(4:end,1); 
u0 = USdata(1:3,2);
u = USdata(4:end,2);
T = length(y);
    
disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;    

store_lBF = zeros(R,1);
for bigloop = 1:R
    TV_NAIRU_AR2;
    store_lBF(bigloop) = lBF;    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );
BFhat = mean(store_lBF)';
BFstd = std(store_lBF)/sqrt(R);  
        
% report estimates
fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 

fprintf('lambda      | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('phi_1       | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('phi_2       | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('sigma2_u    | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('omega2_g    | %.2f (%.2f)\n', thetahat(6), thetastd(6)); 
fprintf('omega2_nu   | %.2f (%.2f)\n', thetahat(7), thetastd(7)); 

% report log BFs
fprintf('\n'); 
fprintf('log BF: %.1f (%.2f)\n', BFhat, BFstd); 




figure;
hold on
    plot(omnu_grid,pomnuhat,'LineWidth',1,'Color','blue'); box off;
    plot(omnu_grid,priden_omnu,'--','LineWidth',1,'Color','red'); box off;
hold off
legend('posterior','prior',1);

tid = linspace(1948,2013.75,T)';
figure;
hold on 
    plotCI(tid,nuCI(:,1),nuCI(:,2)); 
    plot(tid,nuhat,'LineWidth',1,'Color','black'); 
    plot(tid,u,'--','LineWidth',1,'Color','blue'); 
hold off
xlim([tid(1)-.5,tid(end)+.5]); ylim([0 12 ]);


% 
% % plot some figures
% tid = linspace(1955,2013.75,T)';
% figure;
% subplot(1,2,1);
% hold on
%     plot(omh_grid,pomhhat,'LineWidth',1,'Color','blue');
%     plot(omh_grid,priden_omh,'--','LineWidth',1,'Color','red');
% hold off
% title('\omega_h');
% subplot(1,2,2);
% hold on
%     plot(omg_grid,pomghat,'LineWidth',1,'Color','blue'); 
%     plot(omg_grid,priden_omg,'--','LineWidth',1,'Color','red');
% hold off
% box off; legend('posterior','prior',1);
% title('\omega_g');
% set(gcf,'Position',[100 100 800 300]);
% 
% % plot h and g
% figure
% subplot(1,2,1); 
% hold on 
%     plotCI(tid,hCI(:,1),hCI(:,2)); 
%     plot(tid,hhat,'LineWidth',1,'Color','blue');     
% hold off
% xlim([tid(1)-.5,tid(end)+.5]); box off;
% title('exp(h_t/2)');
% subplot(1,2,2); 
% hold on 
%     plotCI(tid,gCI(:,1),gCI(:,2)); 
%     plot(tid,ghat,'LineWidth',1,'Color','blue');     
% hold off
% xlim([tid(1)-.5,tid(end)+.5]); box off;
% title('exp(g_t/2)');
% set(gcf,'Position',[100 100 800 300]);


 