% This script estimates the UCSV model in Chan and Song (2018)
% 
% See:
% Chan, J.C.C. and Song, Y. (2018). Measuring Inflation Expectations 
% Uncertainty Using High-Frequency Data, Journal of Money, Credit and 
% Banking, 50(6), 1139-1166.

disp(['Estimating ' model_name '.... ']);

%% Data information
Data.y = y;
Data.T = T;

%% Prior 
sigma2h = .1; %variance of h equation
sigma2g = .1; %variance of g equation

Prior.vh = 6;
Prior.vg = 6;

Prior.sh = 2 * (Prior.vh/2-1) * sigma2h;
Prior.sg = 2 * (Prior.vg/2-1) * sigma2g;

Prior.Vg = 10; %variance of g1
Prior.mg = 1;

Prior.Vh = 10; %variance of h1
Prior.mh = 1; %calibrated

Prior.mystar = 5; % mean of y1star.
Prior.Vystar = 10 * exp(-Prior.mg); %varaince of y1* is Vystar * exp(g1); variance of y1 is var(y1*) + exp(h1)

%% Initial values
Ini.sigma2h = sigma2h;
Ini.sigma2g = sigma2g;
Ini.ystar = Data.y * 0.9;
Ini.g = zeros(T, 1) + Prior.mg;
Ini.h = zeros(T, 1) + Prior.mh;
    
%% MCMC
rng(123);
Post = sy_SW_original_v3(Data, Prior, Ini, MCMC);

figure;
hold on
    plotCI(tid,quantile(Post.expg',.16)',quantile(Post.expg',.84)');
    plot(tid,quantile(Post.expg',.5)','LineWidth',1.5,'color','blue');
hold off
box off; xlim([tid(1)-.5 tid(end)+.5]); ylim([0 2]); 
title([model_name ': exp(g_t/2) estimates']);




