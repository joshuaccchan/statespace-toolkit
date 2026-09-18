% This script estimates the UCSV-RV-SV model in Chan and Song (2018)
% 
% See:
% Chan, J.C.C. and Song, Y. (2018). Measuring Inflation Expectations 
% Uncertainty Using High-Frequency Data, Journal of Money, Credit and 
% Banking, 50(6), 1139-1166.

disp(['Estimating ' model_name '.... ']);

%% Data information
Data.y = y;
Data.T = T;
Data.z = z;

%% Prior 
sigma2h = 0.1; %variance of h equation (y volatility)
sigma2g = 0.1; %variance of g equation -- log volatility of ystar (expectation)
sigma2v = 0.1; %variance of v equation

ma =  [0; 1]; %coefficients of x equation

Prior.vh = 6;
Prior.vg = 6;
Prior.vv = 6;

Prior.sh = 2 * (Prior.vh/2-1) * sigma2h;
Prior.sg = 2 * (Prior.vg/2-1) * sigma2g;
Prior.sv = 2 * (Prior.vv/2-1) * sigma2v;

Prior.Vg = 10; %variance of g1
Prior.mg = 1;  %mean of g1

Prior.Vystar = 10 * exp(-Prior.mg); %varaince of y1* is Vystar * exp(g1); variance of y1 is var(y1*) + exp(h1)
Prior.mystar = 5;

Prior.Vh = 10; %variance of h1
Prior.mh = 1; %mean of h1

Prior.Vv = 10; %variance of v1
Prior.mv = 0.5; %mean of v1

Prior.Ka = eye(2); %precision of coefficients of the logz equation
Prior.ma = ma;

%% Initial values
Ini.sigma2h = sigma2h;
Ini.sigma2g = sigma2g;
Ini.sigma2v = sigma2v;
Ini.ystar = Data.y * 0.5;
Ini.g = zeros(T, 1) + Prior.mg;
Ini.h = zeros(T, 1) + Prior.mh;
Ini.v = zeros(T, 1) + Prior.mv;
Ini.a = ma;
    
%% MCMC
rng(123);
Post = sy_CS4(Data, Prior, Ini, MCMC);

figure;
hold on
    plotCI(tid,quantile(Post.expg',.16)',quantile(Post.expg',.84)');
    plot(tid,quantile(Post.expg',.5)','LineWidth',1.5,'color','blue');
hold off
box off; xlim([tid(1)-.5 tid(end)+.5]); ylim([0 2]); 
title([model_name ': exp(g_t/2) estimates']);