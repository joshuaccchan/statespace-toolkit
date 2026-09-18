% This script estimates the marginal likelihood of the standard SV model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [ml mlstd] = ml_sv(y,store_theta,hhat,prior,M)
% ht is the starting value for the Newton-Raphson

M = 20*ceil(M/20);
thetahat = mean(store_theta)';
thetavar = var(store_theta);
thetastd = sqrt(thetavar);
temp = gamfit(1./store_theta(:,4));
nuomegah2hat = temp(1); Somegah2hat = 1./temp(2);    

theta_IS = zeros(M,4);
theta_IS(:,1) = thetahat(1) + thetastd(1)*randn(M,1); 
theta_IS(:,2) = thetahat(2) + thetastd(2)*randn(M,1);
theta_IS(:,3) = tnormrnd(thetahat(3),thetavar(3),-.999,.999,M);
theta_IS(:,4) = 1./gamrnd(nuomegah2hat,1./Somegah2hat,M,1);
store_w = zeros(M,1);

phih_const = 1/(normcdf(1,thetahat(3),thetastd(3))-normcdf(-1,thetahat(3),thetastd(3)));
gIS = @(m,mh,ph,oh) -.5*log(2*pi*thetavar(1)) -.5*(m-thetahat(1))^2/thetavar(1)...
    -.5*log(2*pi*thetavar(2)) - .5*(mh-thetahat(2))^2/thetavar(2) ...
    -.5*log(2*pi*thetavar(3)) + log(phih_const) -.5*(ph-thetahat(3))^2/thetavar(3) ...
    + nuomegah2hat*log(Somegah2hat)-gammaln(nuomegah2hat)-(nuomegah2hat+1)*log(oh)- Somegah2hat/oh; 
for loop = 1:M
    theta = theta_IS(loop,:)';
    mu = theta(1);
    muh = theta(2);
    phih = theta(3);
    omegah2 = theta(4);    
    s2 = (y-mu).^2;     
    llike = intlike_sv(s2,muh,phih,omegah2,hhat,50);    
    store_w(loop) = llike + prior(mu,muh,phih,omegah2) - gIS(mu,muh,phih,omegah2);    
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end