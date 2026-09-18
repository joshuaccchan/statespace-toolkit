% This script estimates the marginal likelihood of the SV-2 SV model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [ml mlstd] = ml_sv_2(y,store_theta,hhat,prior,M)
% ht is the starting value for the Newton-Raphson

M = 20*ceil(M/20);
thetahat = mean(store_theta)';
thetavar = var(store_theta);
thetastd = sqrt(thetavar);
temp = gamfit(1./store_theta(:,5));
nuomegah2hat = temp(1); Somegah2hat = 1./temp(2);    

store_w = zeros(M,1);
theta_IS = zeros(M,5);
theta_IS(:,1) = thetahat(1) + thetastd(1)*randn(M,1); 
theta_IS(:,2) = thetahat(2) + thetastd(2)*randn(M,1);
theta_IS(:,5) = 1./gamrnd(nuomegah2hat,1./Somegah2hat,M,1);

R = 10^5; %% estimate the prob that (phih,rhoh) is in the stationarity region
phihhat = thetahat(3:4);
phihcov = cov(store_theta(:,3:4));
phihldet = log(det(phihcov));
temp = repmat(phihhat',R,1) + (chol(phihcov,'lower')*randn(2,R))';
id = find((sum(temp,2)< .999)&(temp(:,2)-temp(:,1)<.999)&(abs(temp(:,2))<.999));
phih_const = -log(length(id)/R);
gIS = @(m,mh,ph,rh,oh) -.5*log(2*pi*thetavar(1)) -.5*(m-thetahat(1))^2/thetavar(1)...
    -.5*log(2*pi*thetavar(2)) - .5*(mh-thetahat(2))^2/thetavar(2) ...
    -log(2*pi) - .5*phihldet + log(phih_const) ...
    -.5*(([ph rh] - phihhat')*(phihcov\([ph;rh]-phihhat))) ...
    + nuomegah2hat*log(Somegah2hat)-gammaln(nuomegah2hat) ...
    -(nuomegah2hat+1)*log(oh)- Somegah2hat/oh; 
theta_IS(:,3:4) = temp(id(1:M),:); %% use only those (phih,rhoh) that are stationary 

for loop = 1:M
    theta = theta_IS(loop,:)';
    mu = theta(1);
    muh = theta(2);
    phih = theta(3);
    rhoh = theta(4);
    omegah2 = theta(5);    
    s2 = (y-mu).^2;     
    llike = intlike_sv_2(s2,muh,phih,rhoh,omegah2,hhat,50);
    store_w(loop) = llike + prior(mu,muh,phih,rhoh,omegah2) - gIS(mu,muh,phih,rhoh,omegah2);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end