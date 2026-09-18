% This script estimates the marginal likelihood of the SV-J model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [ml mlstd] = ml_sv_j(y,store_theta,hhat,prior,M)
% ht is the starting value for the Newton-Raphson

M = 20*ceil(M/20);
thetahat = mean(store_theta(:,1:3))';
thetavar = var(store_theta(:,1:3));
thetastd = sqrt(thetavar(:,1:3));
temp = gamfit(1./store_theta(:,4));
nuomegah2hat = temp(1); Somegah2hat = 1./temp(2); 
phat = betafit(store_theta(:,5));
store_delta = [store_theta(:,6) log(store_theta(:,7))];
deltahat = mean(store_delta)';
deltacov = cov(store_delta);
deltastd = chol(deltacov,'lower');

theta_IS = zeros(M,7);
theta_IS(:,1) = thetahat(1) + thetastd(1)*randn(M,1); 
theta_IS(:,2) = thetahat(2) + thetastd(2)*randn(M,1);
theta_IS(:,3) = tnormrnd(thetahat(3),thetavar(3),-.999,.999,M);
theta_IS(:,4) = 1./gamrnd(nuomegah2hat,1./Somegah2hat,M,1);
theta_IS(:,5) = betarnd(phat(1),phat(2),M,1);
theta_IS(:,6:7) = repmat(deltahat',M,1) + (deltastd*randn(2,M))';

store_w = zeros(M,1);
phih_const = 1/(normcdf(1,thetahat(3),thetastd(3))-normcdf(-1,thetahat(3),thetastd(3)));
gIS = @(m,mh,ph,oh,k,d) -.5*log(2*pi*thetavar(1)) -.5*(m-thetahat(1))^2/thetavar(1)...
    -.5*log(2*pi*thetavar(2)) - .5*(mh-thetahat(2))^2/thetavar(2) ...
    -.5*log(2*pi*thetavar(3)) + log(phih_const) -.5*(ph-thetahat(3))^2/thetavar(3) ...
    + nuomegah2hat*log(Somegah2hat)-gammaln(nuomegah2hat)-(nuomegah2hat+1)*log(oh)- Somegah2hat/oh ...
    + (phat(1)-1)*log(k) + (phat(2)-1)*log(1-k) - betaln(phat(1),phat(2)) ...
    - log(2*pi) -.5*log(det(deltacov)) -.5*(d-deltahat)'*(deltacov\(d-deltahat));
    
for loop = 1:M
    theta = theta_IS(loop,:)';
    mu = theta(1);
    muh = theta(2);
    phih = theta(3);
    omegah2 = theta(4);    
    kappa = theta(5);
    delta = theta(6:7);
    llike = intlike_sv_j(y,mu,kappa,delta,muh,phih,omegah2,hhat,50);
    store_w(loop) = llike + prior(mu,muh,phih,omegah2,kappa,delta) ...
        - gIS(mu,muh,phih,omegah2,kappa,delta);    
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end