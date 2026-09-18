% This script estimates the marginal likelihood of the HP-AR model
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

function [ml mlstd] = ml_HP_AR(y,store_theta,prior,M)
M = 20*ceil(M/20);
lam = 1600;

phihat = mean(store_theta(:,1:2))';
phivar = cov(store_theta(:,1:2));

tmp = gamfit(1./store_theta(:,3));
nusigc2hat = tmp(1); Ssigc2hat = 1./tmp(2);
tau0hat = mean(store_theta(:,4:5))';
tau0var = cov(store_theta(:,4:5));

theta_IS = zeros(M,5);
theta_IS(:,1:2) = repmat(phihat',M,1) + (chol(phivar,'lower')*randn(2,M))';
theta_IS(:,3) = 1./gamrnd(nusigc2hat,1./Ssigc2hat,M,1);
theta_IS(:,4:5) = repmat(tau0hat',M,1) + (chol(tau0var,'lower')*randn(2,M))';

store_w = zeros(M,1);

gIS = @(ph,sy,t0) -log(2*pi)-.5*log(det(phivar))-.5*(ph-phihat)'*(phivar\(ph-phihat))...
    + nusigc2hat*log(Ssigc2hat)-gammaln(nusigc2hat)-(nusigc2hat+1)*log(sy)- Ssigc2hat/sy ...
    -log(2*pi)-.5*log(det(tau0var))-.5*(t0-tau0hat)'*(tau0var\(t0-tau0hat)); 

for isim = 1:M    
    theta = theta_IS(isim,:)';
    phi = theta(1:2);
    sigc2 = theta(3);    
    tau0 = theta(4:5);
    
    llike = intlike_HP_aug(y,phi,sigc2,tau0,lam);
    store_w(isim) = llike + prior(phi,sigc2,tau0) - gIS(phi,sigc2,tau0);    
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end