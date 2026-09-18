% This script estimates the marginal likelihood of the HP filter
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

function [ml mlstd] = ml_HP(y,store_theta,prior,lam,M)
M = 20*ceil(M/20);
tmp = gamfit(1./store_theta(:,1));
nusigy2hat = tmp(1); Ssigy2hat = 1./tmp(2);
tau0hat = mean(store_theta(:,2:3))';
tau0var = cov(store_theta(:,2:3));

theta_IS = zeros(M,3);
theta_IS(:,1) = 1./gamrnd(nusigy2hat,1./Ssigy2hat,M,1);
theta_IS(:,2:3) = repmat(tau0hat',M,1) + (chol(tau0var,'lower')*randn(2,M))';

store_w = zeros(M,1);

gIS = @(sy,t0) nusigy2hat*log(Ssigy2hat)-gammaln(nusigy2hat)-(nusigy2hat+1)*log(sy)- Ssigy2hat/sy ...
    -log(2*pi)-.5*log(det(tau0var))-.5*(t0-tau0hat)'*(tau0var\(t0-tau0hat)); 

for loop = 1:M    
    theta = theta_IS(loop,:)';
    sigy2 = theta(1);
    tau0 = theta(2:3);
    
    llike = intlike_UCUR_2M_lam(y,[0;0],sigy2,lam,0,tau0);
        store_w(loop) = llike + prior(sigy2,tau0) - gIS(sigy2,tau0);    
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end