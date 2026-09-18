% This script estimates the marginal likelihood of the UC model with 
% correlated errors and twoe breaks
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552

function [ml mlstd] = ml_UCUR_break2(y,store_theta,t0,t1,prior,M)
% [mu, phi, sigc2, sigtau2, rho, tau0]
M = 20*ceil(M/20);
muhat = mean(store_theta(:,1:3))';
muvar = var(store_theta(:,1:3))';
phihat = mean(store_theta(:,4:5))';
phivar = cov(store_theta(:,4:5));
temp = gamfit(1./store_theta(:,6));
nusigc2hat = temp(1); Ssigc2hat = 1./temp(2);
temp = gamfit(1./store_theta(:,7));
nusigtau2hat = temp(1); Ssigtau2hat = 1./temp(2);
rhohat = mean(store_theta(:,8));
rhovar = var(store_theta(:,8));
tau0hat = mean(store_theta(:,9));
tau0var = var(store_theta(:,9));

theta_IS = zeros(M,9);
theta_IS(:,1) = muhat(1) + sqrt(muvar(1))*randn(M,1); 
theta_IS(:,2) = muhat(2) + sqrt(muvar(2))*randn(M,1); 
theta_IS(:,3) = muhat(3) + sqrt(muvar(3))*randn(M,1);
theta_IS(:,4:5) = repmat(phihat',M,1) + (chol(phivar,'lower')*randn(2,M))';
theta_IS(:,6) = 1./gamrnd(nusigc2hat,1./Ssigc2hat,M,1);
theta_IS(:,7) = 1./gamrnd(nusigtau2hat,1./Ssigtau2hat,M,1);
theta_IS(:,8) = tnormrnd(rhohat,rhovar,-.999,.999,M);
theta_IS(:,9) = tau0hat + sqrt(tau0var)*randn(M,1);

store_w = zeros(M,1);

rho_const = 1/(normcdf(1,rhohat,sqrt(rhovar))-normcdf(-1,rhohat,sqrt(rhovar)));
gIS = @(m,ph,sy,st,r,ta0) -3/2*log(2*pi)-.5*sum(log(muvar)) -.5*sum((m-muhat).^2./muvar) ...
    -log(2*pi)-.5*log(det(phivar))-.5*(ph-phihat)'*(phivar\(ph-phihat))...
    + nusigc2hat*log(Ssigc2hat)-gammaln(nusigc2hat)-(nusigc2hat+1)*log(sy)- Ssigc2hat/sy ...
    + nusigtau2hat*log(Ssigtau2hat)-gammaln(nusigtau2hat)-(nusigtau2hat+1)*log(st) - Ssigtau2hat/st...
    -.5*log(2*pi*rhovar) + log(rho_const) -.5*(r-rhohat)^2/rhovar ...    
    -.5*log(2*pi*tau0var) - .5*(ta0-tau0hat)^2/tau0var;
for isim = 1:M    
    theta = theta_IS(isim,:)';
    mu = theta(1:3);
    phi = theta(4:5);
    sigc2 = theta(6);
    sigtau2 = theta(7);
    rho = theta(8);
    tau0 = theta(9);
    
    if sum(phi) < .99 && phi(2) - phi(1) < .99 && phi(2) > -.99
        llike = intlike_UCUR_break2(y,mu,phi,sigc2,sigtau2,rho,tau0,t0,t1);
        store_w(isim) = llike + prior(mu,phi,sigc2,sigtau2,rho,tau0) ...
            - gIS(mu,phi,sigc2,sigtau2,rho,tau0);    
    else
        store_w(isim) = -10^(100);
    end
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end