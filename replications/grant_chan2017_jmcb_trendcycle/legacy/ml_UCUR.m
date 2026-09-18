% This script estimates the marginal likelihood of the UC model with 
% correlated errors.
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552

function [ml mlstd] = ml_UCUR(y,store_theta,prior,M)
M = 20*ceil(M/20);
muhat = mean(store_theta(:,1));
muvar = var(store_theta(:,1));
phihat = mean(store_theta(:,2:3))';
phivar = cov(store_theta(:,2:3));
temp = gamfit(1./store_theta(:,4));
nusigc2hat = temp(1); Ssigc2hat = 1./temp(2);
temp = gamfit(1./store_theta(:,5));
nusigtau2hat = temp(1); Ssigtau2hat = 1./temp(2);
rhohat = mean(store_theta(:,6));
rhovar = var(store_theta(:,6));
tau0hat = mean(store_theta(:,7));
tau0var = var(store_theta(:,7));

theta_IS = zeros(M,7);
theta_IS(:,1) = muhat + sqrt(muvar)*randn(M,1); 
theta_IS(:,2:3) = repmat(phihat',M,1) + (chol(phivar,'lower')*randn(2,M))';
theta_IS(:,4) = 1./gamrnd(nusigc2hat,1./Ssigc2hat,M,1);
theta_IS(:,5) = 1./gamrnd(nusigtau2hat,1./Ssigtau2hat,M,1);
theta_IS(:,6) = tnormrnd(rhohat,rhovar,-.999,.999,M);
theta_IS(:,7) = tau0hat + sqrt(tau0var)*randn(M,1);

store_w = zeros(M,1);

rho_const = 1/(normcdf(1,rhohat,sqrt(rhovar))-normcdf(-1,rhohat,sqrt(rhovar)));
gIS = @(m,ph,sy,st,r,t0) -.5*log(2*pi*muvar) -.5*(m-muhat)^2/muvar...
    -log(2*pi)-.5*log(det(phivar))-.5*(ph-phihat)'*(phivar\(ph-phihat))...
    + nusigc2hat*log(Ssigc2hat)-gammaln(nusigc2hat)-(nusigc2hat+1)*log(sy)- Ssigc2hat/sy ...
    + nusigtau2hat*log(Ssigtau2hat)-gammaln(nusigtau2hat)-(nusigtau2hat+1)*log(st) - Ssigtau2hat/st...
    -.5*log(2*pi*rhovar) + log(rho_const) -.5*(r-rhohat)^2/rhovar ...    
    -.5*log(2*pi*tau0var) - .5*(t0-tau0hat)^2/tau0var;
for isim = 1:M    
    theta = theta_IS(isim,:)';
    mu = theta(1);
    phi = theta(2:3);
    sigc2 = theta(4);
    sigtau2 = theta(5);
    rho = theta(6);
    tau0 = theta(7);
    
    if sum(phi) < .99 && phi(2) - phi(1) < .99 && phi(2) > -.99
        llike = intlike_UCUR(y,mu,phi,sigc2,sigtau2,rho,tau0);
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