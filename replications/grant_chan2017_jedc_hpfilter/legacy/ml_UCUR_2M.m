% This script estimates the marginal likelihood of the UCUR-2M model
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

function [ml mlstd] = ml_UCUR_2M(y,store_theta,prior,M)
M = 20*ceil(M/20);
phihat = mean(store_theta(:,1:2))';
phivar = cov(store_theta(:,1:2));
temp = gamfit(1./store_theta(:,3));
nusigy2hat = temp(1); Ssigy2hat = 1./temp(2);
temp = gamfit(1./store_theta(:,4));
nusigtau2hat = temp(1); Ssigtau2hat = 1./temp(2);
rhohat = mean(store_theta(:,5));
rhovar = var(store_theta(:,5));
tau0hat = mean(store_theta(:,6:7))';
tau0var = cov(store_theta(:,6:7));

theta_IS = zeros(M,7);
theta_IS(:,1:2) = repmat(phihat',M,1) + (chol(phivar,'lower')*randn(2,M))';
theta_IS(:,3) = 1./gamrnd(nusigy2hat,1./Ssigy2hat,M,1);
theta_IS(:,4) = 1./gamrnd(nusigtau2hat,1./Ssigtau2hat,M,1);
theta_IS(:,5) = tnormrnd(rhohat,rhovar,-.999,.999,M);
theta_IS(:,6:7) = repmat(tau0hat',M,1) + (chol(tau0var,'lower')*randn(2,M))';

store_w = zeros(M,1);

rho_const = 1/(normcdf(1,rhohat,sqrt(rhovar))-normcdf(-1,rhohat,sqrt(rhovar)));
gIS = @(ph,sy,st,r,t0) -log(2*pi)-.5*log(det(phivar))-.5*(ph-phihat)'*(phivar\(ph-phihat))...
    + nusigy2hat*log(Ssigy2hat)-gammaln(nusigy2hat)-(nusigy2hat+1)*log(sy)- Ssigy2hat/sy ...
    + nusigtau2hat*log(Ssigtau2hat)-gammaln(nusigtau2hat)-(nusigtau2hat+1)*log(st) - Ssigtau2hat/st...
    -.5*log(2*pi*rhovar) + log(rho_const) -.5*(r-rhohat)^2/rhovar ...  
    -log(2*pi)-.5*log(det(tau0var))-.5*(t0-tau0hat)'*(tau0var\(t0-tau0hat)); 

for loop = 1:M    
    theta = theta_IS(loop,:)';
    phi = theta(1:2);
    sigy2 = theta(3);
    sigtau2 = theta(4);
    rho = theta(5);
    tau0 = theta(6:7);
    
    if sum(phi) < .99 && phi(2) - phi(1) < .99 && phi(2) > -.99
        llike = intlike_UCUR_2M(y,phi,sigy2,sigtau2,rho,tau0);
        store_w(loop) = llike + prior(phi,sigy2,sigtau2,rho,tau0) ...
            - gIS(phi,sigy2,sigtau2,rho,tau0);    
    else
        store_w(loop) = -10^(100);
    end
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end