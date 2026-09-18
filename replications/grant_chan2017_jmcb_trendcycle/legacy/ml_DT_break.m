% This script estimates the marginal likelihood of the deterministic trend 
% model with one break.

% See:
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552

function [ml mlstd] = ml_DT_break(y,store_theta,t0,prior,M)
M = 20*ceil(M/20);
T = length(y);
muhat = mean(store_theta(:,1:2))';
muvar = var(store_theta(:,1:2))';
phihat = mean(store_theta(:,3:4))';
phivar = cov(store_theta(:,3:4));
temp = gamfit(1./store_theta(:,5));
nusigc2hat = temp(1); Ssigc2hat = 1./temp(2);
tau0hat = mean(store_theta(:,6));
tau0var = var(store_theta(:,6));

theta_IS = zeros(M,6);
theta_IS(:,1) = muhat(1) + sqrt(muvar(1))*randn(M,1); 
theta_IS(:,2) = muhat(2) + sqrt(muvar(2))*randn(M,1);
theta_IS(:,3:4) = repmat(phihat',M,1) + (chol(phivar,'lower')*randn(2,M))';
theta_IS(:,5) = 1./gamrnd(nusigc2hat,1./Ssigc2hat,M,1);
theta_IS(:,6) = tau0hat + sqrt(tau0var)*randn(M,1);

H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
store_w = zeros(M,1);
gIS = @(m,ph,sy,ta0) -log(2*pi)-.5*sum(log(muvar)) -.5*sum((m-muhat).^2./muvar) ...
    -log(2*pi)-.5*log(det(phivar))-.5*(ph-phihat)'*(phivar\(ph-phihat))...
    + nusigc2hat*log(Ssigc2hat)-gammaln(nusigc2hat)-(nusigc2hat+1)*log(sy)- Ssigc2hat/sy ...
    -.5*log(2*pi*tau0var) - .5*(ta0-tau0hat)^2/tau0var;
Xdel = [ones(T,1) H\((1:T)'<t0) H\((1:T)'>=t0) ];
for loop = 1:M    
    theta = theta_IS(loop,:)';
    mu = theta(1:2);
    phi = theta(3:4);
    sigc2 = theta(5);    
    tau0 = theta(6);
    del = [tau0; mu];
    if sum(phi) < .99 && phi(2) - phi(1) < .99 && phi(2) > -.99
        e = y - Xdel*del;
        Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
            - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);    
        llike = -T/2*log(2*pi*sigc2) - .5/sigc2*e'*(Hphi'*Hphi)*e; 
        store_w(loop) = llike + prior(mu,phi,sigc2,tau0) ...
            - gIS(mu,phi,sigc2,tau0);
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