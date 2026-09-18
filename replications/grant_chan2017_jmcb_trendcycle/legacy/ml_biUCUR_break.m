% This script estimates the marginal likelihood of the bivariate UC model 
% with correlated errors and one break
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552

function [ml,mlstd] = ml_biUCUR_break(y,store_theta,t0,prior,M)
M = 20*ceil(M/20);
muhat = mean(store_theta(:,1:4))';
muvar = var(store_theta(:,1:4))';
phihat = mean(store_theta(:,5:8))';
phivar = cov(store_theta(:,5:8));
tau0hat = mean(store_theta(:,9:10))';
tau0var = var(store_theta(:,9:10))';
N = size(store_theta,1);
store_Sig = zeros(N,4,4);
for i=1:N
    Sig = eye(4);
    dSig = store_theta(i,11:14)';
    Sig([3 8 2 4 7 12]) = store_theta(i,15:end);
    Sig([9 14 5 13 10 15]) = store_theta(i,15:end);
    Sig = Sig.*(sqrt(dSig)*sqrt(dSig)');
    store_Sig(i,:,:) = Sig;
end
nuhat = size(y,1)/2;
Sighat = squeeze(mean(store_Sig))*nuhat;

theta_IS = zeros(M,12);
theta_IS(:,1) = muhat(1) + sqrt(muvar(1))*randn(M,1); 
theta_IS(:,2) = muhat(2) + sqrt(muvar(2))*randn(M,1); 
theta_IS(:,3) = muhat(3) + sqrt(muvar(3))*randn(M,1);
theta_IS(:,4) = muhat(4) + sqrt(muvar(4))*randn(M,1);
theta_IS(:,5:8) = repmat(phihat',M,1) + (chol(phivar,'lower')*randn(4,M))';
theta_IS(:,9) = tau0hat(1) + sqrt(tau0var(1))*randn(M,1);
theta_IS(:,10) = tau0hat(2) + sqrt(tau0var(2))*randn(M,1);
store_w = zeros(M,1);
gIS = @(m,ph,ta0,S) -4/2*log(2*pi)-.5*sum(log(muvar)) -.5*sum((m-muhat).^2./muvar) ...
     -4/2*log(2*pi)-.5*log(det(phivar))-.5*(ph-phihat)'*(phivar\(ph-phihat))...
     -2/2*log(2*pi)-.5*sum(log(tau0var))-.5*sum((ta0-tau0hat).^2./tau0var) ...
     +linvwishpdf(S,nuhat,Sighat);
for isim = 1:M    
    theta = theta_IS(isim,:)';
    mu = theta(1:4);
    phi = theta(5:8);    
    tau0 = theta(9:10);    
    Sig = iwishrnd(Sighat,nuhat);    
    
    if sum(phi([1 3]))<.999 && phi(3)-phi(1)<.999 && phi(3)>-.999 ...
            && sum(phi([2 4]))<.999 && phi(4)-phi(2)<.999 && phi(4)>-.999
        llike = intlike_biUCUR_break(y,mu,phi,Sig,tau0,t0);
        store_w(isim) = llike + prior(mu,phi,tau0,Sig) - gIS(mu,phi,tau0,Sig);
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