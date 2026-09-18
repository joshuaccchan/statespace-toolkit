% This script estimates the marginal likelihood of the GARCH-M model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [ml mlstd] = ml_garch_m(y,store_theta,prior,sig20,M)
M = 20*ceil(M/20);
store_gam = log(store_theta(:,3:end));
muhat = mean(store_theta(:,1))';
mupre = 1/var(store_theta(:,1));
mustd = sqrt(var(store_theta(:,1)));
lamhat = mean(store_theta(:,2))';
lampre = 1/var(store_theta(:,2));
lamstd = sqrt(var(store_theta(:,2)));
gamhat = mean(store_gam)';
gamstd = chol(cov(store_gam),'lower');
gampre = cov(store_gam)\speye(3);

theta_IS = zeros(M,5);
theta_IS(:,1) = muhat + mustd*randn(M,1); 
theta_IS(:,2) = lamhat + lamstd*randn(M,1);
theta_IS(:,3:end) = repmat(gamhat',M,1) + (gamstd*randn(3,M))';

store_w = zeros(M,1);
gIS = @(m,l,g) -.5*log(2*pi/mupre) -.5*(m-muhat)^2*mupre ...
    -.5*log(2*pi/lampre) -.5*(l-lamhat)^2*lampre ... 
    -3/2*log(2*pi) - sum(log(diag(gamstd))) -.5*(g-gamhat)'*gampre*(g-gamhat);

for loop = 1:M
    theta = theta_IS(loop,:)';
    mu = theta(1);
    lam = theta(2);
    gam = theta(3:end);
    if sum(exp(gam(2:3)))<1  % check the draw is stationary        
        llike = loglike_garch_m(y-mu,gam,lam,sig20);
        store_w(loop) = llike + prior(mu,lam,gam) - gIS(mu,lam,gam);
    else
        store_w(loop) = -10^100;
    end
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end