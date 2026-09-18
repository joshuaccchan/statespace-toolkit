% This script estimates the marginal likelihood of the GARCH-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [ml mlstd] = ml_garch_t(y,store_theta,prior,sig20,M)
M = 20*ceil(M/20);
store_gam = log(store_theta(:,2:4));
muhat = mean(store_theta(:,1))';
mupre = 1/var(store_theta(:,1));
gamhat = mean(store_gam)';
gamstd = chol(cov(store_gam),'lower');
gampre = cov(store_gam)\speye(3);
nuhat = mean(store_theta(:,5))';
nupre = 1/var(store_theta(:,5));

theta_IS = zeros(M,5);
theta_IS(:,1) = muhat + sqrt(1/mupre)*randn(M,1); 
theta_IS(:,2:4) = repmat(gamhat',M,1) + (gamstd*randn(3,M))';
theta_IS(:,5) = nuhat + sqrt(1/nupre)*randn(M,1);
store_w = zeros(M,1);
gIS = @(m,g,n) -.5*log(2*pi/mupre) -.5*(m-muhat)^2*mupre ...
    -3/2*log(2*pi) - sum(log(diag(gamstd))) -.5*(g-gamhat)'*gampre*(g-gamhat) ...
    -.5*log(2*pi/nupre) -.5*(n-nuhat)^2*nupre;

for loop = 1:M
    theta = theta_IS(loop,:)';
    if sum(exp(theta(3:4)))<1  % check the draw is stationary
        mu = theta(1);
        gam = theta(2:4);
        nu = theta(5);
        llike = loglike_garch_t(y-mu,gam,sig20,nu);
        store_w(loop) = llike + prior(mu,gam,nu) - gIS(mu,gam,nu);
    else
        store_w(loop) = -inf;
    end
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
end