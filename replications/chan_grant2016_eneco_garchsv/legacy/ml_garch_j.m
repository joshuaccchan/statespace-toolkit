% This script estimates the marginal likelihood of the GARCH-J model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [ml mlstd] = ml_garch_j(y,store_theta,prior,sig20,M)
M = 20*ceil(M/20);
store_gam = log(store_theta(:,5:end));
muhat = mean(store_theta(:,1))';
mupre = 1/var(store_theta(:,1));
mustd = sqrt(var(store_theta(:,1)));
store_delta = [store_theta(:,2) log(store_theta(:,3))];
deltahat = mean(store_delta)';
deltapre = cov(store_delta)\speye(2);
deltastd = chol(cov(store_delta),'lower');
gamhat = mean(store_gam)';
gamstd = chol(cov(store_gam),'lower');
gampre = cov(store_gam)\speye(3);
phat = betafit(store_theta(:,4));

theta_IS = zeros(M,7);
theta_IS(:,1) = muhat + mustd*randn(M,1); 
theta_IS(:,2:3) = repmat(deltahat',M,1) + (deltastd*randn(2,M))';
theta_IS(:,4) = betarnd(phat(1),phat(2),M,1);
theta_IS(:,5:end) = repmat(gamhat',M,1) + (gamstd*randn(3,M))';

store_w = zeros(M,1);
gIS = @(m,d,k,g) -.5*log(2*pi/mupre) -.5*(m-muhat)^2*mupre ...
    - log(2*pi) - sum(log(diag(deltastd))) -.5*(d-deltahat)'*deltapre*(d-deltahat) ...
    + (phat(1)-1)*log(k) + (phat(2)-1)*log(1-k) - betaln(phat(1),phat(2)) ...
    -3/2*log(2*pi) - sum(log(diag(gamstd))) -.5*(g-gamhat)'*gampre*(g-gamhat);

for loop = 1:M
    theta = theta_IS(loop,:)';
    if sum(exp(theta(6:7)))<1  % check the draw is stationary
        mu = theta(1);
        delta = theta(2:3);
        kappa = theta(4);
        gam = theta(5:end);
        llike = loglike_garch_j(y-mu,delta,kappa,gam,sig20); 
        store_w(loop) = llike + prior(mu,delta,kappa,gam) ...
            - gIS(mu,delta,kappa,gam);
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