function f = ar_model(y, p, nsim, burnin)
% ar_model - an AR(p) with intercept and constant variance, estimated on y with its
% first p values as presample, under the independent normal and inverse-gamma prior
% of the book's linreg_indep_predictive.m: beta ~ N(0, 100 I), sigma^2 ~ IG(4, 1).
% Returns the predictive distributions of y_{T+1} and of the average of
% y_{T+1}, ..., y_{T+4} at each posterior draw, both normal given the draw: the
% means iterate the recursion, and the variance of the average is
% sigma^2 sum_k (w_k/4)^2 with w_k = sum_{j=0}^{4-k} psi_j, the MA weights psi_j.
%   f.m1, f.v1, f.m4, f.v4 : nsim x 1 means and variances of the two targets
%   f.beta, f.sig2         : the posterior draws
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Chapter 2.
T = numel(y) - p;
X = ones(T, p + 1);
for j = 1:p, X(:, j+1) = y(p+1-j:end-j); end
Y = y(p+1:end);
k = p + 1;
iV = eye(k)/100; nu0 = 4; S0 = 1;
beta = X\Y; sig2 = mean((Y - X*beta).^2);
f.beta = zeros(nsim, k); f.sig2 = zeros(nsim, 1);
for loop = 1:nsim + burnin
    C = chol(iV + X'*X/sig2, 'lower');
    beta = C'\(C\(X'*Y/sig2)) + C'\randn(k,1);
    e = Y - X*beta;
    sig2 = 1/gamrnd(nu0 + T/2, 1/(S0 + e'*e/2));
    if loop > burnin
        f.beta(loop - burnin,:) = beta'; f.sig2(loop - burnin) = sig2;
    end
end
% the predictive distributions
lag = repmat(flipud(y(end-p+1:end))', nsim, 1);   % y_T, ..., y_{T-p+1}
yh = zeros(nsim, 4);
for q = 1:4
    yh(:,q) = f.beta(:,1) + sum(f.beta(:,2:end).*lag, 2);
    lag = [yh(:,q) lag(:,1:end-1)];
end
ps = zeros(nsim, 4); ps(:,1) = 1;                     % psi_0, ..., psi_3
for j = 1:3
    for i = 1:min(j, p), ps(:,j+1) = ps(:,j+1) + f.beta(:,i+1).*ps(:,j+1-i); end
end
w = cumsum(ps, 2);                                     % w_k = psi_0 + ... + psi_{4-k}
f.m1 = yh(:,1); f.v1 = f.sig2;
f.m4 = mean(yh, 2); f.v4 = f.sig2.*sum((w/4).^2, 2);
end
