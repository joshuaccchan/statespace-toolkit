%% guide - the code of guides/tvp_compare/README.md: four TVP regressions, which differ in the
% coefficients that follow random walks, compared by their marginal likelihoods. The sections
% called "the code to adapt" are all a comparison needs. Saves fig_prior.png next to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));

%% the data of guides/tvp_states, and the data of guides/tvp_gibbs with a constant slope
rng(1, 'twister');
T = 200; k = 2;
X = [ones(T,1) randn(T,1)];         % x_t = (1, x_t)'
beta = [2 1] + cumsum(randn(T,k)*sqrt(diag([.005 .02])));
y1 = sum(X.*beta, 2) + randn(T,1);  % both coefficients vary
rng(2, 'twister');
tau = 2 + cumsum(sqrt(.005)*randn(T,1));
y2 = tau + X(:,2) + randn(T,1);     % the intercept varies, the slope is 1
Y = {y1, y2}; dnames = {'both vary', 'constant slope'};
rng(3, 'twister');

%% 1. the four models and their marginal likelihoods: the code to adapt
nu_sig = 3; S_sig = 2;        % sig2 ~ IG(nu_sig, S_sig)
nu_om = 3; S_om = [.02 .02];  % omega_i^2 ~ IG(nu_om, S_om(i))
V0 = 100;                     % beta_i0 ~ N(0, V0), constant beta_i too
logig = @(x, nu, S) nu*log(S) - gammaln(nu) - (nu + 1)*log(x) - S./x;
vary = {[], 1, 2, [1 2]};     % the coefficients that vary, by model
mnames = {'none', 'intercept', 'slope', 'both'};
nsim = 5000; burnin = 500; R = 10000;
lml = zeros(4, 2); nse = lml; acc = lml; draws = cell(4, 2); LP = cell(4, 2);
for d = 1:2
    for m = 1:4
        M = model_matrices(X, vary{m}, V0);
        % log p(y | theta) + log p(theta) + the Jacobian, at phi = (log sig2, log omega_i^2)
        lp = @(phi) ssm.intlike(Y{d}, M.Z, speye(T)*exp(-phi(1)), M.P(exp(-phi(2:end))), M.b) ...
            + logig(exp(phi(1)), nu_sig, S_sig) ...
            + sum(logig(exp(phi(2:end)), nu_om, S_om(vary{m})')) + sum(phi);
        [draws{m,d}, acc(m,d)] = draw_variances(lp, 1 + numel(vary{m}), nsim, burnin);
        [lml(m,d), nse(m,d)] = is_logml(lp, draws{m,d}, R);
        LP{m,d} = lp;
    end
end

%% 1. the checks
fprintf('\nlog marginal likelihoods by importance sampling (numerical standard error)\n');
fprintf('%-10s %24s %24s\n', 'varies', dnames{:});
for m = 1:4
    fprintf('%-10s %16.2f (%.3f) %16.2f (%.3f)\n', mnames{m}, lml(m,1), nse(m,1), ...
        lml(m,2), nse(m,2));
end
fprintf('acceptance rates of the Metropolis-Hastings samplers: %.2f to %.2f\n', ...
    min(acc(:)), max(acc(:)));
glml = zeros(4, 2); ng = [400 60 60 24];
for d = 1:2
    for m = 1:4
        glml(m,d) = grid_logml(LP{m,d}, draws{m,d}, ng(m), 6);
    end
end
fprintf('the same on a grid, which spans 6 standard deviations of the draws\n');
for m = 1:4
    fprintf('%-10s %24.2f %24.2f\n', mnames{m}, glml(m,1), glml(m,2));
end
fprintf('importance sampling minus grid, at most %.3f, or %.1f numerical standard errors\n', ...
    max(abs(lml(:) - glml(:))), max(abs(lml(:) - glml(:))./nse(:)));
for d = 1:2
    [~, best] = max(lml(:,d));
    fprintf('%s: the best model lets %s vary; log Bayes factors against it: %s\n', dnames{d}, ...
        mnames{best}, mat2str(round(lml(:,d)' - lml(best,d), 1)));
end
rv = exp(draws{4,1}); rc = exp(draws{4,2});
fprintf(['posterior means of omega2_2 in the model where both vary: %.4f on the data whose ' ...
    'slope varies, %.5f on the data whose slope is constant\n'], mean(rv(:,3)), mean(rc(:,3)));
derr = 0;                               % ssm.intlike against the dense density
for d = 1:2
    for m = 1:4
        th = exp(mean(draws{m,d}, 1));  % (sig2, omega_i^2) at the mean of phi
        M = model_matrices(X, vary{m}, V0);
        ll = ssm.intlike(Y{d}, M.Z, speye(T)/th(1), M.P(1./th(2:end)), M.b);
        llD = dense_loglike(Y{d}, X, vary{m}, V0, th(1), th(2:end));
        derr = max(derr, abs(ll - llD)/abs(llD));
    end
end
fprintf('ssm.intlike against the dense normal density of y: at most %.0e relative\n', derr);
th = exp(mean(draws{2,1}, 1)); M = model_matrices(X, 1, V0);  % only the intercept varies
K = M.P(1/th(2)) + M.Z'*M.Z/th(1);
n = size(K, 1); pf = [n 1:n-1];         % the constant slope moved first
fprintf(['nonzero entries of the Cholesky factor of K when only the intercept varies: %d ' ...
    'with the constant slope last, %d with it first\n'], nnz(chol(K, 'lower')), ...
    nnz(chol(K(pf,pf), 'lower')));

%% 2. the prior of the slope's variance: the log Bayes factor of "both" against "intercept"
pmean = [1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1];  % prior means of omega2_2, S/(nu - 1)
lbf = zeros(numel(pmean), 2); lbse = lbf; LP2 = cell(numel(pmean), 2); DR = LP2;
M = model_matrices(X, [1 2], V0);
for d = 1:2
    for p = 1:numel(pmean)
        S2 = 2*pmean(p);
        lp = @(phi) ssm.intlike(Y{d}, M.Z, speye(T)*exp(-phi(1)), M.P(exp(-phi(2:end))), M.b) ...
            + logig(exp(phi(1)), nu_sig, S_sig) + logig(exp(phi(2)), nu_om, S_om(1)) ...
            + logig(exp(phi(3)), nu_om, S2) + sum(phi);
        dr = draw_variances(lp, 3, nsim, burnin);
        [l3, s3] = is_logml(lp, dr, R);
        lbf(p,d) = l3 - lml(2,d); lbse(p,d) = sqrt(s3^2 + nse(2,d)^2);
        LP2{p,d} = lp; DR{p,d} = dr;
    end
end

%% 2. the checks
fprintf('\nlog Bayes factor of "both" against "intercept", by the prior mean of omega2_2\n');
fprintf('%-12s %24s %24s\n', 'prior mean', dnames{:});
for p = 1:numel(pmean)
    fprintf('%-12g %16.2f (%.3f) %16.2f (%.3f)\n', pmean(p), lbf(p,1), lbse(p,1), ...
        lbf(p,2), lbse(p,2));
end
gbf = zeros(size(lbf));
for d = 1:2
    for p = 1:numel(pmean)
        gbf(p,d) = grid_logml(LP2{p,d}, DR{p,d}, 24, 6) - glml(2,d);
    end
end
fprintf('importance sampling minus grid, at most %.3f\n', max(abs(lbf(:) - gbf(:))));

%% the figure: the log Bayes factor against the prior mean of the slope's variance
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 16 9]); hold on
yline(0, 'Color', [.6 .6 .6]);
h1 = plot(pmean, lbf(:,1), 'k-o', 'LineWidth', 1.3, 'MarkerFaceColor', 'k');
h2 = plot(pmean, lbf(:,2), 'k--s', 'LineWidth', 1.3);
hold off; box off; set(gca, 'XScale', 'log'); xlim([5e-5 2e-1]);
xlabel('prior mean of \omega_2^2'); ylabel('log Bayes factor');
legend([h1 h2], {'data whose slope varies', 'data whose slope is constant'}, 'Location', 'best');
legend boxoff
exportgraphics(fig, fullfile(here, 'fig_prior.png'), 'Resolution', 150); close(fig);

%% 3. three regressors and eight models: the adaptation
rng(4, 'twister');
k = 3; X = [ones(T,1) randn(T,2)];
y = X*[2; 1; -1] + randn(T,1);
vary = arrayfun(@(m) find(bitget(m, 1:k)), 0:2^k - 1, 'UniformOutput', false);
derr = 0;
for m = 1:numel(vary)
    om2 = .01*ones(1, numel(vary{m}));
    M = model_matrices(X, vary{m}, V0);
    ll = ssm.intlike(y, M.Z, speye(T), M.P(1./om2), M.b);
    llD = dense_loglike(y, X, vary{m}, V0, 1, om2);
    derr = max(derr, abs(ll - llD)/abs(llD));
end
sub = cellfun(@(v) ['{' strjoin(arrayfun(@num2str, v, 'UniformOutput', false), ', ') '}'], ...
    vary, 'UniformOutput', false);
fprintf('\nthree regressors, %d models, which let these coefficients vary: %s\n', ...
    numel(vary), strjoin(sub, ' '));
fprintf('ssm.intlike against the dense normal density of y: at most %.0e relative\n', derr);

%% local functions
function M = model_matrices(X, V, V0)
% the stacked design Z, prior mean b and prior precision P(iom) of the regression whose
% coefficients V follow random walks from beta_i0 ~ N(0, V0) and whose others are constant
% with the prior N(0, V0); the states are (beta_V0', beta_V1', ..., beta_VT', beta_C')',
% and iom holds 1/omega_i^2 for i in V
[T, k] = size(X); C = setdiff(1:k, V); kv = numel(V); kc = numel(C);
if kv > 0
    G = kron(ssm.diffmat(T+1), speye(kv));  % the first differences of the paths
    n = kv*(T+1);
    A0 = G'*spdiags([ones(kv,1)/V0; zeros(kv*T,1)], 0, n, n)*G;
    Ai = cell(1, kv);
    for i = 1:kv
        sel = [zeros(kv,1); repmat(double((1:kv)' == i), T, 1)];
        Ai{i} = blkdiag(G'*spdiags(sel, 0, n, n)*G, sparse(kc, kc));
    end
    Zv = [sparse(T, kv) ssm.surform(X(:,V))];
else
    A0 = sparse(0, 0); Ai = {}; Zv = sparse(T, 0);
end
A0 = blkdiag(A0, speye(kc)/V0);
M.Z = [Zv sparse(X(:,C))];
M.b = zeros(size(A0, 1), 1);
M.P = @(iom) prior_precision(A0, Ai, iom);
end

function P = prior_precision(A0, Ai, iom)
% A0 + sum_i Ai{i}*iom(i)
P = A0;
for i = 1:numel(Ai)
    P = P + Ai{i}*iom(i);
end
end

function [draws, acc] = draw_variances(lp, d, nsim, burnin)
% independence-chain Metropolis-Hastings on phi, with a t proposal (5 degrees of freedom) at
% the mode of lp and scaled by the inverse of its negative Hessian there, as in
% guides/integrated_likelihood; draws holds one draw of phi per row
opt = optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxFunEvals', 1e4, 'MaxIter', 1e4);
phihat = fminsearch(@(phi) -lp(phi), [0; log(.01)*ones(d-1,1)], opt);
h = 1e-3; E = h*eye(d); Hs = zeros(d);
for i = 1:d  % the Hessian by finite differences
    for j = 1:d
        Hs(i,j) = (lp(phihat + E(:,i) + E(:,j)) - lp(phihat + E(:,i) - E(:,j)) ...
            - lp(phihat - E(:,i) + E(:,j)) + lp(phihat - E(:,i) - E(:,j)))/(4*h^2);
    end
end
C = chol(inv(-Hs), 'lower'); nu = 5;
logq = @(phi) -(nu + d)/2*log(1 + sum((C\(phi - phihat)).^2)/nu);
phi = phihat; lw = lp(phi) - logq(phi);
draws = zeros(nsim, d); acc = 0;
for isim = 1:burnin + nsim
    phic = phihat + C*randn(d,1)/sqrt(gamrnd(nu/2, 2/nu));  % a draw from the proposal
    lwc = lp(phic) - logq(phic);
    if log(rand) < lwc - lw
        phi = phic; lw = lwc; acc = acc + (isim > burnin);
    end
    if isim > burnin, draws(isim - burnin, :) = phi'; end
end
acc = acc/nsim;
end

function [lml, nse] = is_logml(lp, draws, R)
% log p(y) by importance sampling. The importance density is a t with 5 degrees of freedom
% at the mean and covariance of the posterior draws: the cross-entropy fit of a normal
% density (Chan and Eisenstat, 2015), with heavier tails. R is a multiple of 20, and the
% numerical standard error comes from 20 batches.
nu = 5; d = size(draws, 2);
m = mean(draws, 1)'; C = chol(cov(draws), 'lower');
z = randn(d, R)./sqrt(chi2rnd(nu, 1, R)/nu);
phi = m + C*z;
lconst = gammaln((nu + d)/2) - gammaln(nu/2) - d/2*log(nu*pi) - sum(log(diag(C)));
logw = zeros(R, 1);
for r = 1:R  % log p(y, theta) - log g(phi)
    logw(r) = lp(phi(:,r)) - (lconst - (nu + d)/2*log(1 + z(:,r)'*z(:,r)/nu));
end
lml = logmeanexp(logw);
lb = zeros(20, 1); wb = reshape(logw, R/20, 20);
for b = 1:20
    lb(b) = logmeanexp(wb(:,b));
end
nse = std(lb)/sqrt(20);
end

function lml = grid_logml(lp, draws, n, s)
% log p(y) as the sum of exp(lp) times the cell volume over a grid of n points per
% dimension that spans s standard deviations of each column of draws around its mean
d = size(draws, 2); g = cell(1, d); G = cell(1, d);
for i = 1:d
    mi = mean(draws(:,i)); si = std(draws(:,i));
    g{i} = linspace(mi - s*si, mi + s*si, n);
end
[G{1:d}] = ndgrid(g{:});
lpg = zeros(numel(G{1}), 1);
for j = 1:numel(lpg)
    lpg(j) = lp(cellfun(@(x) x(j), G)');
end
lml = max(lpg) + log(sum(exp(lpg - max(lpg)))) + sum(cellfun(@(x) log(x(2) - x(1)), g));
end

function ll = dense_loglike(y, X, V, V0, sig2, om2)
% log p(y | sig2, omega2) from the T x T covariance matrix of y: cov(beta_it, beta_is) is
% V0 + min(t,s)*om2(j) if coefficient i = V(j) varies and V0 if it is constant
T = size(X, 1); [t, s] = ndgrid(1:T);
Sy = sig2*eye(T);
for i = 1:size(X, 2)
    Ci = V0*ones(T);
    j = find(V == i);
    if ~isempty(j), Ci = Ci + min(t, s)*om2(j); end
    Sy = Sy + (X(:,i)*X(:,i)').*Ci;
end
ll = -T/2*log(2*pi) - sum(log(diag(chol(Sy)))) - .5*y'*(Sy\y);
end

function m = logmeanexp(x)
% log(mean(exp(x))) without overflow
c = max(x);
m = c + log(mean(exp(x - c)));
end
