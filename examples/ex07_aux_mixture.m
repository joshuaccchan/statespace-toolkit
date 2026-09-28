%% ex07 - The auxiliary mixture sampler for stochastic volatility, on generated data
%
% The standard stochastic volatility model is
%
%   y_t = exp(h_t/2)*e_t,   e_t ~ N(0,1),     h_t = h_{t-1} + u_t,   u_t ~ N(0, sigma2_h),
%
% with priors h_0 ~ N(a0, b0) and sigma2_h ~ IG(nu_h, S_h). It is nonlinear in h_t. The
% auxiliary mixture sampler of Kim, Shephard and Chib (1998) makes it linear in two steps:
% squaring and taking logs gives y*_t = h_t + e*_t, with y*_t = log(y_t^2) and
% e*_t = log(e_t^2) ~ log chi^2_1; then a seven-component Gaussian mixture approximates
% the log chi^2_1 density, so that, given the component indicators s_t, the model is
% linear and Gaussian and h is drawn in one block from its banded precision by the
% precision sampler of Chan and Jeliazkov (2009). The code sets y*_t = log(y_t^2 + c)
% with c = 1e-4 to avoid log 0, so the exact density of y*_t given h_t is that of
% log(exp(h_t)*e_t^2 + c).
%
% Section 1 compares three densities for e*_t: the exact log chi^2_1, a single Gaussian
% with the same mean and variance, and the seven-component mixture.
%
% Section 2 runs the collapsed Gibbs sampler of Del Negro and Primiceri (2015), in the
% four-block form of Section 10.1.1 of the book Bayesian Macroeconometrics (Chan,
% forthcoming), on 1,000 periods generated from the model. The blocks are sigma2_h, h_0,
% s and h, and one call to ssm.ksc_rw_h0 draws s and then h. The section checks that
% the posterior recovers the path of h and the parameters that generated the data.
%
% Section 3 reweights the draws, which come from the posterior of the mixture model, to
% the exact posterior, as Kim, Shephard and Chib (1998) propose. The reweighting barely
% moves the posterior means of the parameters; the posterior mean of h_t moves most near
% returns close to zero, where log(y_t^2) - h_t falls in the left tail of log chi^2_1 and
% the mixture fits worst.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Sections 10.1.1 and 10.1.2, Example 10.1 and
% Table 10.1.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
% Del Negro, M. and Primiceri, G.E. (2015). Time Varying Structural Vector
% Autoregressions and Monetary Policy: A Corrigendum, Review of Economic Studies, 82(4):
% 1342-1345.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
% Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
% 361-393.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex07: the auxiliary mixture sampler for stochastic volatility ===\n');

% Table 10.1 of the book: weights, means before the shift by -1.2704, variances
pj = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mj = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819];
s2j = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
m_eps = -1.2704; v_eps = 4.9348;           % mean and variance of log chi^2_1

%% Section 1: the log chi^2_1 density and its two approximations
x = linspace(-15, 5, 4001)';
dx = x(2) - x(1);
f_exact = exp(log_chi2_1(x));
f_gauss = normpdf(x, m_eps, sqrt(v_eps));
f_mix = exp(log_mixture(x, pj, mj + m_eps, s2j));

mix_mean = pj*(mj' + m_eps);
mix_var = pj*(s2j' + (mj' + m_eps).^2) - mix_mean^2;
fprintf('\n-- Section 1: the density of log(e_t^2)\n');
fprintf('%-26s %9s %9s %12s %12s\n', '', 'mean', 'variance', 'max |error|', 'L1 distance');
fprintf('%-26s %9.4f %9.4f\n', 'log chi^2_1 (exact)', psi(.5) + log(2), pi^2/2);
fprintf('%-26s %9.4f %9.4f %12.4f %12.4f\n', 'single Gaussian', m_eps, v_eps, ...
    max(abs(f_gauss - f_exact)), sum(abs(f_gauss - f_exact))*dx);
fprintf('%-26s %9.4f %9.4f %12.4f %12.4f\n', 'seven-component mixture', mix_mean, mix_var, ...
    max(abs(f_mix - f_exact)), sum(abs(f_mix - f_exact))*dx);

figure('Name', 'ex07 densities');
subplot(1,2,1);
plot(x, f_exact, 'k', x, f_gauss, '--', x, f_mix, ':', 'LineWidth', 1.3);
box off; xlim([-12 4]);
legend('log \chi^2_1', 'single Gaussian', 'seven-component mixture', 'Location', 'northwest');
legend boxoff; title('density');
subplot(1,2,2);
plot(x, log(f_exact), 'k', x, log(f_gauss), '--', x, log(f_mix), ':', 'LineWidth', 1.3);
box off; xlim([-12 4]); ylim([-12 0]); title('log density');

%% Section 2: the collapsed Gibbs sampler on generated data
rng(42);
T = 1000;
h0_true = log(.5^2);                       % a daily standard deviation of 0.5 percent
% the priors of chapter10/SVRW_YEN.m in the book's code repository
a0 = 0; b0 = 100; nu_h = 3; S_h = .2^2*(nu_h - 1);
sig2h_true = S_h/(nu_h - 1);               % the prior mean, 0.04
h_true = h0_true + cumsum(sqrt(sig2h_true)*randn(T,1));
y = exp(h_true/2).*randn(T,1);
c = 1e-4;
ystar = log(y.^2 + c);

nsim = 10000; burnin = 1000;
H = ssm.diffmat(T);
% the start of SVRW_YEN.m
sig2h = .05;
h0 = log(var(y));
h = sv_gaussian_approx(ystar, h0, sig2h, H, m_eps, v_eps);
store_h = zeros(nsim, T);
store_theta = zeros(nsim, 2);              % [h0 sigma2_h]
lw = zeros(nsim, 1);                       % log weights for Section 3

fprintf('\n-- Section 2: %d periods, %d draws after %d burn-in\n', T, nsim, burnin);
start_time = tic;
for loop = 1:nsim + burnin
    % Blocks 1 and 2: sigma2_h and h_0, which depend on the indicators only through h
    u = H*h - [h0; zeros(T-1,1)];
    sig2h = 1/gamrnd(nu_h + T/2, 1/(S_h + u'*u/2));
    Kh0 = 1/b0 + 1/sig2h;
    h0 = (a0/b0 + h(1)/sig2h)/Kh0 + randn/sqrt(Kh0);

    % Blocks 3 and 4: the indicators s, then h given s
    h = ssm.ksc_rw_h0(ystar, h, sig2h, h0);

    if loop > burnin
        i = loop - burnin;
        store_h(i,:) = h';
        store_theta(i,:) = [h0, sig2h];
        lw(i) = sum(log_exact(y.^2, ystar, h) - log_mixture(ystar - h, pj, mj + m_eps, s2j));
    end
end
fprintf('sampling takes %.1f seconds\n', toc(start_time));

fprintf('\n%-28s %9s %9s %9s\n', 'quantity', 'RMSE', 'post sd', 'coverage');
report('log-volatility h_t', store_h, h_true);
fprintf('\n%-28s %9s %20s %9s\n', 'parameter', 'estimate', '90% interval', 'true');
pn = {'h_0', 'sigma2_h'}; pt = [h0_true, sig2h_true];
for j = 1:2
    q = quantile(store_theta(:,j), [.05 .95]);
    fprintf('%-28s %9.3f %9.3f %9.3f %9.3f\n', pn{j}, mean(store_theta(:,j)), q, pt(j));
end

%% Section 3: reweighting the draws to the exact posterior
w = exp(lw - max(lw)); w = w/sum(w);
ess = 1/sum(w.^2);
fprintf('\n-- Section 3: importance weights p_exact(y*|h)/p_mixture(y*|h)\n');
fprintf('effective sample size: %.0f of %d draws (%.1f%%)\n', ess, nsim, 100*ess/nsim);
fprintf('standard deviation of the log weights: %.3f\n', std(lw));
sd_theta = std(store_theta);
fprintf('\n%-28s %11s %11s %22s\n', 'posterior mean', 'mixture', 'reweighted', 'shift in posterior sd');
for j = 1:2
    m_mix = mean(store_theta(:,j)); m_rw = w'*store_theta(:,j);
    fprintf('%-28s %11.4f %11.4f %22.3f\n', pn{j}, m_mix, m_rw, (m_rw - m_mix)/sd_theta(j));
end
shift_h = (w'*store_h - mean(store_h))./std(store_h);
[mx, tm] = max(abs(shift_h));
fprintf('%-28s %46.3f\n', 'h_t, largest over t', mx);
fprintf('the largest shift is at t = %d, where log(y_t^2) - h_t is %.1f: the left tail of\n', ...
    tm, log(y(tm)^2) - mean(store_h(:,tm)));
fprintf('log chi^2_1, where the mixture fits worst\n');

figure('Name', 'ex07 volatility');
tid = (1:T)';
sq = quantile(exp(store_h/2), [.05 .95])';
ssm.shaded_band(tid, sq(:,1), sq(:,2)); hold on
plot(tid, mean(exp(store_h/2))', 'k', tid, exp(h_true/2), 'r--', 'LineWidth', 1.1);
hold off; box off; xlim([1 T]);
legend('90% band', 'posterior mean', 'true', 'Location', 'northwest'); legend boxoff
title('standard deviation exp(h_t/2)');

function h = sv_gaussian_approx(ystar, h0, sig2h, H, m_eps, v_eps)
% the posterior mean of h when e*_t is replaced by N(m_eps, v_eps), as in the book's
% Example 10.1
T = numel(ystar);
P = H'*H/sig2h;
K = P + speye(T)/v_eps;
C = chol(K, 'lower');
h = C'\(C\(P*(h0*ones(T,1)) + (ystar - m_eps)/v_eps));
end

function lf = log_chi2_1(x)
% the log density of log(e^2), e ~ N(0,1)
lf = -.5*log(2*pi) + .5*x - .5*exp(x);
end

function lf = log_mixture(x, pj, mj, s2j)
% the log density of the Gaussian mixture at each element of x
L = log(pj) - .5*log(2*pi*s2j) - .5*(x - mj).^2./s2j;
mx = max(L, [], 2);
lf = mx + log(sum(exp(L - mx), 2));
end

function lf = log_exact(y2, ystar, h)
% the log density of y*_t = log(y_t^2 + c) given h_t under the exact model:
% y_t^2*exp(-h_t) is chi^2_1, and the Jacobian of the map from y*_t to it is
% exp(y*_t - h_t)
z = y2.*exp(-h);
lf = -.5*log(2*pi) - .5*log(z) - .5*z + ystar - h;
end

function report(name, draws, truth)
% RMSE of the posterior mean, the average posterior standard deviation, and the share of
% true values inside the 90 percent bands
mhat = mean(draws, 1)';
band = quantile(draws, [.05 .95], 1)';
fprintf('%-28s %9.3f %9.3f %8.1f%%\n', name, sqrt(mean((mhat - truth).^2)), ...
    mean(std(draws, 0, 1)), 100*mean(truth >= band(:,1) & truth <= band(:,2)));
end
