%% ex02 - Two models of US PCE inflation compared by marginal likelihood
%
% The data are quarterly US PCE inflation, 1960Q1-2024Q4. M1 is the local level model
% of ex01, y_t = tau_t + eps_t with eps_t ~ N(0, sig2). M2 adds an AR(1) transitory
% component, eps_t = rho*eps_{t-1} + u_t with u_t ~ N(0, sig2), eps_0 = 0 and
% rho ~ U(-1, 1). In both, tau_t = tau_{t-1} + eta_t, eta_t ~ N(0, omega2), with
% tau_1 ~ N(tau0, omega2). Given theta = (rho, sig2, omega2, tau0), each is a linear
% Gaussian state space model with banded precision matrices, and ssm.intlike returns the
% integrated likelihood p(y | theta), with the T = 260 states integrated out: the method
% of Chan and Jeliazkov (2009).
%
% The marginal likelihood p(y) is then an integral over theta alone, three parameters
% in M1 and four in M2. It is estimated by importance sampling on phi = (atanh rho,
% log sig2, log omega2, tau0). The importance density is a multivariate t with 5
% degrees of freedom, located at the mean of the posterior draws of phi and scaled by
% their covariance: the cross-entropy fit of a Gaussian (Chan and Eisenstat, 2015),
% with heavier tails to keep the importance weights bounded.
%
% Two checks are printed:
% 1. At each posterior mean, ssm.intlike equals the Kalman filter log likelihood.
% 2. With tau0 moved into the states, ssm.intlike integrates it out as well, leaving
%    two parameters in M1 and three in M2, few enough for quadrature on a grid. The
%    quadrature values of log p(y) agree with the importance sampling estimates to
%    within 0.01.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Sections 5.1.4 and 5.1.6 and Exercise 9.3.
% Chan, J.C.C. and Eisenstat, E. (2015). Marginal Likelihood Estimation with the
% Cross-Entropy Method, Econometric Reviews, 34(3): 256-285.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex02: two models of US PCE inflation compared by marginal likelihood ===\n');

data = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', ...
    'USPCE_OutputGap.csv'), 'Range', 'B2:B261');
y = data;                                                  % 1960Q1-2024Q4
T = length(y);
pri.a0 = 5; pri.b0 = 100;                                  % tau0 ~ N(a0, b0)
pri.nu_sig0 = 3; pri.S_sig0 = 1*(pri.nu_sig0-1);           % sig2 ~ IG(nu_sig0, S_sig0)
pri.nu_omega0 = 3; pri.S_omega0 = .25^2*(pri.nu_omega0-1); % omega2 ~ IG(nu_omega0, S_omega0)
H = ssm.diffmat(T); HH = H'*H;

%% 1. Posterior draws of theta = (rho, sig2, omega2, tau0)
nsim = 20000; burnin = 1000;
rng(42);
[theta1, tau1] = gibbs_uc(y, pri, nsim, burnin, false);    % M1: rho held at 0
rng(42);
[theta2, tau2] = gibbs_uc(y, pri, nsim, burnin, true);
th1 = mean(theta1)'; th2 = mean(theta2)';
fprintf('\nPosterior means, %d draws after %d burn-in\n', nsim, burnin);
fprintf('   M1 local level:          sig2 = %.2f, omega2 = %.2f, tau0 = %.2f\n', th1(2:4));
fprintf('   M2 AR(1) transitory:     rho = %.2f, sig2 = %.2f, omega2 = %.2f, tau0 = %.2f\n', th2);

%% 2. The integrated likelihood against the Kalman filter
il = [loglike(y, HH, th1); loglike(y, HH, th2)];
kf = [kalman_loglik(y, th1); kalman_loglik(y, th2)];
fprintf('\nlog p(y | theta) at the posterior mean\n');
fprintf('   %-4s %14s %14s %12s\n', '', 'ssm.intlike', 'Kalman filter', 'difference');
fprintf('   %-4s %14.4f %14.4f %12.1e\n', 'M1', il(1), kf(1), il(1) - kf(1));
fprintf('   %-4s %14.4f %14.4f %12.1e\n', 'M2', il(2), kf(2), il(2) - kf(2));

%% 3. Log marginal likelihoods: importance sampling, and quadrature as the check
phi1 = [log(theta1(:,2:3)) theta1(:,4)];                   % M1 has no rho
phi2 = [atanh(theta2(:,1)) log(theta2(:,2:3)) theta2(:,4)];
R = 20000;
rng(42);
[lml1, nse1] = is_logml(phi1, @(phi) logpost(y, HH, phi, pri, false), R);
[lml2, nse2] = is_logml(phi2, @(phi) logpost(y, HH, phi, pri, true), R);
npts = 17;
q1 = quad_logml(y, pri, phi1, false, npts);
q2 = quad_logml(y, pri, phi2, true, npts);
fprintf('\nlog p(y)\n');
fprintf('   %-4s %26s %13s\n', '', 'importance sampling (NSE)', 'quadrature');
fprintf('   %-4s %18.2f (%.3f) %13.2f\n', 'M1', lml1, nse1, q1);
fprintf('   %-4s %18.2f (%.3f) %13.2f\n', 'M2', lml2, nse2, q2);
fprintf('\nLog Bayes factor of M2 against M1: %.2f (quadrature %.2f)\n', lml2 - lml1, q2 - q1);

%% Figure
tid = 1960 + (0:T-1)'/4;                                   % quarterly, 1960Q1-2024Q4
figure('Name', 'ex02 two models of US PCE inflation');
hold on
hy = plot(tid, y, 'Color', [.7 .7 .7]);
h1 = plot(tid, tau1, 'k', 'LineWidth', 1.5);
h2 = plot(tid, tau2, 'r', 'LineWidth', 1.5);
hold off; box off; xlim([tid(1) tid(end)])
title('US PCE inflation: posterior mean of the trend under each model')
legend([hy h1 h2], {'PCE inflation', 'M1 local level', 'M2 AR(1) transitory'}, ...
    'Location', 'best'); legend boxoff
drawnow

fprintf('\nex02 done.\n');

function [store_theta, tau_mean] = gibbs_uc(y, pri, nsim, burnin, sample_rho)
% The Gibbs sampler of the book's UC_AR.m; with sample_rho false, rho stays at 0 and
% the sampler is that of M1. tau_mean is the posterior mean of the trend.
T = length(y);
H = ssm.diffmat(T); HH = H'*H; HHiota = HH*ones(T,1);
sig2 = 1; omega2 = .1; tau0 = 5; rho = 0;
H_rho = ssm.diffmat(T, rho); HH_rho = H_rho'*H_rho;
store_theta = zeros(nsim,4);                               % [rho sig2 omega2 tau0]
tau_mean = zeros(T,1);
for isim = 1:nsim+burnin
    % sample tau
    Ktau = HH/omega2 + HH_rho/sig2;
    tau = ssm.simulate_states(Ktau, tau0/omega2*HHiota + HH_rho*y/sig2);
    e = y - tau;

    % sample rho
    if sample_rho
        Krho = sum(e(1:T-1).^2)/sig2;
        rho_hat = e(1:T-1)'*e(2:T)/sum(e(1:T-1).^2);
        rho = ssm.tnormrnd(rho_hat, 1/Krho, -1, 1);
        H_rho = ssm.diffmat(T, rho); HH_rho = H_rho'*H_rho;
    end

    % sample sig2
    u = H_rho*e;
    sig2 = 1/gamrnd(pri.nu_sig0 + T/2, 1/(pri.S_sig0 + u'*u/2));

    % sample omega2
    omega2 = 1/gamrnd(pri.nu_omega0 + T/2, ...
        1/(pri.S_omega0 + (tau-tau0)'*HH*(tau-tau0)/2));

    % sample tau0
    Ktau0 = 1/pri.b0 + 1/omega2;
    tau0_hat = Ktau0\(pri.a0/pri.b0 + tau(1)/omega2);
    tau0 = tau0_hat + sqrt(1/Ktau0)*randn;

    if isim > burnin
        store_theta(isim-burnin,:) = [rho sig2 omega2 tau0];
        tau_mean = tau_mean + tau/nsim;
    end
end
end

function ll = loglike(y, HH, th)
% log p(y | theta) by ssm.intlike, theta = (rho, sig2, omega2, tau0)
T = length(y);
H_rho = ssm.diffmat(T, th(1));
ll = ssm.intlike(y, speye(T), H_rho'*H_rho/th(2), HH/th(3), th(4)*ones(T,1));
end

function th = phi2theta(phi, has_rho)
% theta = (rho, sig2, omega2, tau0) from phi = (atanh rho, log sig2, log omega2,
% tau0); in M1, phi has no rho entry and rho = 0
if has_rho
    th = [tanh(phi(1)); exp(phi(2:3)); phi(4)];
else
    th = [0; exp(phi(1:2)); phi(3)];
end
end

function lk = logpost(y, HH, phi, pri, has_rho)
% log p(y | theta) + log p(theta) + log |d theta / d phi|, the posterior kernel of phi
th = phi2theta(phi, has_rho);
lk = logprior(th, pri, has_rho) + log(th(2)) + log(th(3));
if has_rho
    lk = lk + log(1 - th(1)^2);
end
if lk > -Inf                                               % |rho| = 1 after rounding
    lk = lk + loglike(y, HH, th);
end
end

function lp = logprior(th, pri, has_rho)
% log p(theta); rho ~ U(-1, 1) in M2
lp = -.5*log(2*pi*pri.b0) - .5*(th(4) - pri.a0)^2/pri.b0 ...
    + ligampdf(th(2), pri.nu_sig0, pri.S_sig0) ...
    + ligampdf(th(3), pri.nu_omega0, pri.S_omega0);
if has_rho
    lp = lp + log(.5*(abs(th(1)) < 1));
end
end

function [lml, nse] = is_logml(phi_draws, logkernel, R)
% Log marginal likelihood by importance sampling, with a multivariate t importance
% density (5 degrees of freedom) located at the mean of the posterior draws of phi and
% scaled by their covariance. logkernel(phi) is the log posterior kernel of phi. The
% numerical standard error comes from 20 batches.
nu = 5;
m = mean(phi_draws)'; C = chol(cov(phi_draws),'lower'); k = numel(m);
z = randn(k,R)./sqrt(chi2rnd(nu,1,R)/nu);
phi = m + C*z;
lconst = gammaln((nu+k)/2) - gammaln(nu/2) - k/2*log(nu*pi) - sum(log(diag(C)));
logw = zeros(R,1);
for r = 1:R
    logg = lconst - (nu+k)/2*log(1 + z(:,r)'*z(:,r)/nu);  % t density of phi(:,r)
    logw(r) = logkernel(phi(:,r)) - logg;
end
lml = logmeanexp(logw);
lml_batch = zeros(20,1);
wb = reshape(logw, R/20, 20);
for b = 1:20
    lml_batch(b) = logmeanexp(wb(:,b));
end
nse = std(lml_batch)/sqrt(20);
end

function lml = quad_logml(y, pri, phi_draws, has_rho, npts)
% log p(y) by quadrature. tau0 joins the states (tau0, tau_1, ..., tau_T)', whose
% prior has precision G'*inv(Q)*G with G = ssm.diffmat(T+1) and Q = diag(b0, omega2,
% ..., omega2), and a0 as the mean of every element, so ssm.intlike integrates tau0
% out with the trend. The grid runs over the other entries of phi, npts points per
% dimension within 8 posterior standard deviations of the posterior mean.
T = length(y);
G = ssm.diffmat(T+1); Z = [sparse(T,1) speye(T)]; b = pri.a0*ones(T+1,1);
x = phi_draws(:,1:end-1);                                  % tau0, the last entry, goes
d = size(x,2);                                             % into the states
grids = cell(1,d);
h = 1;
for c = 1:d
    grids{c} = mean(x(:,c)) + std(x(:,c))*linspace(-8, 8, npts);
    h = h*(grids{c}(2) - grids{c}(1));
end
pts = cell(1,d);
[pts{:}] = ndgrid(grids{:});
pts = cell2mat(cellfun(@(g) g(:), pts, 'UniformOutput', false));   % one point per row
f = zeros(size(pts,1),1);
for i = 1:size(pts,1)
    th = phi2theta([pts(i,:)'; 0], has_rho);               % the tau0 slot is unused
    H_rho = ssm.diffmat(T, th(1));
    P = G'*spdiags([1/pri.b0; ones(T,1)/th(3)], 0, T+1, T+1)*G;
    f(i) = ssm.intlike(y, Z, H_rho'*H_rho/th(2), P, b) ...
        + ligampdf(th(2), pri.nu_sig0, pri.S_sig0) ...
        + ligampdf(th(3), pri.nu_omega0, pri.S_omega0) ...
        + log(th(2)) + log(th(3));                         % Jacobian of the log scale
    if has_rho
        f(i) = f(i) + log(.5) + log(1 - th(1)^2);          % U(-1, 1) prior, Jacobian
    end
end
lml = logmeanexp(f) + log(numel(f)*h);
end

function ll = kalman_loglik(y, th)
% Kalman filter log likelihood of M2, state (tau_t, eps_t)' with (tau_1, eps_1)' ~
% N((tau0, 0)', diag(omega2, sig2)); rho = 0 gives M1
rho = th(1); sig2 = th(2); omega2 = th(3); tau0 = th(4);
F = [1 0; 0 rho]; Q = diag([omega2 sig2]); z = [1 1];
a = [tau0; 0]; P = Q; ll = 0;
for t = 1:numel(y)
    f = z*P*z'; v = y(t) - z*a;
    ll = ll - .5*(log(2*pi*f) + v^2/f);
    k = P*z'/f;
    a = F*(a + k*v); P = F*(P - k*(z*P))*F' + Q;
end
end

function logden = ligampdf(x, a, b)
% log density of IG(a, b), p(x) = b^a/Gamma(a)*x^(-(a+1))*exp(-b/x)
logden = a.*log(b) - gammaln(a) - (a+1).*log(x) - b./x;
end

function m = logmeanexp(x)
% log(mean(exp(x))) without overflow
c = max(x);
m = c + log(mean(exp(x - c)));
end
