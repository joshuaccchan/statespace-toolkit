%% ex01 - The precision sampler on a local level model
%
% y_t = tau_t + eps_t, eps_t ~ N(0, sig2), and tau_t = tau_{t-1} + u_t,
% u_t ~ N(0, omega2), with tau_1 ~ N(tau0, omega2). Given y and the parameters, the
% trend tau = (tau_1, ..., tau_T)' is N(tauhat, K^{-1}) with K = H'*H/omega2 + I/sig2
% tridiagonal, where H = ssm.diffmat(T) is the T x T first-difference matrix.
% ssm.simulate_states draws the whole path at once from the Cholesky factor of K,
% without forming K^{-1}: the precision sampler of Chan and Jeliazkov (2009).
%
% 1. Generated data. tauhat and diag(K^{-1}) equal the Kalman smoother's means and
%    variances to rounding, and 10,000 draws match them within Monte Carlo error.
% 2. US CPI inflation, 1948M1-2019M12, with sig2, omega2 and tau0 estimated: UC.m
%    of the chan-jeliazkov-2009 repository, with the mean and the draw of tau by
%    ssm.simulate_states.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.1.1.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex01: the precision sampler on a local level model ===\n');

%% 1. Generated data: the draws against the Kalman smoother and the truth
rng(1, 'twister');
T = 300; sig2 = 1; omega2 = .05; tau0 = 2;
tau_true = tau0 + cumsum(sqrt(omega2)*randn(T,1));
y = tau_true + sqrt(sig2)*randn(T,1);

H = ssm.diffmat(T); HH = H'*H;
K = HH/omega2 + speye(T)/sig2;                       % tridiagonal
c = tau0/omega2*(HH*ones(T,1)) + y/sig2;             % K*tauhat = c
ndraws = 10000;
[draws, tauhat] = ssm.simulate_states(K, c, ndraws); % T x ndraws, whole paths

[m, v] = kalman_smoother(y, sig2, omega2, tau0);
z = (mean(draws,2) - m)./sqrt(v/ndraws);
vr = var(draws,0,2)./v;
band = quantile(draws, [.05 .95], 2);
fprintf('\n1. Simulated data, T = %d\n', T);
fprintf('   %-42s %.1e\n', 'max |tauhat - smoother mean|', max(abs(tauhat - m)));
fprintf('   %-42s %.1e\n', 'max |diag(K^{-1}) - smoother variance|', ...
    max(abs(diag(inv(full(K))) - v)));               % dense inverse, for this check only
fprintf('   %-42s %.2f\n', sprintf('%d draws: largest |z| of the %d means', ndraws, T), ...
    max(abs(z)));
fprintf('   %-42s %.3f to %.3f\n', sprintf('%d draws: variance / smoother variance', ndraws), ...
    min(vr), max(vr));
fprintf('   %-42s %.2f\n', 'share of the true trend in the 90% band', ...
    mean(tau_true >= band(:,1) & tau_true <= band(:,2)));

%% 2. US CPI inflation: sig2, omega2 and tau0 estimated
rng(42);
nsim = 20000; burnin = 1000;
data = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', 'USCPI.csv'), ...
    'Range', 'B2:B865');
y = data;
T = length(y);
store_tau = zeros(nsim,T);
store_theta = zeros(nsim,3);                         % [sig2, omega2, tau0]
a0 = 5; b0 = 100;                                    % tau0 ~ N(a0, b0)
nu_sig0 = 3; S_sig0 = 1*(nu_sig0-1);                 % sig2 ~ IG(nu_sig0, S_sig0)
nu_omega0 = 3; S_omega0 = .25^2*(nu_omega0-1);       % omega2 ~ IG(nu_omega0, S_omega0)
sig2 = 1; omega2 = .1; tau0 = 5;
H = ssm.diffmat(T); HH = H'*H; HHiota = HH*ones(T,1);
for isim = 1:nsim+burnin
    Ktau = HH/omega2 + speye(T)/sig2;
    tau = ssm.simulate_states(Ktau, tau0/omega2*HHiota + y/sig2);
    sig2 = 1/gamrnd(nu_sig0 + T/2,1/(S_sig0 + (y-tau)'*(y-tau)/2));
    omega2 = 1/gamrnd(nu_omega0 + T/2, ...
        1/(S_omega0 + (tau-tau0)'*HH*(tau-tau0)/2));
    Ktau0 = 1/b0 + 1/omega2;
    tau0_hat = Ktau0\(a0/b0 + tau(1)/omega2);
    tau0 = tau0_hat + sqrt(Ktau0)'\randn;
    if isim > burnin
        isave = isim - burnin;
        store_tau(isave,:) = tau';
        store_theta(isave,:) = [sig2 omega2 tau0];
    end
end
theta_hat = mean(store_theta);
theta_CI = quantile(store_theta, [.025 .975]);
tau_mean = mean(store_tau)';
tau_q = quantile(store_tau, [.05 .95], 1)';
fprintf('\n2. US CPI inflation, T = %d, %d draws after %d burn-in\n', T, nsim, burnin);
fprintf('   posterior means:        sig2 = %.2f,  omega2 = %.2f,  tau0 = %.2f\n', theta_hat);
fprintf('   posterior 95%% CI lower: sig2 = %.2f,  omega2 = %.2f,  tau0 = %.2f\n', theta_CI(1,:));
fprintf('   posterior 95%% CI upper: sig2 = %.2f,  omega2 = %.2f,  tau0 = %.2f\n', theta_CI(2,:));

%% Figures
tid = 1948 + (0:T-1)'/12;                            % monthly, 1948M1-2019M12
figure('Name', 'ex01 precision sampler');
subplot(2,1,1); hold on
hb = ssm.shaded_band((1:numel(tau_true))', band(:,1), band(:,2));
hs = plot(1:numel(tau_true), tau_true, 'k', 'LineWidth', 1.2);
hm = plot(1:numel(tau_true), mean(draws,2), 'r', 'LineWidth', 1.2);
hold off; box off
title('simulated trend: truth, posterior mean, 90% band')
legend([hs hm hb], {'true \tau_t', 'posterior mean', '90% band'}, 'Location', 'best'); legend boxoff
subplot(2,1,2); hold on
hb = ssm.shaded_band(tid, tau_q(:,1), tau_q(:,2));
hy = plot(tid, y, 'Color', [.6 .6 .6]);
ht = plot(tid, tau_mean, 'k', 'LineWidth', 1.8);
hold off; box off; xlim([tid(1) tid(end)])
title('US CPI inflation: trend posterior mean and 90% band')
legend([hy ht hb], {'CPI inflation', 'trend', '90% band'}, 'Location', 'best'); legend boxoff
drawnow

fprintf('\nex01 done.\n');

function [ms, Ps] = kalman_smoother(y, sig2, omega2, tau0)
% Kalman filter and Rauch-Tung-Striebel smoother for the local level model with
% tau_1 ~ N(tau0, omega2): smoothed means and variances of tau_t given y
T = numel(y);
a = zeros(T,1); P = zeros(T,1); ap = zeros(T,1); Pp = zeros(T,1);
for t = 1:T
    if t == 1
        ap(t) = tau0; Pp(t) = omega2;
    else
        ap(t) = a(t-1); Pp(t) = P(t-1) + omega2;
    end
    g = Pp(t)/(Pp(t) + sig2);
    a(t) = ap(t) + g*(y(t) - ap(t));
    P(t) = (1 - g)*Pp(t);
end
ms = a; Ps = P;
for t = T-1:-1:1
    J = P(t)/Pp(t+1);
    ms(t) = a(t) + J*(ms(t+1) - ap(t+1));
    Ps(t) = P(t) + J^2*(Ps(t+1) - Pp(t+1));
end
end
