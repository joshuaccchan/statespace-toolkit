%% guide - the code of guides/uc_states/README.md: the trend of a local level model drawn
% by the precision sampler, given the parameters, on generated data; then the same with a
% persistent transitory component, and with a known variance that changes over time. The
% sections called "the code to adapt" are all a sampler needs. The sections called "the
% checks" form K^{-1} and other dense matrices, which only a small example can afford.
% Saves fig_spy.png, fig_level.png and fig_ar1.png next to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(1, 'twister');
rel = @(a, b) max(abs(a(:) - b(:)))/max(abs(b(:)));

%% 1. the local level model: the code to adapt
T = 200; sig2 = 1; omega2 = .05; tau0 = 2;               % the parameters, known
tau = tau0 + cumsum(sqrt(omega2)*randn(T,1));            % tau_1 ~ N(tau0, omega2)
y = tau + sqrt(sig2)*randn(T,1);

H = ssm.diffmat(T);                                      % first differences
K = H'*H/omega2 + speye(T)/sig2;                         % quadratic terms
c = H'*[tau0; zeros(T-1,1)]/omega2 + y/sig2;             % linear terms: K*tauhat = c
[draws, tauhat] = ssm.simulate_states(K, c, 5000);       % 5,000 independent draws

%% 1. the checks
fprintf('\nthe local level model, T = %d\n', T);
fprintf('the top left corner of K:\n'); disp(full(K(1:4,1:4)));
V = omega2*inv(full(H'*H));                              % the prior covariance of tau
S = V + sig2*eye(T);
mu = tau0 + V*(S\(y - tau0));                            % the covariance form
Vpost = V - V*(S\V);
fprintf('mean and covariance against dense algebra:   %.1e, %.1e (relative)\n', ...
    rel(tauhat, mu), rel(inv(full(K)), Vpost));
fprintf('5,000 draws: means within %.2f posterior sd, variances %.3f to %.3f of the posterior''s\n', ...
    max(abs(mean(draws, 2) - mu)./sqrt(diag(Vpost))), min(var(draws, 0, 2)./diag(Vpost)), ...
    max(var(draws, 0, 2)./diag(Vpost)));
b1 = quantile(draws, [.05 .95], 2);
fprintf('this realization: the 90%% intervals contain the true trend in %d of %d periods\n', ...
    nnz(tau >= b1(:,1) & tau <= b1(:,2)), T);
y1 = y; tau1 = tau; m1 = mean(draws, 2);
Kspy = ssm.diffmat(50)'*ssm.diffmat(50)/omega2 + speye(50)/sig2;   % K for T = 50

%% 2. a persistent transitory component: the code to adapt
rho = .8;                                                % eps_t = rho*eps_{t-1} + u_t
tau = tau0 + cumsum(sqrt(omega2)*randn(T,1));
y = tau + filter(1, [1 -rho], sqrt(sig2)*randn(T,1));    % eps_0 = 0

Hrho = ssm.diffmat(T, rho);                              % I - rho*L
K = H'*H/omega2 + Hrho'*Hrho/sig2;
c = H'*[tau0; zeros(T-1,1)]/omega2 + Hrho'*Hrho*y/sig2;
[draws, tauhat] = ssm.simulate_states(K, c, 5000);

%% 2. the checks
fprintf('\na persistent transitory component, rho = %.1f; K has bandwidth %d\n', rho, ...
    bandwidth(K, 'lower'));
W = sig2*inv(full(Hrho'*Hrho));                          % the covariance of eps
S = V + W;
mu = tau0 + V*(S\(y - tau0));
Vpost = V - V*(S\V);
fprintf('mean and covariance against dense algebra:   %.1e, %.1e (relative)\n', ...
    rel(tauhat, mu), rel(inv(full(K)), Vpost));
b2 = quantile(draws, [.05 .95], 2);
fprintf('this realization: the 90%% intervals contain the true trend in %d of %d periods\n', ...
    nnz(tau >= b2(:,1) & tau <= b2(:,2)), T);
fprintf(['average width of the 90%% intervals: %.2f, against %.2f with a serially ' ...
    'uncorrelated transitory component\n'], mean(b2(:,2) - b2(:,1)), mean(b1(:,2) - b1(:,1)));
y2 = y; tau2 = tau; m2 = mean(draws, 2);

%% 3. a known measurement variance that changes over time: the adaptation
sig2t = .5 + 1.5*(1:T)'/T;                               % from 0.5 to 2
K = H'*H/omega2 + spdiags(1./sig2t, 0, T, T);
c = H'*[tau0; zeros(T-1,1)]/omega2 + y1./sig2t;
[draws, tauhat] = ssm.simulate_states(K, c, 5000);
S = V + diag(sig2t);
fprintf('\na known, time-varying measurement variance: mean against dense algebra %.1e\n', ...
    rel(tauhat, tau0 + V*(S\(y1 - tau0))));

%% the figures
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 16 8]);
subplot(1,2,1); spy(Kspy, 'k', 4); title('K, T = 50', 'FontWeight', 'normal'); xlabel('');
subplot(1,2,2); spy(sparse(abs(inv(full(Kspy))) > 1e-12), 'k', 4);
title('K^{-1}, T = 50', 'FontWeight', 'normal'); xlabel('');
exportgraphics(fig, fullfile(here, 'fig_spy.png'), 'Resolution', 150); close(fig);
figs = {'fig_level.png', 'fig_ar1.png'}; Y = {y1, y2}; TT = {tau1, tau2}; M = {m1, m2}; B = {b1, b2};
for i = 1:2
    fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 7.5]); hold on
    hb = ssm.shaded_band(1:T, B{i}(:,1), B{i}(:,2));
    hy = plot(1:T, Y{i}, '.', 'Color', [.6 .6 .6]);
    ht = plot(1:T, TT{i}, 'k--', 'LineWidth', 1);
    hm = plot(1:T, M{i}, 'k', 'LineWidth', 1.3);
    hold off; box off; xlim([1 T]);
    legend([hy ht hm hb], {'y_t', 'true trend', 'posterior mean', '90% pointwise credible interval'}, ...
        'Location', 'best', 'NumColumns', 2); legend boxoff
    exportgraphics(fig, fullfile(here, figs{i}), 'Resolution', 150); close(fig);
end
