%% guide - the code of guides/sv_states/README.md: the log-volatilities of a stochastic
% volatility model with a stationary AR(1) log-volatility, drawn given the parameters on
% generated data, first by the auxiliary mixture sampler and then from their exact
% conditional distribution by accept-reject Metropolis-Hastings; then the mixture step for
% a random-walk log-volatility. The sections called "the code to adapt" are all a sampler
% needs. Saves fig_mixture.png and fig_vol.png next to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(42, 'twister');

%% the data, in percent
T = 1000; mu = .1; muh = 0; phi = .97; sig2h = .04;     % the parameters, known
h_true = zeros(T,1);
h_true(1) = muh + sqrt(sig2h/(1 - phi^2))*randn;        % h_1 ~ N(mu_h, sigma_h^2/(1 - phi^2))
for t = 2:T
    h_true(t) = muh + phi*(h_true(t-1) - muh) + sqrt(sig2h)*randn;
end
y = mu + exp(h_true/2).*randn(T,1);

%% 1. the transformed data: the code to adapt
offset = 1e-4;
ystar = log((y - mu).^2 + offset);  % y*_t = log((y_t - mu)^2 + c)

%% 1. the checks
fprintf('\nthe data: T = %d, sample sd %.2f, sd of the true e^{h_t/2} from %.2f to %.2f\n', ...
    T, std(y), min(exp(h_true/2)), max(exp(h_true/2)));
h = muh*ones(T,1);                                     % the current path, for one draw
state = rng;                                            % for the check against ssm.ksc_ar1_mean

%% 2. one draw of the indicators and of h: the code to adapt
pj = [.0073 .10556 .00002 .04395 .34001 .24566 .2575];                        % p_i
mj = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  % mu_i - 1.2704
s2j = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];                 % sigma_i^2

Hphi = ssm.diffmat(T, phi);                             % H_phi = I - phi*L
iS = spdiags([1 - phi^2; ones(T-1,1)]/sig2h, 0, T, T);  % S^{-1}
P = Hphi'*iS*Hphi;                                      % P = H_phi'*S^{-1}*H_phi

q = pj.*normpdf(ystar, h + mj, sqrt(s2j));           % p_i*N(y*_t; h_t + mu_i - 1.2704, sigma_i^2)
s = 1 + sum(rand(T,1) > cumsum(q./sum(q,2), 2), 2);  % s_t from its seven probabilities
K = P + spdiags(1./s2j(s)', 0, T, T);                % K = P + Sigma_s^{-1}
c = P*(muh*ones(T,1)) + (ystar - mj(s)')./s2j(s)';   % c = P*mu_h*1 + Sigma_s^{-1}*(y* - d_s)
h = ssm.simulate_states(K, c);

%% 2. the checks
rng(state);
[h_ksc, s_ksc] = ssm.ksc_ar1_mean(ystar, muh*ones(T,1), muh, phi, sig2h);
fprintf('\nthe top left corner of K at this draw:\n'); disp(full(K(1:4,1:4)));
fprintf('indicators of the first four periods: %s\n', mat2str(s(1:4)'));
fprintf('the steps above against ssm.ksc_ar1_mean, same random numbers: indicators equal: %d, ', ...
    isequal(s, s_ksc));
fprintf('largest difference in h %.1e\n', max(abs(h - h_ksc)));
x = linspace(-30, 8, 76001)'; dx = x(2) - x(1);
f_exact = exp((x - exp(x))/2)/sqrt(2*pi);               % the density of log chi^2_1
f_mix = normpdf(x, mj, sqrt(s2j))*pj';
fprintf(['the mixture against log chi^2_1: means %.4f and %.4f, variances %.4f and %.4f, ' ...
    'L1 distance %.4f\n'], pj*mj', psi(.5) + log(2), pj*(s2j + mj.^2)' - (pj*mj')^2, pi^2/2, ...
    sum(abs(f_mix - f_exact))*dx);
lr = abs(log(f_mix./f_exact));                          % the log ratio of the two densities
fprintf('log of the mixture density over the exact one: largest absolute value %.2f on (-8, 2), %.2f on (-15, -8)\n', ...
    max(lr(x > -8 & x < 2)), max(lr(x > -15 & x <= -8)));
tic

%% 3. the chain: the code to adapt
nsim = 5000; burnin = 500;
h = muh*ones(T,1);                                    % the starting path
store_h = zeros(T, nsim);
for isim = 1:burnin + nsim
    h = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h);  % s given h, then h given s
    if isim > burnin
        store_h(:, isim - burnin) = h;
    end
end

%% 3. the checks
t_mix = toc;
L = 100;                                                % Bartlett truncation lag
IFm = ssm.inefficiency_factor(store_h', L);
vol = exp(store_h/2);                                   % the standard deviation e^{h_t/2}
bm = quantile(vol, [.05 .95], 2); vm = mean(vol, 2);
vt = exp(h_true/2);
fprintf('\nthe mixture sampler, %d draws after a burn-in of %d: %.2f ms per draw\n', nsim, ...
    burnin, 1000*t_mix/(nsim + burnin));
fprintf('inefficiency factors of the h_t: median %.1f, largest %.1f\n', median(IFm), max(IFm));
fprintf(['this realization: the 90%% intervals of e^{h_t/2} contain the truth in %d of %d ' ...
    'periods; average width %.2f\n'], nnz(vt >= bm(:,1) & vt <= bm(:,2)), T, mean(bm(:,2) - bm(:,1)));
tic

%% 4. the exact conditional by accept-reject Metropolis-Hastings: the code to adapt
e2 = (y - mu).^2;                                                     % no offset here
logf = @(x) -.5*(x - muh)'*P*(x - muh) - .5*sum(x) - .5*exp(-x)'*e2;  % log p(h | y) + const
gradK = @(x) deal(-P*(x - muh) - .5 + .5*e2.*exp(-x), ...             % the gradient of logf
    P + spdiags(.5*e2.*exp(-x), 0, T, T));                            % and its negative Hessian
h = ssm.mode_newton(muh*ones(T,1), gradK);                            % start at the mode
store_hx = zeros(T, nsim); nacc = 0;
for isim = 1:burnin + nsim
    [h, acc] = ssm.armh(h, logf, gradK, 'ForceAccept', isim == 1);
    if isim > burnin
        store_hx(:, isim - burnin) = h;
        nacc = nacc + acc;
    end
end

%% 4. the checks
t_armh = toc;
IFx = ssm.inefficiency_factor(store_hx', L);
volx = exp(store_hx/2);
psd = std(volx, 0, 2);
fprintf('\nthe exact conditional by ssm.armh: acceptance rate %.3f, %.2f ms per draw\n', ...
    nacc/nsim, 1000*t_armh/(nsim + burnin));
fprintf('inefficiency factors of the h_t: median %.1f, largest %.1f\n', median(IFx), max(IFx));
fprintf(['posterior means of e^{h_t/2}, mixture minus exact: largest difference %.3f, ' ...
    '%.2f posterior sd\n'], max(abs(vm - mean(volx, 2))), max(abs(vm - mean(volx, 2))./psd));

%% 5. a random-walk log-volatility: the adaptation
h0 = 0;                                         % the known initial value
h = muh*ones(T,1);
state = rng;
q = pj.*normpdf(ystar, h + mj, sqrt(s2j));
s = 1 + sum(rand(T,1) > cumsum(q./sum(q,2), 2), 2);
H = ssm.diffmat(T);                             % first differences
K = H'*H/sig2h + spdiags(1./s2j(s)', 0, T, T);  % K = H'*H/sigma_h^2 + Sigma_s^{-1}
c = H'*[h0; zeros(T-1,1)]/sig2h ...             % c = H'*htilde_0/sigma_h^2
    + (ystar - mj(s)')./s2j(s)';                %   + Sigma_s^{-1}*(y* - d_s)
h = ssm.simulate_states(K, c);
rng(state);
h_rw = ssm.ksc_rw_h0(ystar, muh*ones(T,1), sig2h, h0);
fprintf('\na random walk: the answer against ssm.ksc_rw_h0, same random numbers: %.1e\n', ...
    max(abs(h - h_rw)));

%% the figures
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 7.5]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on
plot(x, f_exact, 'k', 'LineWidth', 1.3); plot(x, f_mix, '--', 'Color', [.45 .45 .45], 'LineWidth', 1.3);
hold off; box off; xlim([-12 4]); title('density', 'FontWeight', 'normal');
nexttile; hold on
plot(x, log(f_exact), 'k', 'LineWidth', 1.3); plot(x, log(f_mix), '--', 'Color', [.45 .45 .45], 'LineWidth', 1.3);
hold off; box off; xlim([-12 4]); ylim([-12 0]); title('log density', 'FontWeight', 'normal');
lg = legend({'log \chi^2_1', 'seven-component mixture'}, 'Orientation', 'horizontal', 'Box', 'off');
lg.Layout.Tile = 'south';
exportgraphics(fig, fullfile(here, 'fig_mixture.png'), 'Resolution', 150); close(fig);

fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 12]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(1:T, y, 'Color', [.45 .45 .45], 'LineWidth', .5); box off; xlim([1 T]);
title('y_t', 'FontWeight', 'normal');
nexttile; hold on
hb = ssm.shaded_band(1:T, bm(:,1), bm(:,2));
ht = plot(1:T, vt, 'k--', 'LineWidth', 1);
hm = plot(1:T, vm, 'k', 'LineWidth', 1.2);
hold off; box off; xlim([1 T]); title('e^{h_t/2}', 'FontWeight', 'normal');
lg = legend([ht hm hb], {'true value', 'posterior mean', '90% pointwise credible interval'}, ...
    'Orientation', 'horizontal', 'Box', 'off');
lg.Layout.Tile = 'south';
exportgraphics(fig, fullfile(here, 'fig_vol.png'), 'Resolution', 150); close(fig);
