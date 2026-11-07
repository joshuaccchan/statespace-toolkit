%% guide - the code of guides/sv_gibbs/README.md: a Gibbs sampler that estimates the mean,
% the log-volatilities and the three parameters of their AR(1) process in the stochastic
% volatility model of guides/sv_states, on the data of that guide; then the same posterior
% by the sampler of Chan and Grant (2016), which draws the log-volatilities from their
% exact conditional distribution, as a check; then a random-walk log-volatility. The
% sections called "the code to adapt" are all a sampler needs. Saves fig_chain.png next to
% this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(42, 'twister');

%% the data of guides/sv_states
T = 1000; mu_true = .1; muh_true = 0; phi_true = .97; sig2h_true = .04;
h_true = zeros(T,1);
h_true(1) = muh_true + sqrt(sig2h_true/(1 - phi_true^2))*randn;
for t = 2:T
    h_true(t) = muh_true + phi_true*(h_true(t-1) - muh_true) + sqrt(sig2h_true)*randn;
end
y = mu_true + exp(h_true/2).*randn(T,1);
tic

%% 1. the Gibbs sampler: the code to adapt
mu0 = 0; Vmu = 10;                                      % mu ~ N(mu0, Vmu)
muh0 = 1; Vmuh = 10;                                    % mu_h ~ N(muh0, Vmuh)
phi0 = .97; Vphi = .1^2;                                % phi ~ TN(phi0, Vphi) on (-1, 1)
nuh = 5; Sh = .16;                                      % sigma_h^2 ~ IG(nuh, Sh)
offset = 1e-4;
nsim = 20000; burnin = 1000;
muh = log(var(y)); phi = .98; sig2h = .04;              % starting values
h = muh*ones(T,1);
store_theta = zeros(nsim, 4); nacc = 0;
store_h = zeros(T, nsim/10);
for isim = 1:burnin + nsim
    Kmu = 1/Vmu + sum(exp(-h));                         % K_mu = 1/V_mu + sum_t e^{-h_t}
    mu = (mu0/Vmu + exp(-h)'*y)/Kmu + randn/sqrt(Kmu);  % mu ~ N(c_mu/K_mu, 1/K_mu)

    ystar = log((y - mu).^2 + offset);  % s, then h: guides/sv_states
    h = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h);

    hc = h - muh;                                          % h_t - mu_h
    u = [sqrt(1 - phi^2)*hc(1); hc(2:T) - phi*hc(1:T-1)];  % the state errors u_t
    sig2h = 1/gamrnd(nuh + T/2, 1/(Sh + u'*u/2));          % IG(nu_h + T/2, S_h + u'u/2)

    Kphi = 1/Vphi + hc(1:T-1)'*hc(1:T-1)/sig2h;             % K_phi and phihat of the proposal
    phihat = (phi0/Vphi + hc(1:T-1)'*hc(2:T)/sig2h)/Kphi;
    phic = 2;
    while abs(phic) >= 1, phic = phihat + randn/sqrt(Kphi); end
    g = @(x) .5*log(1 - x^2) - .5*(1 - x^2)*hc(1)^2/sig2h;  % log g(phi)
    if g(phic) - g(phi) > log(rand)
        phi = phic;
        nacc = nacc + (isim > burnin);
    end

    Kmuh = 1/Vmuh + (1 - phi^2 + (T-1)*(1 - phi)^2)/sig2h;  % mu_h given h, phi and sigma_h^2
    cmuh = muh0/Vmuh + ((1 - phi^2)*h(1) + (1 - phi)*sum(h(2:T) - phi*h(1:T-1)))/sig2h;
    muh = cmuh/Kmuh + randn/sqrt(Kmuh);

    if isim > burnin
        store_theta(isim - burnin, :) = [mu muh phi sig2h];
        if mod(isim - burnin, 10) == 0, store_h(:, (isim - burnin)/10) = h; end
    end
end

%% 1. the checks
t_gibbs = toc;
names = {'mu', 'mu_h', 'phi', 'sigma_h^2'};
truth = [mu_true muh_true phi_true sig2h_true];
Ls = [50 100 200 400];
IFL = zeros(numel(Ls), 4);
for j = 1:numel(Ls), IFL(j,:) = ssm.inefficiency_factor(store_theta, Ls(j)); end
L = 200;                                                 % Bartlett truncation lag
IF = ssm.inefficiency_factor(store_theta, L);
mcse = ssm.mcse(store_theta, L);
pm = mean(store_theta); psd = std(store_theta); pq = quantile(store_theta, [.05 .95]);
ZL = ssm.geweke(store_theta, [], [], L);
fprintf('\nthe Gibbs sampler, %d draws after a burn-in of %d: %.1f s; phi accepted in %.3f of the draws\n', ...
    nsim, burnin, t_gibbs, nacc/nsim);
fmt = '%-10s true %6.3f, posterior mean %7.4f (sd %.4f), 90%% interval (%7.4f, %7.4f), MCSE %.4f\n';
for i = 1:4
    fprintf(fmt, names{i}, truth(i), pm(i), psd(i), pq(1,i), pq(2,i), mcse(i));
end
fprintf('inefficiency factors by truncation lag L (rows L = %s):\n', mat2str(Ls));
disp(round(IFL, 1));
fprintf('Geweke''s Z with L = %d: %s\n', L, mat2str(round(ZL, 2)));
vol = exp(store_h/2); bv = quantile(vol, [.05 .95], 2); vt = exp(h_true/2);
wm = exp(-h_true)'*y/sum(exp(-h_true));                 % the mean of y weighted by the true e^{-h_t}
fprintf('mu: the weighted mean of y with the true volatilities is %.4f, standard error %.4f\n', ...
    wm, 1/sqrt(sum(exp(-h_true))));
fprintf(['e^{h_t/2}: the 90%% intervals contain the truth in %d of %d periods; average ' ...
    'width %.2f\n'], nnz(vt >= bv(:,1) & vt <= bv(:,2)), T, mean(bv(:,2) - bv(:,1)));
[~, im] = sort(abs(store_theta(:,2)), 'descend');      % the draws of mu_h farthest from 0
fprintf('the three draws of mu_h farthest from 0: %s, with phi %s and sigma_h^2 %s\n', ...
    mat2str(round(store_theta(im(1:3),2)', 2)), mat2str(round(store_theta(im(1:3),3)', 4)), ...
    mat2str(round(store_theta(im(1:3),4)', 4)));
fprintf('phi above 0.995 in %d of %d draws\n', nnz(store_theta(:,3) > .995), nsim);

%% 2. the check: the sampler of Chan and Grant (2016), with the exact conditional of h
muh = log(var(y)); phi = .98; sig2h = .04; mu = mean(y);
P = ssm.diffmat(T, phi)'*spdiags([1 - phi^2; ones(T-1,1)]/sig2h, 0, T, T)*ssm.diffmat(T, phi);
e2 = (y - mu).^2;
grad0 = @(x) deal(-P*(x - muh) - .5 + .5*e2.*exp(-x), P + spdiags(.5*e2.*exp(-x), 0, T, T));
h = ssm.mode_newton(muh*ones(T,1), grad0);               % start at the mode
store_x = zeros(nsim, 4); nacc_h = 0;
tic
for isim = 1:burnin + nsim
    Kmu = 1/Vmu + sum(exp(-h));
    mu = (mu0/Vmu + exp(-h)'*y)/Kmu + randn/sqrt(Kmu);

    Hphi = ssm.diffmat(T, phi);                          % h by accept-reject MH: guides/sv_states
    P = Hphi'*spdiags([1 - phi^2; ones(T-1,1)]/sig2h, 0, T, T)*Hphi;
    e2 = (y - mu).^2;
    logf = @(x) -.5*(x - muh)'*P*(x - muh) - .5*sum(x) - .5*exp(-x)'*e2;
    gradK = @(x) deal(-P*(x - muh) - .5 + .5*e2.*exp(-x), P + spdiags(.5*e2.*exp(-x), 0, T, T));
    [h, acc] = ssm.armh(h, logf, gradK, 'ForceAccept', isim == 1);
    nacc_h = nacc_h + acc*(isim > burnin);

    hc = h - muh;
    u = [sqrt(1 - phi^2)*hc(1); hc(2:T) - phi*hc(1:T-1)];
    sig2h = 1/gamrnd(nuh + T/2, 1/(Sh + u'*u/2));

    Kphi = 1/Vphi + hc(1:T-1)'*hc(1:T-1)/sig2h;
    phihat = (phi0/Vphi + hc(1:T-1)'*hc(2:T)/sig2h)/Kphi;
    phic = 2;
    while abs(phic) >= 1, phic = phihat + randn/sqrt(Kphi); end
    g = @(x) .5*log(1 - x^2) - .5*(1 - x^2)*hc(1)^2/sig2h;
    if g(phic) - g(phi) > log(rand), phi = phic; end

    Kmuh = 1/Vmuh + (1 - phi^2 + (T-1)*(1 - phi)^2)/sig2h;
    cmuh = muh0/Vmuh + ((1 - phi^2)*h(1) + (1 - phi)*sum(h(2:T) - phi*h(1:T-1)))/sig2h;
    muh = cmuh/Kmuh + randn/sqrt(Kmuh);

    if isim > burnin, store_x(isim - burnin, :) = [mu muh phi sig2h]; end
end
t_armh = toc;
mcse_x = ssm.mcse(store_x, L);
fprintf(['\nthe sampler of Chan and Grant (2016): %.1f s; h accepted in %.3f of the draws; ' ...
    'inefficiency factors %s\n'], t_armh, nacc_h/nsim, mat2str(round(ssm.inefficiency_factor(store_x, L), 1)));
fprintf('posterior means %s, sd %s\n', mat2str(round(mean(store_x), 4)), mat2str(round(std(store_x), 4)));
fprintf('posterior means, mixture minus exact, in Monte Carlo standard errors of the difference: %s\n', ...
    mat2str(round((pm - mean(store_x))./sqrt(mcse.^2 + mcse_x.^2), 1)));
fprintf('posterior sd, mixture over exact: %s\n', mat2str(round(psd./std(store_x), 3)));

%% 3. a random-walk log-volatility: the adaptation
rng(2, 'twister');
h0_true = 0; sig2h_rw = .01;
h_rw = h0_true + cumsum(sqrt(sig2h_rw)*randn(T,1));  % h_t = h_{t-1} + u_t
y_rw = mu_true + exp(h_rw/2).*randn(T,1);

a0 = 0; b0 = 10;          % h_0 ~ N(a0, b0)
nuh_rw = 5; Sh_rw = .04;  % sigma_h^2 ~ IG(nuh_rw, Sh_rw)
H = ssm.diffmat(T);
h0 = log(var(y_rw)); sig2h = .01; h = h0*ones(T,1);
store_rw = zeros(nsim, 4);
for isim = 1:burnin + nsim
    Kmu = 1/Vmu + sum(exp(-h));
    mu = (mu0/Vmu + exp(-h)'*y_rw)/Kmu + randn/sqrt(Kmu);

    h = ssm.ksc_rw_h0(log((y_rw - mu).^2 + offset), h, sig2h, h0);  % s, then h, given h_0

    u = H*h - [h0; zeros(T-1,1)];  % h_t - h_{t-1}, with h_0
    sig2h = 1/gamrnd(nuh_rw + T/2, 1/(Sh_rw + u'*u/2));

    Kh0 = 1/b0 + 1/sig2h;  % h_0 given h_1 and sigma_h^2
    h0 = (a0/b0 + h(1)/sig2h)/Kh0 + randn/sqrt(Kh0);

    if isim > burnin, store_rw(isim - burnin, :) = [mu sig2h h0 h(1)]; end
end
qrw = quantile(store_rw(:,2), [.05 .95]);
fprintf('\na random-walk log-volatility, on data generated from it, %d draws after %d\n', nsim, burnin);
fprintf('sigma_h^2 true %.3f, posterior mean %.4f, 90%% interval (%.4f, %.4f)\n', ...
    sig2h_rw, mean(store_rw(:,2)), qrw(1), qrw(2));
% the same posterior with h drawn from its exact conditional distribution by ssm.armh
HH = H'*H; nsim_x = 10000;
h0 = log(var(y_rw)); sig2h = .01; e2 = (y_rw - mean(y_rw)).^2;
g0 = @(x) deal(-HH*(x - h0)/sig2h - .5 + .5*e2.*exp(-x), HH/sig2h + spdiags(.5*e2.*exp(-x), 0, T, T));
h = ssm.mode_newton(h0*ones(T,1), g0);  % start at the mode
store_rwx = zeros(nsim_x, 4); nacc_h = 0;
for isim = 1:burnin + nsim_x
    Kmu = 1/Vmu + sum(exp(-h));
    mu = (mu0/Vmu + exp(-h)'*y_rw)/Kmu + randn/sqrt(Kmu);
    e2 = (y_rw - mu).^2;
    logf = @(x) -.5*(x - h0)'*HH*(x - h0)/sig2h - .5*sum(x) - .5*exp(-x)'*e2;
    gradK = @(x) deal(-HH*(x - h0)/sig2h - .5 + .5*e2.*exp(-x), HH/sig2h + spdiags(.5*e2.*exp(-x), 0, T, T));
    [h, acc] = ssm.armh(h, logf, gradK, 'ForceAccept', isim == 1);
    nacc_h = nacc_h + acc*(isim > burnin);
    u = H*h - [h0; zeros(T-1,1)];
    sig2h = 1/gamrnd(nuh_rw + T/2, 1/(Sh_rw + u'*u/2));
    Kh0 = 1/b0 + 1/sig2h;
    h0 = (a0/b0 + h(1)/sig2h)/Kh0 + randn/sqrt(Kh0);
    if isim > burnin, store_rwx(isim - burnin, :) = [mu sig2h h0 h(1)]; end
end
fprintf('the same with h from its exact conditional, %d draws: h accepted in %.3f\n', ...
    nsim_x, nacc_h/nsim_x);
fprintf('posterior means, mixture minus exact, in Monte Carlo standard errors of the difference: %s\n', ...
    mat2str(round((mean(store_rw) - mean(store_rwx))./sqrt(ssm.mcse(store_rw, L).^2 + ssm.mcse(store_rwx, L).^2), 1)));

%% the figure: the draws of the four parameters, their histograms and the check
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 20 11]);
tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
lab = {'\mu', '\mu_h', '\phi', '\sigma_h^2'};
for i = 1:4
    nexttile(i);
    plot(store_theta(:,i), 'Color', [.35 .35 .35], 'LineWidth', .3); box off
    xlim([0 nsim]); title(lab{i}, 'FontWeight', 'normal');
end
for i = 1:4
    nexttile(4 + i); hold on
    xl = quantile(store_theta(:,i), [.0005 .9995]);   % the central 99.9 percent of the draws
    hh = histogram(store_theta(:,i), 60, 'BinLimits', xl, 'Normalization', 'pdf', ...
        'FaceColor', [.75 .75 .75], 'EdgeColor', 'none');
    [fk, xk] = ksdensity(store_x(:,i));
    hk = plot(xk, fk, 'k', 'LineWidth', 1.2);
    hx = xline(truth(i), 'k--', 'LineWidth', 1);
    hold off; box off; xlim(xl);
end
lg = legend([hh hk hx], {'Gibbs sampler with the mixture', ...
    'sampler with the exact conditional of h', 'true value'}, 'Orientation', 'horizontal', 'Box', 'off');
lg.Layout.Tile = 'south';
exportgraphics(fig, fullfile(here, 'fig_chain.png'), 'Resolution', 150); close(fig);
