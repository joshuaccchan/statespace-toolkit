%% guide - the code of guides/sv_t/README.md: a Gibbs sampler for the stochastic volatility
% model of guides/sv_gibbs with Student-t errors, on data generated with the log-volatilities
% and errors of guides/sv_states and t errors with 5 degrees of freedom; the model with
% normal errors on the same data, to show what the t errors change; the same posterior by
% the sampler of Chan and Grant (2016), which draws the log-volatilities from their exact
% conditional distribution, as a check; then errors with an outlier component. The sections
% called "the code to adapt" are all a sampler needs. Saves fig_vol.png and fig_nu.png next
% to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(42, 'twister');

%% the data: the h_t and eps_t of guides/sv_states, with t errors
T = 1000; mu_true = .1; muh_true = 0; phi_true = .97; sig2h_true = .04; nu_true = 5;
h_true = zeros(T,1);
h_true(1) = muh_true + sqrt(sig2h_true/(1 - phi_true^2))*randn;
for t = 2:T
    h_true(t) = muh_true + phi_true*(h_true(t-1) - muh_true) + sqrt(sig2h_true)*randn;
end
epsn = randn(T,1);                                        % the eps_t of guides/sv_states
lam_true = 1./gamrnd(nu_true/2, 2/nu_true, T, 1);        % lambda_t ~ IG(nu/2, nu/2)
y = mu_true + sqrt(lam_true).*exp(h_true/2).*epsn;
tic

%% 1. the Gibbs sampler for SV-t: the code to adapt
mu0 = 0; Vmu = 10;                                  % mu ~ N(mu0, Vmu)
muh0 = 1; Vmuh = 10;                                % mu_h ~ N(muh0, Vmuh)
phi0 = .97; Vphi = .1^2;                            % phi ~ TN(phi0, Vphi) on (-1, 1)
nuh = 5; Sh = .16;                                  % sigma_h^2 ~ IG(nuh, Sh)
nuub = 100;                                         % nu ~ U(2, nuub)
offset = 1e-4;
nsim = 20000; burnin = 1000;
muh = log(var(y)); phi = .98; sig2h = .04; nu = 5;  % starting values
h = muh*ones(T,1); lam = ones(T,1);
store_theta = zeros(nsim, 5); nacc = 0;
store_h = zeros(T, nsim/10); store_lam = zeros(T,1);
for isim = 1:burnin + nsim
    w = 1./(lam.*exp(h));                           % 1/(lambda_t e^{h_t})
    Kmu = 1/Vmu + sum(w);                           % K_mu = 1/V_mu + sum_t 1/(lambda_t e^{h_t})
    mu = (mu0/Vmu + w'*y)/Kmu + randn/sqrt(Kmu);

    ystar = log((y - mu).^2./lam + offset);  % y*_t = log((y_t - mu)^2/lambda_t + c)
    h = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h);

    e2 = (y - mu).^2.*exp(-h);  % lambda_t ~ IG((nu + 1)/2, (nu + e2_t)/2)
    lam = 1./gamrnd((nu + 1)/2, 2./(nu + e2));

    s1 = sum(log(lam)); s2 = sum(1./lam);  % nu: log p(lambda | nu), its mode and curvature
    fnu = @(x) T*(x/2*log(x/2) - gammaln(x/2)) - (x/2 + 1)*s1 - x/2*s2;
    gradK = @(x) deal(T/2*(log(x/2) + 1 - psi(x/2)) - (s1 + s2)/2, T/4*psi(1, x/2) - T/(2*x));
    [nuhat, Knu] = ssm.mode_newton(nu, gradK);
    nuc = nuhat + randn/sqrt(Knu);         % the proposal N(nuhat, 1/K_nu)
    if nuc > 2 && nuc < nuub && fnu(nuc) - fnu(nu) ...
            + .5*Knu*((nuc - nuhat)^2 - (nu - nuhat)^2) > log(rand)
        nu = nuc;
        nacc = nacc + (isim > burnin);
    end

    hc = h - muh;  % sigma_h^2, phi and mu_h: guides/sv_gibbs
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

    if isim > burnin
        store_theta(isim - burnin, :) = [mu muh phi sig2h nu];
        store_lam = store_lam + lam/nsim;
        if mod(isim - burnin, 10) == 0, store_h(:, (isim - burnin)/10) = h; end
    end
end

%% 1. the checks
t_svt = toc;
L = 400;                                                 % Bartlett truncation lag
names = {'mu', 'mu_h', 'phi', 'sigma_h^2', 'nu'};
truth = [mu_true muh_true phi_true sig2h_true nu_true];
IF = ssm.inefficiency_factor(store_theta, L);
mcse = ssm.mcse(store_theta, L);
pm = mean(store_theta); psd = std(store_theta); pq = quantile(store_theta, [.05 .95]);
ZL = ssm.geweke(store_theta, [], [], L);
fprintf('\nSV-t, %d draws after a burn-in of %d: %.1f s; nu accepted in %.3f of the draws\n', ...
    nsim, burnin, t_svt, nacc/nsim);
fmt = '%-10s true %6.3f, posterior mean %7.4f (sd %.4f), 90%% interval (%7.4f, %7.4f), inefficiency %.1f\n';
for i = 1:5
    fprintf(fmt, names{i}, truth(i), pm(i), psd(i), pq(1,i), pq(2,i), IF(i));
end
fprintf('Geweke''s Z with L = %d: %s\n', L, mat2str(round(ZL, 2)));
[fk, xk] = ksdensity(store_theta(:,5)); [~, ik] = max(fk);  % the posterior mode of nu
s1t = sum(log(lam_true)); s2t = sum(1./lam_true);       % nu given the true lambda_t
nu_ml = fminbnd(@(x) -(T*(x/2*log(x/2) - gammaln(x/2)) - (x/2 + 1)*s1t - x/2*s2t), 2, 100);
fprintf(['nu: posterior median %.2f, mode %.2f; the maximum likelihood estimate from the ' ...
    'true lambda_t is %.2f\n'], median(store_theta(:,5)), xk(ik), nu_ml);
wt = 1./(lam_true.*exp(h_true));                         % the true weights 1/(lambda_t e^{h_t})
fprintf('mu: the mean of y weighted by the true 1/(lambda_t e^{h_t}) is %.4f, standard error %.4f\n', ...
    wt'*y/sum(wt), 1/sqrt(sum(wt)));

%% 2. the model with normal errors on the same data: guides/sv_gibbs
muh = log(var(y)); phi = .98; sig2h = .04; h = muh*ones(T,1);
store_sv = zeros(nsim, 4); store_hn = zeros(T, nsim/10);
for isim = 1:burnin + nsim
    Kmu = 1/Vmu + sum(exp(-h));
    mu = (mu0/Vmu + exp(-h)'*y)/Kmu + randn/sqrt(Kmu);
    h = ssm.ksc_ar1_mean(log((y - mu).^2 + offset), h, muh, phi, sig2h);
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
    if isim > burnin
        store_sv(isim - burnin, :) = [mu muh phi sig2h];
        if mod(isim - burnin, 10) == 0, store_hn(:, (isim - burnin)/10) = h; end
    end
end
qsv = quantile(store_sv, [.05 .95]);
fprintf('\nSV with normal errors on the same data:\n');
for i = 1:4
    fprintf('%-10s posterior mean %7.4f, 90%% interval (%7.4f, %7.4f)\n', names{i}, ...
        mean(store_sv(:,i)), qsv(1,i), qsv(2,i));
end
vt = exp(h_true/2);                                      % the volatility of eps_t's scale
vm_t = mean(exp(store_h/2), 2); vm_n = mean(exp(store_hn/2), 2);
bt = quantile(exp(store_h/2), [.05 .95], 2);
[~, big] = sort(lam_true, 'descend');
fprintf('posterior mean of e^{h_t/2} against the truth: RMSE %.3f under SV-t, %.3f under SV\n', ...
    sqrt(mean((vm_t - vt).^2)), sqrt(mean((vm_n - vt).^2)));
fprintf('SV-t: 90%% intervals contain the true e^{h_t/2} in %d of %d periods\n', ...
    nnz(vt >= bt(:,1) & vt <= bt(:,2)), T);
fprintf('the five largest lambda_t:\n');
fprintf(['  period %4d: lambda_t %6.2f, posterior mean %5.2f; |eps_t| %.2f, |sqrt(lambda_t)*eps_t| %.2f; ' ...
    'e^{h_t/2} true %.2f, SV-t %.2f, SV %.2f\n'], [big(1:5)'; lam_true(big(1:5))'; store_lam(big(1:5))'; ...
    abs(epsn(big(1:5)))'; abs(sqrt(lam_true(big(1:5))).*epsn(big(1:5)))'; vt(big(1:5))'; vm_t(big(1:5))'; ...
    vm_n(big(1:5))']);

%% 3. the check: the sampler of Chan and Grant (2016), with the exact conditional of h
nsim_x = 10000;
muh = log(var(y)); phi = .98; sig2h = .04; nu = 5; mu = mean(y); lam = ones(T,1);
Hphi = ssm.diffmat(T, phi);
P = Hphi'*spdiags([1 - phi^2; ones(T-1,1)]/sig2h, 0, T, T)*Hphi;
e2 = (y - mu).^2./lam;
grad0 = @(x) deal(-P*(x - muh) - .5 + .5*e2.*exp(-x), P + spdiags(.5*e2.*exp(-x), 0, T, T));
h = ssm.mode_newton(muh*ones(T,1), grad0);               % start at the mode
store_x = zeros(nsim_x, 5); nacc_h = 0;
for isim = 1:burnin + nsim_x
    w = 1./(lam.*exp(h));
    Kmu = 1/Vmu + sum(w);
    mu = (mu0/Vmu + w'*y)/Kmu + randn/sqrt(Kmu);

    Hphi = ssm.diffmat(T, phi);
    P = Hphi'*spdiags([1 - phi^2; ones(T-1,1)]/sig2h, 0, T, T)*Hphi;
    e2 = (y - mu).^2./lam;
    logf = @(x) -.5*(x - muh)'*P*(x - muh) - .5*sum(x) - .5*exp(-x)'*e2;
    gradh = @(x) deal(-P*(x - muh) - .5 + .5*e2.*exp(-x), P + spdiags(.5*e2.*exp(-x), 0, T, T));
    [h, acc] = ssm.armh(h, logf, gradh, 'ForceAccept', isim == 1);
    nacc_h = nacc_h + acc*(isim > burnin);

    lam = 1./gamrnd((nu + 1)/2, 2./(nu + (y - mu).^2.*exp(-h)));
    s1 = sum(log(lam)); s2 = sum(1./lam);
    fnu = @(x) T*(x/2*log(x/2) - gammaln(x/2)) - (x/2 + 1)*s1 - x/2*s2;
    gradK = @(x) deal(T/2*(log(x/2) + 1 - psi(x/2)) - (s1 + s2)/2, T/4*psi(1, x/2) - T/(2*x));
    [nuhat, Knu] = ssm.mode_newton(nu, gradK);
    nuc = nuhat + randn/sqrt(Knu);
    if nuc > 2 && nuc < nuub && fnu(nuc) - fnu(nu) ...
            + .5*Knu*((nuc - nuhat)^2 - (nu - nuhat)^2) > log(rand)
        nu = nuc;
    end

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

    if isim > burnin, store_x(isim - burnin, :) = [mu muh phi sig2h nu]; end
end
mcse_x = ssm.mcse(store_x, L);
fprintf('\nthe sampler of Chan and Grant (2016), %d draws: h accepted in %.3f of the draws\n', ...
    nsim_x, nacc_h/nsim_x);
fprintf('posterior means %s\n', mat2str(round(mean(store_x), 4)));
fprintf('posterior means, mixture minus exact, in Monte Carlo standard errors of the difference: %s\n', ...
    mat2str(round((pm - mean(store_x))./sqrt(mcse.^2 + mcse_x.^2), 1)));

%% 4. an outlier component: the adaptation
kap = 10;         % lambda_t = kap with probability p, else 1
ap = 1; bp = 19;  % p ~ B(ap, bp)
muh = log(var(y)); phi = .98; sig2h = .04; p = .05;
h = muh*ones(T,1); lam = ones(T,1);
store_o = zeros(nsim, 5); pout = zeros(T,1);
for isim = 1:burnin + nsim
    w = 1./(lam.*exp(h));
    Kmu = 1/Vmu + sum(w);
    mu = (mu0/Vmu + w'*y)/Kmu + randn/sqrt(Kmu);

    h = ssm.ksc_ar1_mean(log((y - mu).^2./lam + offset), h, muh, phi, sig2h);

    e2 = (y - mu).^2.*exp(-h);              % Pr(lambda_t = kap) given y_t, mu, h_t, p
    a1 = p/sqrt(kap)*exp(-e2/(2*kap)); a0 = (1 - p)*exp(-e2/2);
    lam = 1 + (kap - 1)*(rand(T,1) < a1./(a1 + a0));
    nout = nnz(lam > 1);
    p = betarnd(ap + nout, bp + T - nout);  % p ~ B(ap + n_out, bp + T - n_out)

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

    if isim > burnin
        store_o(isim - burnin, :) = [mu muh phi sig2h p];
        pout = pout + (lam > 1)/nsim;
    end
end
% the outlier probability against the two normal densities it comes from
a1n = p*normpdf(y, mu, sqrt(kap*exp(h))); a0n = (1 - p)*normpdf(y, mu, sqrt(exp(h)));
e2 = (y - mu).^2.*exp(-h);
a1 = p/sqrt(kap)*exp(-e2/(2*kap)); a0 = (1 - p)*exp(-e2/2);
fprintf(['\nan outlier component: the probability against normpdf, at the last draw: largest ' ...
    'difference %.1e\n'], max(abs(a1./(a1 + a0) - a1n./(a1n + a0n))));
qo = quantile(store_o, [.05 .95]);
fprintf('p: posterior mean %.4f, 90%% interval (%.4f, %.4f); sigma_h^2: %.4f (%.4f, %.4f)\n', ...
    mean(store_o(:,5)), qo(:,5), mean(store_o(:,4)), qo(:,4));
[~, bigz] = sort(abs(sqrt(lam_true).*epsn), 'descend');  % the largest true errors, in sd of e^{h_t/2}
fprintf(['periods with an outlier probability above 0.5: %d; among the ten largest lambda_t: %d; ' ...
    'among the ten largest |sqrt(lambda_t)*eps_t|: %d\n'], nnz(pout > .5), nnz(pout(big(1:10)) > .5), ...
    nnz(pout(bigz(1:10)) > .5));

%% the figures
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 12]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(1:T, y, 'Color', [.45 .45 .45], 'LineWidth', .5); box off; xlim([1 T]);
title('y_t', 'FontWeight', 'normal');
nexttile; hold on
hb = ssm.shaded_band(1:T, bt(:,1), bt(:,2));
ht = plot(1:T, vt, 'k--', 'LineWidth', 1);
hn = plot(1:T, vm_n, 'Color', [.55 .55 .55], 'LineWidth', 1);
hm = plot(1:T, vm_t, 'k', 'LineWidth', 1.2);
hold off; box off; xlim([1 T]); title('e^{h_t/2}', 'FontWeight', 'normal');
lg = legend([ht hm hb hn], {'true value', 'posterior mean, SV-t', ...
    '90% pointwise credible interval, SV-t', 'posterior mean, SV'}, 'NumColumns', 2, 'Box', 'off');
lg.Layout.Tile = 'south';
exportgraphics(fig, fullfile(here, 'fig_vol.png'), 'Resolution', 150); close(fig);

fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 6.5]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(store_theta(:,5), 'Color', [.35 .35 .35], 'LineWidth', .3); box off; xlim([0 nsim]);
title('the draws of \nu', 'FontWeight', 'normal');
nexttile; hold on
hh = histogram(store_theta(:,5), 60, 'Normalization', 'pdf', 'FaceColor', [.75 .75 .75], ...
    'EdgeColor', 'none');
hx = xline(nu_true, 'k--', 'LineWidth', 1);
hold off; box off; title('their histogram', 'FontWeight', 'normal');
lg = legend([hh hx], {'posterior draws of \nu', 'true value'}, 'Orientation', 'horizontal', 'Box', 'off');
lg.Layout.Tile = 'south';
exportgraphics(fig, fullfile(here, 'fig_nu.png'), 'Resolution', 150); close(fig);
