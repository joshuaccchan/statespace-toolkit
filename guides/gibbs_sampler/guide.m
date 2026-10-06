%% guide - the code of guides/gibbs_sampler/README.md: a Gibbs sampler that estimates the
% two variances and the initial value of the trend of the local level model of
% guides/uc_states, along with the trend, on the data of that guide; then the same with an
% AR(1) transitory component. The sections called "the code to adapt" are all a sampler
% needs. The checks compute the posterior on a grid with ssm.intlike, without the sampler.
% Saves fig_chain.png next to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(1, 'twister');

%% the data of guides/uc_states
T = 200; sig2_true = 1; omega2_true = .05; tau0_true = 2;
tau_true = tau0_true + cumsum(sqrt(omega2_true)*randn(T,1));
y = tau_true + sqrt(sig2_true)*randn(T,1);

%% 1. the Gibbs sampler: the code to adapt
a0 = 5; b0 = 100;                                        % tau0 ~ N(a0, b0)
nu_sig = 3; S_sig = 2;                                   % sig2 ~ IG(nu_sig, S_sig)
nu_om = 3; S_om = 2*.25^2;                               % omega2 ~ IG(nu_om, S_om)
nsim = 20000; burnin = 1000;
H = ssm.diffmat(T); HH = H'*H;                           % computed once
sig2 = 1; omega2 = .1; tau0 = a0;                        % starting values
store_theta = zeros(nsim, 3); store_tau = zeros(T, nsim);
for isim = 1:burnin + nsim
    K = HH/omega2 + speye(T)/sig2;                       % the states: guides/uc_states
    c = H'*[tau0; zeros(T-1,1)]/omega2 + y/sig2;
    tau = ssm.simulate_states(K, c);
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + (y - tau)'*(y - tau)/2));
    eta = H*tau - [tau0; zeros(T-1,1)];                  % the state equation residuals
    omega2 = 1/gamrnd(nu_om + T/2, 1/(S_om + eta'*eta/2));
    Ktau0 = 1/b0 + 1/omega2;                             % tau0 given tau_1 and omega2
    tau0 = (a0/b0 + tau(1)/omega2)/Ktau0 + randn/sqrt(Ktau0);
    if isim > burnin
        store_theta(isim - burnin, :) = [sig2 omega2 tau0];
        store_tau(:, isim - burnin) = tau;
    end
end

%% 1. the checks
L = 200;                                                 % Bartlett truncation lag
IF = ssm.inefficiency_factor(store_theta, L);
pm = mean(store_theta); psd = std(store_theta); pq = quantile(store_theta, [.05 .95]);
names = {'sig2', 'omega2', 'tau0'}; truth = [sig2_true omega2_true tau0_true];
fmt = '%-6s true %.3f, posterior mean %.3f, 90%% interval (%.3f, %.3f), inefficiency %.1f\n';
fprintf('\nthe Gibbs sampler, %d draws after a burn-in of %d\n', nsim, burnin);
for k = 1:3
    fprintf(fmt, names{k}, truth(k), pm(k), pq(1,k), pq(2,k), IF(k));
end
% The posterior of (sig2, omega2) on a grid of their logs, with the states and tau0
% integrated out: alpha = (tau0, tau_1, ..., tau_T)' has prior mean a0 and precision
% G'*Q^{-1}*G = A0 + A1/omega2, with Q = diag(b0, omega2, ..., omega2). The grid spans 8
% standard deviations of the logged draws on either side of their mean.
G = ssm.diffmat(T+1); Z = [sparse(T,1) speye(T)]; b = a0*ones(T+1,1);
A0 = sparse(1, 1, 1/b0, T+1, T+1);
A1 = G'*spdiags([0; ones(T,1)], 0, T+1, T+1)*G;
e1 = sparse(1, 1, 1, T+1, 1);
logig = @(x, nu, S) nu*log(S) - gammaln(nu) - (nu + 1)*log(x) - S./x;
span = @(x, n) linspace(mean(x) - 8*std(x), mean(x) + 8*std(x), n);
ng = 60;
ls = span(log(store_theta(:,1)), ng); lo = span(log(store_theta(:,2)), ng);
[S2, O2] = ndgrid(exp(ls), exp(lo));
lp = zeros(ng); m0 = lp; v0 = lp;
for k = 1:ng^2
    P = A0 + A1/O2(k);
    [ll, ahat, Kg] = ssm.intlike(y, Z, speye(T)/S2(k), P, b);
    lp(k) = ll + logig(S2(k), nu_sig, S_sig) + logig(O2(k), nu_om, S_om) ...
        + log(S2(k)) + log(O2(k));                       % the densities of the logs
    m0(k) = ahat(1);
    v0(k) = sum((chol(Kg, 'lower')\e1).^2);              % (K^{-1})_{11}
end
w = exp(lp - max(lp(:))); w = w/sum(w(:));
gm = [sum(w.*S2, 'all'), sum(w.*O2, 'all'), sum(w.*m0, 'all')];
gsd = sqrt([sum(w.*S2.^2, 'all'), sum(w.*O2.^2, 'all'), ...
    sum(w.*(v0 + m0.^2), 'all')] - gm.^2);
mcse = ssm.mcse(store_theta, L);
fprintf('\nthe posterior on a %d x %d grid, from ssm.intlike\n', ng, ng);
fprintf('largest weight on the edge of the grid, relative to the largest weight: %.1e\n', ...
    max([w(1,:), w(end,:), w(:,1)', w(:,end)'])/max(w(:)));
fprintf('posterior means, Gibbs minus grid, in Monte Carlo standard errors: %s\n', ...
    mat2str(round((pm - gm)./mcse, 1)));
fprintf('posterior standard deviations, Gibbs over grid: %s\n', mat2str(round(psd./gsd, 3)));
mt = zeros(T, 1);                                        % E(tau | y) from the grid
for k = find(w(:) > 1e-12)'
    [~, ahat] = ssm.intlike(y, Z, speye(T)/S2(k), A0 + A1/O2(k), b);
    mt = mt + w(k)*ahat(2:end);
end
IFt = ssm.inefficiency_factor(store_tau', L)';
zt = (mean(store_tau, 2) - mt)./ssm.mcse(store_tau', L)';
fprintf(['trend: posterior means, Gibbs minus grid, at most %.1f Monte Carlo ' ...
    'standard errors\n'], max(abs(zt)));
fprintf('trend: inefficiency factors from %.1f to %.1f\n', min(IFt), max(IFt));

% the trend with the parameters known, as in guides/uc_states
Kk = HH/omega2_true + speye(T)/sig2_true;
ck = H'*[tau0_true; zeros(T-1,1)]/omega2_true + y/sig2_true;
[dk, tauhat_k] = ssm.simulate_states(Kk, ck, 5000);
bk = quantile(dk, [.05 .95], 2);
bt = quantile(store_tau, [.05 .95], 2);
fprintf('\nthe trend, with the parameters estimated and known:\n');
fprintf('largest difference between the posterior means: %.2f posterior sd\n', ...
    max(abs(mean(store_tau, 2) - tauhat_k)./std(store_tau, 0, 2)));
fprintf('average width of the 90%% intervals: %.2f and %.2f\n', ...
    mean(bt(:,2) - bt(:,1)), mean(bk(:,2) - bk(:,1)));
fprintf('the intervals contain the true trend in %d and %d of the %d periods\n', ...
    nnz(tau_true >= bt(:,1) & tau_true <= bt(:,2)), ...
    nnz(tau_true >= bk(:,1) & tau_true <= bk(:,2)), T);

%% 2. an AR(1) transitory component: the adaptation
rng(2, 'twister');
rho_true = .5;
tau_true2 = tau0_true + cumsum(sqrt(omega2_true)*randn(T,1));
y2 = tau_true2 + filter(1, [1 -rho_true], sqrt(sig2_true)*randn(T,1));
sig2 = 1; omega2 = .1; tau0 = a0; rho = 0;
store_ar = zeros(nsim, 4);
for isim = 1:burnin + nsim
    Hrho = ssm.diffmat(T, rho); HHrho = Hrho'*Hrho;
    K = HH/omega2 + HHrho/sig2;                          % Step 4 of guides/uc_states
    c = H'*[tau0; zeros(T-1,1)]/omega2 + HHrho*y2/sig2;
    tau = ssm.simulate_states(K, c);
    e = y2 - tau;                                        % the transitory component
    Krho = e(1:T-1)'*e(1:T-1)/sig2;
    rhohat = (e(1:T-1)'*e(2:T))/(e(1:T-1)'*e(1:T-1));
    rho = ssm.tnormrnd(rhohat, 1/Krho, -1, 1);
    u = ssm.diffmat(T, rho)*e;                           % the AR(1) residuals
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + u'*u/2));
    eta = H*tau - [tau0; zeros(T-1,1)];
    omega2 = 1/gamrnd(nu_om + T/2, 1/(S_om + eta'*eta/2));
    Ktau0 = 1/b0 + 1/omega2;
    tau0 = (a0/b0 + tau(1)/omega2)/Ktau0 + randn/sqrt(Ktau0);
    if isim > burnin
        store_ar(isim - burnin, :) = [rho sig2 omega2 tau0];
    end
end

%% 2. the checks
IFa = ssm.inefficiency_factor(store_ar, L);
pma = mean(store_ar); psda = std(store_ar); pqa = quantile(store_ar, [.05 .95]);
namesa = {'rho', 'sig2', 'omega2', 'tau0'}; trutha = [rho_true truth];
fprintf('\nAR(1) transitory component, %d draws after a burn-in of %d\n', nsim, burnin);
for k = 1:4
    fprintf(fmt, namesa{k}, trutha(k), pma(k), pqa(1,k), pqa(2,k), IFa(k));
end
% the posterior of (rho, sig2, omega2) on a grid of (atanh(rho), log(sig2), log(omega2)),
% with tau0 in the states as above; H_rho'*H_rho = I + rho*B1 + rho^2*B2
na = 24;
g1 = span(atanh(store_ar(:,1)), na);
g2 = span(log(store_ar(:,2)), na); g3 = span(log(store_ar(:,3)), na);
[R3, S3, O3] = ndgrid(tanh(g1), exp(g2), exp(g3));
Lag = sparse(2:T, 1:T-1, 1, T, T); B1 = -(Lag + Lag'); B2 = Lag'*Lag;
lpa = zeros(na, na, na); m0a = lpa;
for k = 1:na^3
    iR = (speye(T) + R3(k)*B1 + R3(k)^2*B2)/S3(k);
    [ll, ahat] = ssm.intlike(y2, Z, iR, A0 + A1/O3(k), b);
    lpa(k) = ll + logig(S3(k), nu_sig, S_sig) + logig(O3(k), nu_om, S_om) ...
        + log((1 - R3(k)^2)/2) + log(S3(k)) + log(O3(k)); % rho ~ U(-1, 1)
    m0a(k) = ahat(1);
end
wa = exp(lpa - max(lpa(:))); wa = wa/sum(wa(:));
gma = [sum(wa.*R3, 'all'), sum(wa.*S3, 'all'), sum(wa.*O3, 'all'), sum(wa.*m0a, 'all')];
gsda = sqrt([sum(wa.*R3.^2, 'all'), sum(wa.*S3.^2, 'all'), ...
    sum(wa.*O3.^2, 'all')] - gma(1:3).^2);
mcsea = ssm.mcse(store_ar, L);
fprintf('the posterior on a %d x %d x %d grid, from ssm.intlike\n', na, na, na);
fprintf('largest weight on a face of the grid, relative to the largest weight: %.1e\n', ...
    max([reshape(wa([1 end],:,:), 1, []), reshape(wa(:,[1 end],:), 1, []), ...
    reshape(wa(:,:,[1 end]), 1, [])])/max(wa(:)));
fprintf('posterior means, Gibbs minus grid, in Monte Carlo standard errors: %s\n', ...
    mat2str(round((pma - gma)./mcsea, 1)));
fprintf('posterior standard deviations of rho, sig2, omega2, Gibbs over grid: %s\n', ...
    mat2str(round(psda(1:3)./gsda, 3)));

%% the figure: the draws of section 1 and their histograms, with the grid posterior
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 10]);
lab = {'\sigma^2', '\omega^2', '\tau_0'};
pw = {sum(w, 2)'./exp(ls)/(ls(2) - ls(1)), sum(w, 1)./exp(lo)/(lo(2) - lo(1))};
gx = {exp(ls), exp(lo)};
x0 = linspace(min(store_theta(:,3)), max(store_theta(:,3)), 200);
gx{3} = x0;                                              % a mixture of normals for tau0
pw{3} = sum(w(:)'.*exp(-(x0' - m0(:)').^2./(2*v0(:)'))./sqrt(2*pi*v0(:)'), 2)';
for k = 1:3
    subplot(2, 3, k);
    plot(store_theta(:,k), 'Color', [.35 .35 .35], 'LineWidth', .3); box off
    xlim([0 nsim]); title(lab{k}, 'FontWeight', 'normal');
    subplot(2, 3, 3 + k); hold on
    histogram(store_theta(:,k), 60, 'Normalization', 'pdf', 'FaceColor', [.75 .75 .75], ...
        'EdgeColor', 'none');
    keep = pw{k} > 1e-3*max(pw{k});
    plot(gx{k}(keep), pw{k}(keep), 'k', 'LineWidth', 1.2);
    xline(truth(k), 'k--', 'LineWidth', 1);
    hold off; box off;
end
exportgraphics(fig, fullfile(here, 'fig_chain.png'), 'Resolution', 150); close(fig);
