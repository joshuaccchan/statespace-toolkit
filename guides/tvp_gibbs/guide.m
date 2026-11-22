%% guide - the code of guides/tvp_gibbs/README.md: a Gibbs sampler for the TVP regression of
% guides/tvp_states. The sections called "the code to adapt" are all a sampler needs. Saves
% fig_chain.png next to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(1, 'twister');

%% the data of guides/tvp_states
T = 200; k = 2; sig2_true = 1;
Omega_true = diag([.005 .02]); beta0_true = [2; 1];
X = [ones(T,1) randn(T,1)];  % x_t = (1, x_t)'
beta_true = beta0_true' + cumsum(randn(T,k)*sqrt(Omega_true));  % T x k
y = sum(X.*beta_true, 2) + sqrt(sig2_true)*randn(T,1);

%% 1. the Gibbs sampler: the code to adapt
a0 = zeros(k,1); V0 = 100*ones(k,1);                   % beta_0 ~ N(a0, diag(V0))
nu_sig = 3; S_sig = 2;                                 % sig2 ~ IG(nu_sig, S_sig)
nu_om = 3; S_om = .02*ones(k,1);                       % omega_i^2 ~ IG(nu_om, S_om(i))
nsim = 20000; burnin = 1000;
Z = ssm.surform(X); ZZ = Z'*Z; Zy = Z'*y;              % computed once
H = kron(ssm.diffmat(T), speye(k));                    % H = H_T kron I_k
sig2 = var(y); omega2 = .01*ones(k,1); b0 = a0;        % starting values
store_theta = zeros(nsim, 1 + 2*k); store_beta = zeros(k*T, nsim);
for isim = 1:burnin + nsim
    iS = kron(speye(T), spdiags(1./omega2, 0, k, k));  % S^{-1} = I_T kron Omega^{-1}
    btil = [b0; zeros((T-1)*k,1)];                     % betatilde_0 = (beta_0', 0', ..., 0')'
    K = H'*iS*H + ZZ/sig2;                             % the coefficients: guides/tvp_states
    c = H'*iS*btil + Zy/sig2;
    beta = ssm.simulate_states(K, c);

    e = y - Z*beta;  % the measurement errors
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + e'*e/2));

    eta = reshape(H*beta - btil, k, T);  % the state errors, a row per coefficient
    omega2 = 1./gamrnd(nu_om + T/2, 1./(S_om + sum(eta.^2, 2)/2));

    Kb0 = 1./V0 + 1./omega2;  % beta_0 given beta_1 and Omega
    b0 = (a0./V0 + beta(1:k)./omega2)./Kb0 + randn(k,1)./sqrt(Kb0);

    if isim > burnin
        store_theta(isim - burnin, :) = [sig2 omega2' b0'];
        store_beta(:, isim - burnin) = beta;
    end
end

%% 1. the checks
L = 200;                                         % Bartlett truncation lag
IF = ssm.inefficiency_factor(store_theta, L);
pm = mean(store_theta); psd = std(store_theta); pq = quantile(store_theta, [.05 .95]);
names = {'sig2', 'omega2_1', 'omega2_2', 'beta0_1', 'beta0_2'};
truth = [sig2_true diag(Omega_true)' beta0_true'];
fmt = '%-8s true %.3f, posterior mean %.4f, 90%% interval (%.4f, %.4f), inefficiency %.1f\n';
fprintf('\nthe Gibbs sampler, %d draws after a burn-in of %d\n', nsim, burnin);
for i = 1:numel(names)
    fprintf(fmt, names{i}, truth(i), pm(i), pq(1,i), pq(2,i), IF(i));
end
% The posterior of (sig2, omega2_1, omega2_2) on a grid of their logs, with the coefficients
% and beta_0 integrated out: alpha = (beta_0', beta_1', ..., beta_T')' has prior mean
% (a0', ..., a0')' and precision G'*Q^{-1}*G = A0 + A1/omega2_1 + A2/omega2_2, with G the
% first differences of alpha and Q = diag(V0, Omega, ..., Omega). The grid spans 6
% standard deviations of the logged draws on either side of their mean.
G = kron(ssm.diffmat(T+1), speye(k));
Za = [sparse(T, k) Z]; ba = repmat(a0, T+1, 1);
A0 = G'*spdiags([1./V0; zeros(k*T,1)], 0, k*(T+1), k*(T+1))*G;
Ai = cell(1, k);
for i = 1:k
    sel = [zeros(k,1); repmat(double((1:k)' == i), T, 1)];
    Ai{i} = G'*spdiags(sel, 0, k*(T+1), k*(T+1))*G;
end
logig = @(x, nu, S) nu*log(S) - gammaln(nu) - (nu + 1)*log(x) - S./x;
span = @(x, n, s) linspace(mean(x) - s*std(x), mean(x) + s*std(x), n);
ng = 40;
gs = span(log(store_theta(:,1)), ng, 6); g1 = span(log(store_theta(:,2)), ng, 6);
g2 = span(log(store_theta(:,3)), ng, 6);
[S2, O1, O2] = ndgrid(exp(gs), exp(g1), exp(g2));
lp = zeros(ng, ng, ng); m0 = zeros(k, ng^3);
for j = 1:ng^3
    P = A0 + Ai{1}/O1(j) + Ai{2}/O2(j);
    [ll, ahat] = ssm.intlike(y, Za, speye(T)/S2(j), P, ba);
    lp(j) = ll + logig(S2(j), nu_sig, S_sig) + logig(O1(j), nu_om, S_om(1)) ...
        + logig(O2(j), nu_om, S_om(2)) ...
        + log(S2(j)) + log(O1(j)) + log(O2(j));  % the densities of the logs
    m0(:,j) = ahat(1:k);
end
w = exp(lp - max(lp(:))); w = w/sum(w(:));
gm = [sum(w.*S2, 'all'), sum(w.*O1, 'all'), sum(w.*O2, 'all'), (m0*w(:))'];
gsd = sqrt([sum(w.*S2.^2, 'all'), sum(w.*O1.^2, 'all'), sum(w.*O2.^2, 'all')] - gm(1:3).^2);
mcse = ssm.mcse(store_theta, L);
fprintf('\nthe posterior on a %d x %d x %d grid, from ssm.intlike\n', ng, ng, ng);
fprintf('largest weight on a face of the grid, relative to the largest weight: %.1e\n', ...
    max([reshape(w([1 end],:,:), 1, []), reshape(w(:,[1 end],:), 1, []), ...
    reshape(w(:,:,[1 end]), 1, [])])/max(w(:)));
fprintf('posterior means, Gibbs minus grid, in Monte Carlo standard errors: %s\n', ...
    mat2str(round((pm - gm)./mcse, 1)));
fprintf('posterior standard deviations of the variances, Gibbs over grid: %s\n', ...
    mat2str(round(psd(1:3)./gsd, 3)));
mb = zeros(k*T, 1);                              % E(beta | y) from the grid
for j = find(w(:) > 1e-12)'
    [~, ahat] = ssm.intlike(y, Za, speye(T)/S2(j), A0 + Ai{1}/O1(j) + Ai{2}/O2(j), ba);
    mb = mb + w(j)*ahat(k+1:end);
end
IFb = ssm.inefficiency_factor(store_beta', L)';
zb = (mean(store_beta, 2) - mb)./ssm.mcse(store_beta', L)';
fprintf(['coefficients: posterior means, Gibbs minus grid, at most %.1f Monte Carlo ' ...
    'standard errors\n'], max(abs(zb)));
fprintf('coefficients: inefficiency factors from %.1f to %.1f\n', min(IFb), max(IFb));

% the coefficients with the parameters known, as in guides/tvp_states
iSk = kron(speye(T), inv(Omega_true));
Kk = H'*iSk*H + ZZ/sig2_true;
ck = H'*iSk*[beta0_true; zeros((T-1)*k,1)] + Zy/sig2_true;
[dk, bhat_k] = ssm.simulate_states(Kk, ck, 5000);
bk = quantile(dk, [.05 .95], 2); bg = quantile(store_beta, [.05 .95], 2);
bt = reshape(beta_true', [], 1);  % stacked by period
fprintf('\nthe coefficients, with the parameters estimated and known:\n');
fprintf('largest difference between the posterior means: %.2f posterior sd\n', ...
    max(abs(mean(store_beta, 2) - bhat_k)./std(store_beta, 0, 2)));
for i = 1:k
    r = i:k:k*T;
    fprintf(['coefficient %d: 90%% intervals %.2f and %.2f wide on average; they contain the ' ...
        'truth in %d and %d of %d periods\n'], i, mean(bg(r,2) - bg(r,1)), ...
        mean(bk(r,2) - bk(r,1)), ...
        nnz(bt(r) >= bg(r,1) & bt(r) <= bg(r,2)), nnz(bt(r) >= bk(r,1) & bt(r) <= bk(r,2)), T);
end

%% 2. a constant slope: the adaptation
rng(2, 'twister');
b2_true = 1; om2_true = Omega_true(1,1); b10_true = beta0_true(1);
x = X(:,2);
tau_true = b10_true + cumsum(sqrt(om2_true)*randn(T,1));  % the intercept, a random walk
y2 = tau_true + b2_true*x + sqrt(sig2_true)*randn(T,1);
as = 0; Vs = 100;                                         % b2 ~ N(as, Vs)
H1 = ssm.diffmat(T); HH1 = H1'*H1;
sig2 = var(y2); om2 = .01; b10 = a0(1); b2 = as;          % starting values
store_cs = zeros(nsim, 4);
for isim = 1:burnin + nsim
    K = HH1/om2 + speye(T)/sig2;                          % the intercept: uc_states on y - b2*x
    c = H1'*[b10; zeros(T-1,1)]/om2 + (y2 - b2*x)/sig2;
    tau = ssm.simulate_states(K, c);

    Kb2 = 1/Vs + x'*x/sig2;  % the constant slope: a regression
    b2 = (as/Vs + x'*(y2 - tau)/sig2)/Kb2 + randn/sqrt(Kb2);

    e = y2 - tau - b2*x;
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + e'*e/2));

    eta = H1*tau - [b10; zeros(T-1,1)];
    om2 = 1/gamrnd(nu_om + T/2, 1/(S_om(1) + eta'*eta/2));

    Kb10 = 1/V0(1) + 1/om2;
    b10 = (a0(1)/V0(1) + tau(1)/om2)/Kb10 + randn/sqrt(Kb10);

    if isim > burnin
        store_cs(isim - burnin, :) = [sig2 om2 b2 b10];
    end
end

%% 2. the checks
IFc = ssm.inefficiency_factor(store_cs, L);
pmc = mean(store_cs); psdc = std(store_cs); pqc = quantile(store_cs, [.05 .95]);
namesc = {'sig2', 'omega2_1', 'b2', 'beta0_1'}; truthc = [sig2_true om2_true b2_true b10_true];
fprintf('\na constant slope, %d draws after a burn-in of %d\n', nsim, burnin);
for i = 1:4
    fprintf(fmt, namesc{i}, truthc(i), pmc(i), pqc(1,i), pqc(2,i), IFc(i));
end
% the posterior of (sig2, omega2_1) on a grid of their logs, with the intercept, its initial
% value and the slope in the states: alpha = (beta0_1, beta_11, ..., beta_1T, b2)'
Gc = ssm.diffmat(T+1);
Zc = [sparse(T,1) speye(T) sparse(x)]; bc = [a0(1)*ones(T+1,1); as];
Ac0 = blkdiag(Gc'*spdiags([1/V0(1); zeros(T,1)], 0, T+1, T+1)*Gc, 1/Vs);
Ac1 = blkdiag(Gc'*spdiags([0; ones(T,1)], 0, T+1, T+1)*Gc, 0);
nc = 60;
cs = span(log(store_cs(:,1)), nc, 8); co = span(log(store_cs(:,2)), nc, 8);
[Sc, Oc] = ndgrid(exp(cs), exp(co));
lpc = zeros(nc); mc = zeros(2, nc^2);
for j = 1:nc^2
    [ll, ahat] = ssm.intlike(y2, Zc, speye(T)/Sc(j), Ac0 + Ac1/Oc(j), bc);
    lpc(j) = ll + logig(Sc(j), nu_sig, S_sig) + logig(Oc(j), nu_om, S_om(1)) ...
        + log(Sc(j)) + log(Oc(j));
    mc(:,j) = ahat([end 1]);  % b2 and beta0_1
end
wc = exp(lpc - max(lpc(:))); wc = wc/sum(wc(:));
gmc = [sum(wc.*Sc, 'all'), sum(wc.*Oc, 'all'), (mc*wc(:))'];
fprintf('the posterior on a %d x %d grid, from ssm.intlike\n', nc, nc);
fprintf('largest weight on the edge of the grid, relative to the largest weight: %.1e\n', ...
    max([wc(1,:), wc(end,:), wc(:,1)', wc(:,end)'])/max(wc(:)));
fprintf('posterior means, Gibbs minus grid, in Monte Carlo standard errors: %s\n', ...
    mat2str(round((pmc - gmc)./ssm.mcse(store_cs, L), 1)));

%% the figure: the draws of the three variances of section 1 and their histograms, with the
% grid posterior
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 10]);
lab = {'\sigma^2', '\omega_1^2', '\omega_2^2'};
gx = {exp(gs), exp(g1), exp(g2)};
pw = {squeeze(sum(w, [2 3]))'./exp(gs)/(gs(2) - gs(1)), ...
    squeeze(sum(w, [1 3]))./exp(g1)/(g1(2) - g1(1)), ...
    squeeze(sum(w, [1 2]))'./exp(g2)/(g2(2) - g2(1))};
for i = 1:3
    subplot(2, 3, i);
    plot(store_theta(:,i), 'Color', [.35 .35 .35], 'LineWidth', .3); box off
    xlim([0 nsim]); title(lab{i}, 'FontWeight', 'normal');
    subplot(2, 3, 3 + i); hold on
    histogram(store_theta(:,i), 60, 'Normalization', 'pdf', 'FaceColor', [.75 .75 .75], ...
        'EdgeColor', 'none');
    keep = pw{i} > 1e-3*max(pw{i});
    plot(gx{i}(keep), pw{i}(keep), 'k', 'LineWidth', 1.2);
    xline(truth(i), 'k--', 'LineWidth', 1);
    hold off; box off;
end
exportgraphics(fig, fullfile(here, 'fig_chain.png'), 'Resolution', 150); close(fig);
