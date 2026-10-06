%% guide - the code of guides/integrated_likelihood/README.md: the integrated
% likelihood of the local level model of guides/uc_states, from ssm.intlike, first with
% the initial value of the trend known and then with it integrated out as well; then a
% sampler that draws the two variances with the trend integrated out, on the data of
% that guide. The sections called "the code to adapt" are all a sampler needs. The
% checks compare ssm.intlike with the dense normal density of y, and the sampler with the
% Gibbs sampler of guides/gibbs_sampler and with the posterior on a grid. Saves
% fig_acf.png next to this file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(1, 'twister');

%% the data of guides/uc_states
T = 200; sig2_true = 1; omega2_true = .05; tau0_true = 2;
tau_true = tau0_true + cumsum(sqrt(omega2_true)*randn(T,1));
y = tau_true + sqrt(sig2_true)*randn(T,1);

%% 1. the integrated likelihood: the code to adapt
sig2 = 1; omega2 = .05; tau0 = 2;                        % any parameter values
H = ssm.diffmat(T); HH = H'*H;
ll = ssm.intlike(y, speye(T), speye(T)/sig2, HH/omega2, tau0*ones(T,1));

%% 1. the checks
% y ~ N(tau0*1, sig2*I + omega2*(H'*H)^{-1}), from dense matrices
lnorm = @(x, m, S) -numel(x)/2*log(2*pi) - sum(log(diag(chol(S)))) - (x - m)'*(S\(x - m))/2;
V = inv(full(HH));
pars = [1 .05 2; .5 .2 0; 2 .01 4; 1.3 .5 -1];
d = zeros(size(pars, 1), 1);
for k = 1:size(pars, 1)
    lk = ssm.intlike(y, speye(T), speye(T)/pars(k,1), HH/pars(k,2), pars(k,3)*ones(T,1));
    ld = lnorm(y, pars(k,3), pars(k,1)*eye(T) + pars(k,2)*V);
    d(k) = abs(lk - ld)/abs(ld);
end
fprintf('\nlog p(y | sig2, omega2, tau0) = %.4f at the true values\n', ll);
fprintf('against the dense normal density at %d parameter values: at most %.1e (relative)\n', ...
    size(pars, 1), max(d));

%% 2. the initial value integrated out as well: the code to adapt
a0 = 5; b0 = 100;                                        % tau0 ~ N(a0, b0)
G = ssm.diffmat(T+1);                                    % alpha = (tau0, tau')'
Z = [sparse(T,1) speye(T)];
b = a0*ones(T+1,1);                                      % the prior mean of alpha
P = G'*spdiags([1/b0; ones(T,1)/omega2], 0, T+1, T+1)*G;  % its prior precision
ll0 = ssm.intlike(y, Z, speye(T)/sig2, P, b);

%% 2. the checks
ld = lnorm(y, a0, sig2*eye(T) + omega2*V + b0*ones(T));
fprintf('\nlog p(y | sig2, omega2) = %.4f, tau0 integrated out; dense %.1e (relative)\n', ...
    ll0, abs(ll0 - ld)/abs(ld));

%% 3. the variances without the states: the code to adapt
nu_sig = 3; S_sig = 2; nu_om = 3; S_om = 2*.25^2;        % as in guides/gibbs_sampler
logig = @(x, nu, S) nu*log(S) - gammaln(nu) - (nu + 1)*log(x) - S./x;
A0 = sparse(1, 1, 1/b0, T+1, T+1);                       % P = A0 + A1/omega2
A1 = G'*spdiags([0; ones(T,1)], 0, T+1, T+1)*G;
lpost = @(phi) ssm.intlike(y, Z, speye(T)*exp(-phi(1)), A0 + A1*exp(-phi(2)), b) ...
    + logig(exp(phi(1)), nu_sig, S_sig) + logig(exp(phi(2)), nu_om, S_om) + sum(phi);
% the proposal: a t distribution with 5 degrees of freedom, located at the mode of lpost
% and scaled by the inverse of its negative Hessian there, from finite differences
phihat = fminsearch(@(phi) -lpost(phi), [0; log(.1)]);
h = 1e-3; E = h*eye(2); Hs = zeros(2);
for i = 1:2
    for j = 1:2
        Hs(i,j) = (lpost(phihat + E(:,i) + E(:,j)) - lpost(phihat + E(:,i) - E(:,j)) ...
            - lpost(phihat - E(:,i) + E(:,j)) + lpost(phihat - E(:,i) - E(:,j)))/(4*h^2);
    end
end
C = chol(inv(-Hs), 'lower'); nu = 5;
logq = @(phi) -(nu + 2)/2*log(1 + sum((C\(phi - phihat)).^2)/nu);
nsim = 20000; burnin = 1000;
rng(2, 'twister'); t0 = tic;
phi = phihat; lw = lpost(phi) - logq(phi); acc = 0;
store_theta = zeros(nsim, 3);
for isim = 1:burnin + nsim
    phic = phihat + C*randn(2,1)/sqrt(gamrnd(nu/2, 2/nu));  % a draw from the proposal
    lwc = lpost(phic) - logq(phic);
    if log(rand) < lwc - lw                              % independence-chain MH
        phi = phic; lw = lwc; acc = acc + (isim > burnin);
    end
    sig2 = exp(phi(1)); omega2 = exp(phi(2));
    P = A0 + A1/omega2;
    alpha = ssm.simulate_states(P + Z'*Z/sig2, P*b + Z'*y/sig2);  % Theorem 9.1
    if isim > burnin
        store_theta(isim - burnin, :) = [sig2 omega2 alpha(1)];
    end
end
time_i = toc(t0);

%% 3. the checks
% the Gibbs sampler of guides/gibbs_sampler, on the same data
rng(2, 'twister'); t0 = tic;
sig2 = 1; omega2 = .1; tau0 = a0;
store_g = zeros(nsim, 3);
for isim = 1:burnin + nsim
    K = HH/omega2 + speye(T)/sig2;
    c = H'*[tau0; zeros(T-1,1)]/omega2 + y/sig2;
    tau = ssm.simulate_states(K, c);
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + (y - tau)'*(y - tau)/2));
    eta = H*tau - [tau0; zeros(T-1,1)];
    omega2 = 1/gamrnd(nu_om + T/2, 1/(S_om + eta'*eta/2));
    Ktau0 = 1/b0 + 1/omega2;
    tau0 = (a0/b0 + tau(1)/omega2)/Ktau0 + randn/sqrt(Ktau0);
    if isim > burnin
        store_g(isim - burnin, :) = [sig2 omega2 tau0];
    end
end
time_g = toc(t0);
L = 200;
IFg = ssm.inefficiency_factor(store_g, L); IFi = ssm.inefficiency_factor(store_theta, L);
fprintf('\n%d draws after a burn-in of %d; the MH step accepts %.0f%% of proposals\n', ...
    nsim, burnin, 100*acc/nsim);
fprintf('inefficiency factors of sig2, omega2, tau0: Gibbs %s, without the states %s\n', ...
    mat2str(round(IFg, 1)), mat2str(round(IFi, 1)));
fprintf('time: Gibbs %.1f s, without the states %.1f s (ratio %.1f)\n', time_g, time_i, ...
    time_i/time_g);
fprintf('effective draws per second: Gibbs %s, without the states %s\n', ...
    mat2str(round(nsim./IFg/time_g, -1)), mat2str(round(nsim./IFi/time_i, -1)));

% the posterior of (sig2, omega2) on a grid of their logs, from lpost
span = @(x, n) linspace(mean(x) - 8*std(x), mean(x) + 8*std(x), n);
ng = 60;
ls = span(log(store_theta(:,1)), ng); lo = span(log(store_theta(:,2)), ng);
lp = zeros(ng); m0 = lp;
for i = 1:ng
    for j = 1:ng
        lp(i,j) = lpost([ls(i); lo(j)]);
        [~, ahat] = ssm.intlike(y, Z, speye(T)*exp(-ls(i)), A0 + A1*exp(-lo(j)), b);
        m0(i,j) = ahat(1);
    end
end
w = exp(lp - max(lp(:))); w = w/sum(w(:));
[S2, O2] = ndgrid(exp(ls), exp(lo));
gm = [sum(w.*S2, 'all'), sum(w.*O2, 'all'), sum(w.*m0, 'all')];
mcse = ssm.mcse(store_theta, L);
fprintf('largest weight on the edge of the grid, relative to the largest weight: %.1e\n', ...
    max([w(1,:), w(end,:), w(:,1)', w(:,end)'])/max(w(:)));
fprintf('posterior means %s, Gibbs %s, grid %s\n', mat2str(round(mean(store_theta), 3)), ...
    mat2str(round(mean(store_g), 3)), mat2str(round(gm, 3)));
fprintf('posterior means, minus the grid, in Monte Carlo standard errors: %s\n', ...
    mat2str(round((mean(store_theta) - gm)./mcse, 1)));

%% 4. an AR(1) transitory component: the adaptation
rho = .8;
Hrho = ssm.diffmat(T, rho);
ll_ar = ssm.intlike(y, speye(T), Hrho'*Hrho/sig2_true, HH/omega2_true, tau0_true*ones(T,1));
ld = lnorm(y, tau0_true, sig2_true*inv(full(Hrho'*Hrho)) + omega2_true*V);
fprintf(['\nAR(1) transitory component, rho = %.1f: against the dense normal density ' ...
    '%.1e (relative)\n'], rho, abs(ll_ar - ld)/abs(ld));

%% the figure: the autocorrelations of the draws of omega2
acf = @(x, l) sum((x(1+l:end) - mean(x)).*(x(1:end-l) - mean(x)))/sum((x - mean(x)).^2);
lags = 0:100;
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 14 7]); hold on
plot(lags, arrayfun(@(l) acf(store_g(:,2), l), lags), 'k--', 'LineWidth', 1.2);
plot(lags, arrayfun(@(l) acf(store_theta(:,2), l), lags), 'k', 'LineWidth', 1.2);
hold off; box off; xlabel('lag'); ylim([-.1 1]);
legend({'Gibbs sampler', 'drawn without the states'}, 'Location', 'northeast'); legend boxoff
exportgraphics(fig, fullfile(here, 'fig_acf.png'), 'Resolution', 150); close(fig);
