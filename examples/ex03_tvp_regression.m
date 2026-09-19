%% ex03 - A time-varying parameter Phillips curve, and the precision sampler timed
%
% y_t = x_t'*beta_t + eps_t, eps_t ~ N(0, sig2), and beta_t = beta_{t-1} + u_t,
% u_t ~ N(0, Omega), Omega = diag(omega_1^2, ..., omega_k^2), with
% beta_1 ~ N(beta0, Omega). Stacked over t, y = Z*beta + eps with
% Z = ssm.surform(X) = diag(x_1', ..., x_T'). Given the parameters, the path
% beta = (beta_1', ..., beta_T')' is N(betahat, K^{-1}) with
% K = kron(H'*H, Omega^{-1}) + Z'*Z/sig2, banded, where H = ssm.diffmat(T), and
% ssm.simulate_states draws the whole path at once: the precision sampler of Chan and
% Jeliazkov (2009).
%
% 1. US PCE inflation on the output gap and lagged inflation, 1960Q2-2019Q4, with
%    x_t = (1, gap_t, y_{t-1})': linreg_tvp.m of the chan-jeliazkov-2009 repository,
%    with Z by ssm.surform and the means and draws of beta and beta0 by
%    ssm.simulate_states.
% 2. Timing on generated data with k = 1, 2, 5 and 10 coefficients and T = 1,000 and
%    5,000: one draw of the path by ssm.simulate_states, with the prior precision and
%    K formed each time as in a Gibbs sampler, against one draw by the Kalman filter
%    with backward sampling of Carter and Kohn (1994) and Fruhwirth-Schnatter (1994),
%    written with k x k matrix operations, and for k = 1 also with scalar arithmetic.
%    Both Carter-Kohn samplers are first checked against betahat and diag(K^{-1}).
%    Which sampler is faster depends on k and on how each is coded; the table reports
%    the times on the machine that runs it.
%
% See Chan (forthcoming), Section 9.3, for a textbook treatment.
%
% See:
% Carter, C.K. and Kohn, R. (1994). On Gibbs Sampling for State Space Models,
% Biometrika, 81: 541-553.
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.3.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
% Fruhwirth-Schnatter, S. (1994). Data Augmentation and Dynamic Linear Models,
% Journal of Time Series Analysis, 15: 183-202.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex03: a TVP Phillips curve, and the precision sampler timed ===\n');

%% 1. TVP Phillips curve on US PCE inflation and the output gap
rng(42);
nsim = 20000; burnin = 1000;
data = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', ...
    'USPCE_OutputGap.csv'), 'Range', 'B2:C241');
infl = data(:,1);                                          % PCE inflation
gap  = data(:,2);                                          % output gap
y = infl(2:end);
g = gap(2:end);
ylag = infl(1:end-1);                                      % y_{t-1}
T = length(y);
X = [ones(T,1), g, ylag];
k = size(X,2);

% prior hyperparameters
beta00 = zeros(k,1);
iVbeta0 = 1/100*eye(k);
nu_sig = 3;  S_sig = 1;                                    % IG prior for sigma^2
nu_om = 3;                                                 % IG prior for omega_j^2
S_om  = [0.125; 0.025; 0.025].^2*(nu_om-1);

% initialize chain
beta0 = zeros(k,1);
beta_ols = (X'*X)\(X'*y);
sig2 = mean((y - X*beta_ols).^2);
omega2 = 0.01^2 * ones(k,1);

H = ssm.diffmat(T);
HH = H'*H;
Z = ssm.surform(X);
ZZ = Z'*Z;
Zy = Z'*y;
store_beta = zeros(nsim, T*k);
store_theta = zeros(nsim, 2*k + 1);                        % [beta0', sig2, omega2']
for isim = 1:nsim + burnin
    % sample beta
    iOmega = sparse(1:k,1:k,1./omega2);
    P = kron(HH, iOmega);                                  % prior precision
    Kbeta = P + ZZ/sig2;
    beta = ssm.simulate_states(Kbeta, P*repmat(beta0,T,1) + Zy/sig2);

    % sample sigma^2
    e = y - Z*beta;
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + e'*e/2));

    % sample omega_j^2
    Beta = reshape(beta,k,T)';
    SSE = sum((Beta - [beta0'; Beta(1:T-1,:)]).^2)';
    omega2 = 1./gamrnd(nu_om + T/2, 1./(S_om + 0.5*SSE));

    % sample beta0
    Kbeta0 = iVbeta0 + sparse(1:k,1:k,1./omega2);
    beta0 = ssm.simulate_states(Kbeta0, iVbeta0*beta00 + beta(1:k)./omega2);

    if isim > burnin
        isave = isim - burnin;
        store_beta(isave,:) = beta';
        store_theta(isave,:) = [beta0', sig2, omega2'];
    end
end
Beta_mean = reshape(mean(store_beta, 1),k,T)';
Beta_q = permute(reshape(quantile(store_beta, [0.05,0.95], 1)',k,T,2),[2,1,3]);
theta_hat = mean(store_theta);
theta_q   = quantile(store_theta, [0.05 0.95]);
names = {'beta0(intercept)', 'beta0(output gap)', 'beta0(lagged infl)', 'sig2', ...
    'omega2(intercept)', 'omega2(output gap)', 'omega2(lagged infl)'};
fprintf('\n1. US PCE inflation, 1960Q2-2019Q4, T = %d, %d draws after %d burn-in\n', ...
    T, nsim, burnin);
fprintf('   posterior means and 90%% CIs\n');
for j = 1:numel(names)
    fprintf('   %-20s = %.4f  [%.4f, %.4f]\n', names{j}, theta_hat(j), theta_q(:,j));
end

%% 2. Timing: the precision sampler against the Carter-Kohn sampler
% First the Carter-Kohn samplers against the exact moments, on generated data
rng(1, 'twister');
ndraws = 1000;
fprintf('\n2. Timing, on generated data\n');
fprintf('   Carter-Kohn samplers against betahat and diag(K^{-1}), T = 200, %d draws:\n', ...
    ndraws);
for kc = [1 2]
    [yc, Xc, sig2c, omega2c] = generate_tvp(200, kc);
    [Kc, bhat] = tvp_posterior(yc, Xc, sig2c, omega2c, zeros(kc,1));
    draws = zeros(200*kc, ndraws);
    for r = 1:ndraws
        if kc == 1
            draws(:,r) = carter_kohn_scalar(yc, Xc, sig2c, omega2c, 0);
        else
            draws(:,r) = reshape(carter_kohn(yc, Xc, sig2c, omega2c, zeros(kc,1)), [], 1);
        end
    end
    v = diag(inv(full(Kc)));                               % dense inverse, for this check only
    z = (mean(draws,2) - bhat)./sqrt(v/ndraws);
    vr = var(draws,0,2)./v;
    code = {'scalar', 'matrix'};
    fprintf(['   k = %d, %s code: largest |z| of the means %.2f, ' ...
        'variance ratios %.3f to %.3f\n'], kc, code{kc}, max(abs(z)), min(vr), max(vr));
end

% Then the timing: median of 10 runs after one warm-up run
nrep = 11;
fprintf('\n   ms per draw of the whole path, median of %d runs\n', nrep - 1);
fprintf('   %4s %6s %16s %12s\n', 'k', 'T', 'simulate_states', 'Carter-Kohn');
tscalar = zeros(1,2);
Ts = [1000 5000];
for kt = [1 2 5 10]
    for iT = 1:2
        Tt = Ts(iT);
        [yt, Xt, sig2t, omega2t] = generate_tvp(Tt, kt);
        b0 = zeros(kt,1);
        Ht = ssm.diffmat(Tt); HHt = Ht'*Ht;
        Zt = ssm.surform(Xt); ZZt = Zt'*Zt; Zyt = Zt'*yt;
        tp = zeros(nrep,1); tk = zeros(nrep,1); ts = zeros(nrep,1);
        for r = 1:nrep
            tic;                                           % as in a Gibbs sampler
            iOmega = sparse(1:kt,1:kt,1./omega2t);
            P = kron(HHt, iOmega);
            Kt = P + ZZt/sig2t;
            [~] = ssm.simulate_states(Kt, P*repmat(b0,Tt,1) + Zyt/sig2t);
            tp(r) = toc;
            tic;
            [~] = carter_kohn(yt, Xt, sig2t, omega2t, b0);
            tk(r) = toc;
            if kt == 1
                tic;
                [~] = carter_kohn_scalar(yt, Xt, sig2t, omega2t, 0);
                ts(r) = toc;
            end
        end
        fprintf('   %4d %6d %16.2f %12.2f\n', kt, Tt, 1e3*median(tp(2:end)), ...
            1e3*median(tk(2:end)));
        if kt == 1
            tscalar(iT) = 1e3*median(ts(2:end));
        end
    end
end
fprintf(['   k = 1, Carter-Kohn in scalar arithmetic: %.2f ms at T = 1000, ' ...
    '%.2f ms at T = 5000\n'], tscalar);

%% Figure: the coefficient paths of the Phillips curve
tid = 1960.25 + (0:T-1)'/4;                                % quarterly, 1960Q2-2019Q4
labels = {'intercept', 'output gap', 'lagged inflation'};
figure('Name', 'ex03 TVP Phillips curve');
for j = 1:k
    subplot(k,1,j); hold on
    hb = ssm.shaded_band(tid, Beta_q(:,j,1), Beta_q(:,j,2));
    hm = plot(tid, Beta_mean(:,j), 'k', 'LineWidth', 1.5);
    hold off; box off; xlim([tid(1) tid(end)])
    title(['coefficient on the ' labels{j}])
    if j == 1
        legend([hm hb], {'posterior mean', '90% band'}, 'Location', 'best'); legend boxoff
    end
end
drawnow

fprintf('\nex03 done.\n');

function [y, X, sig2, omega2] = generate_tvp(T, k)
% A TVP regression with an intercept and k-1 standard normal regressors, random-walk
% coefficients with innovation variance omega2 = 0.01 each, and sig2 = 0.5
sig2 = .5; omega2 = .01*ones(k,1);
X = [ones(T,1) randn(T,k-1)];
B = cumsum(sqrt(omega2').*randn(T,k), 1);
y = sum(X.*B, 2) + sqrt(sig2)*randn(T,1);
end

function [K, bhat] = tvp_posterior(y, X, sig2, omega2, beta0)
% Posterior precision and mean of the stacked path, given the parameters
T = size(X,1); k = size(X,2);
H = ssm.diffmat(T);
P = kron(H'*H, sparse(1:k,1:k,1./omega2));
Z = ssm.surform(X);
K = P + Z'*Z/sig2;
bhat = K\(P*repmat(beta0,T,1) + Z'*y/sig2);
end

function B = carter_kohn(y, X, sig2, omega2, beta0)
% One draw of (beta_1, ..., beta_T) by the Kalman filter and backward sampling of
% Carter and Kohn (1994), in k x k matrix operations; B is k x T
[T, k] = size(X);
Om = diag(omega2);
A = zeros(k,T); Pf = zeros(k,k,T);
a = beta0; P = Om;                                         % beta_1 ~ N(beta0, Omega)
for t = 1:T
    x = X(t,:)';
    f = x'*P*x + sig2;
    g = P*x/f;
    A(:,t) = a + g*(y(t) - x'*a);
    Pf(:,:,t) = P - g*(x'*P);
    a = A(:,t); P = Pf(:,:,t) + Om;
end
z = randn(k,T);
B = zeros(k,T);
B(:,T) = A(:,T) + chol(Pf(:,:,T),'lower')*z(:,T);
for t = T-1:-1:1
    Pt = Pf(:,:,t);
    J = Pt/(Pt + Om);
    m = A(:,t) + J*(B(:,t+1) - A(:,t));
    V = Pt - J*Pt; V = (V + V')/2;
    B(:,t) = m + chol(V,'lower')*z(:,t);
end
end

function b = carter_kohn_scalar(y, x, sig2, omega2, beta0)
% The same sampler for k = 1, in scalar arithmetic; x is the T x 1 regressor
T = numel(y);
a = zeros(T,1); p = zeros(T,1);
ap = beta0; pp = omega2;
for t = 1:T
    f = x(t)^2*pp + sig2;
    g = pp*x(t)/f;
    a(t) = ap + g*(y(t) - x(t)*ap);
    p(t) = pp - g*x(t)*pp;
    ap = a(t); pp = p(t) + omega2;
end
z = randn(T,1);
b = zeros(T,1);
b(T) = a(T) + sqrt(p(T))*z(T);
for t = T-1:-1:1
    j = p(t)/(p(t) + omega2);
    b(t) = a(t) + j*(b(t+1) - a(t)) + sqrt(p(t) - j*p(t))*z(t);
end
end
