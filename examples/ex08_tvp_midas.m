%% ex08 - TVP-MIDAS on generated data: time-varying weights under a linear restriction
%
% A MIDAS regression relates a low-frequency variable to a high-frequency predictor
% through a weighting function. Chan, Poon and Zhu (2026) let the weights and the
% coefficients vary over time and parameterize the weighting function linearly, which
% keeps the model a linear Gaussian state space model:
%
%   y_t = alpha_t + beta_t*theta_t'*V*x_t + e_t,   e_t ~ N(0, exp(g_t)),
%
% where x_t collects the K+1 high-frequency observations falling in period t and
% V = [v_0, ..., v_K] holds the basis functions, so the weight on lag k is
% B(k; theta_t) = theta_t'*v_k. Here v_k is the Fourier basis [1, cos(2*pi*k/m),
% sin(2*pi*k/m)]'; the Almon lag polynomial [1, k, k^2]', the paper's other basis, is
% far more collinear. The coefficients b_t = [alpha_t, beta_t]' and theta_t follow
% random walks with diagonal covariances Omega and Xi, and the log-volatility g_t a
% random walk with variance sigma2_g.
%
% Given theta the model is linear in b, and given b it is linear in theta, so each block
% is drawn in one call to ssm.simulate_states from a banded precision. Separating beta_t
% from theta_t needs a normalization, and the paper imposes the T restrictions
% theta_t'*V*1 = 1, which make the draw of theta Gaussian truncated to a hyperplane.
% Updating an unconstrained draw imposes them,
%
%   theta = thetatilde + K^{-1}*M'*(M*K^{-1}*M')^{-1}*(1 - M*thetatilde),
%
% with M = I_T kron (1'*V'), reusing the Cholesky factor ssm.simulate_states returns, as
% ex05 does for the mixed-frequency aggregation. Under the Fourier basis the cosine and
% sine terms sum to zero over a full period, so the restriction fixes the first
% coefficient at 1/m.
%
% Section 1 checks the two conditional draws against dense algebra: the posterior mean of
% b, and the restriction, the mean and the covariance of the draws of theta.
%
% Section 2 checks that the whole sampler recovers the paths and variances that generated
% the data, on 200 periods generated from the model itself with the innovation variances
% set at their prior means. At five to ten times the prior means the variances
% come back at about a third of their true values and the paths are oversmoothed, since
% at T = 200, against an error variance of about one, they are weakly identified. The
% Monte Carlo experiments of the paper generate the data from beta density and exponential
% Almon weighting functions, which are nonlinear in their parameters, to measure how well
% the linear parameterization approximates them.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Sections 9.3 and 10.1.1.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
% Chan, J.C.C., Poon, A. and Zhu, D. (2026). Time-Varying Parameter MIDAS Models:
% Application to Nowcasting US Real GDP, Journal of Econometrics, forthcoming.
% Ghysels, E., Sinko, A. and Valkanov, R. (2007). MIDAS Regressions: Further Results
% and New Directions, Econometric Reviews, 26(1): 53-90.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
% Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
% 361-393.
% Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
% Chapman & Hall/CRC, Algorithm 2.6.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex08: TVP-MIDAS on generated data ===\n');

rng(42);
m = 12; K = m - 1; pb = 2;                       % high-frequency observations per period
V = [ones(1,K+1); cos(2*pi*(0:K)/m); sin(2*pi*(0:K)/m)];
q = size(V,1);
vsum = V*ones(K+1,1);                            % the restriction is theta_t'*vsum = 1
nuom = 5; Som = .004; nuxi = 10; Sxi = .001; nug = 5; Sg = .04;   % prior, as in the paper
Vb = 10; Vth = 10;

%% Section 1: the conditional draws against dense algebra
T = 30;
[~, VX] = midas_regressors(T, m, K, V);
H1 = kron(ssm.diffmat(T), speye(pb));
H2 = kron(ssm.diffmat(T), speye(q));
M = kron(speye(T), vsum'); Mt = full(M');
g = .2*randn(T,1); iSig = sparse(1:T, 1:T, exp(-g));
om2 = [1e-3; 1e-3]; xi2 = [1e-4; 2e-4; 3e-4];
Th = repmat([1/m 0 0], T, 1) + [zeros(T,1), .02*randn(T,2)];
beta = 1 + .1*randn(T,1);
y = randn(T,1);

X1 = ssm.surform([ones(T,1), sum(Th.*VX, 2)]);
S1 = sparse(1:T*pb, 1:T*pb, [1/Vb*ones(1,pb), repmat(1./om2', 1, T-1)]);
Kb = H1'*S1*H1 + X1'*iSig*X1;  cb = X1'*iSig*y;
[~, bhat] = ssm.simulate_states(Kb, cb);

X2 = ssm.surform(beta.*VX);
S2 = sparse(1:T*q, 1:T*q, [1/Vth*ones(1,q), repmat(1./xi2', 1, T-1)]);
Kth = H2'*S2*H2 + X2'*iSig*X2;  cth = X2'*iSig*y;

ndraws = 20000;
[~, ~, C] = ssm.simulate_states(Kth, cth);
U = C'\(C\Mt);  MU = M*U;
D = zeros(T*q, ndraws);
for i = 1:ndraws
    tht = ssm.simulate_states(Kth, cth);
    D(:,i) = tht + U*(MU\(ones(T,1) - M*tht));
end

Sig = inv(full(Kth));  mu = Sig*cth;             % the dense truncated-Gaussian moments
A = Sig*M';  Gm = M*A;
mu_c = mu + A*(Gm\(ones(T,1) - M*mu));
Sig_c = Sig - A*(Gm\A');

% the restriction leaves the first coefficient no freedom, so its conditional variance is
% zero and the comparison of means is made over the rest
dv = diag(Sig_c);
free = dv > 1e-12*max(dv);
fprintf('\n-- Section 1: the conditional draws, T = %d, %d draws\n', T, ndraws);
fprintf('b: posterior mean against K\\c                  %.2e\n', ...
    norm(bhat - full(Kb)\full(cb), inf));
fprintf('theta: restriction residual, every draw        %.2e\n', max(max(abs(M*D - 1))));
fprintf('theta: the pinned coefficient against 1/m      %.2e\n', ...
    max(max(abs(D(1:q:end,:) - 1/m))));
fprintf('theta: mean against the dense mean             %.2f Monte Carlo standard errors\n', ...
    max(abs(mean(D(free,:),2) - mu_c(free))./sqrt(dv(free)/ndraws)));
fprintf('theta: covariance against the dense covariance %.1f%% of its largest entry\n', ...
    100*max(max(abs(cov(D') - Sig_c)))/max(max(abs(Sig_c))));

%% Section 2: the whole sampler on generated data
rng(42);
nsim = 5000; burnin = 1000;
T = 200;
om2_true = [1e-3; 1e-3]; xi2_true = Sxi/(nuxi-1); sig2g_true = Sg/(nug-1);
[~, VX] = midas_regressors(T, m, K, V);

B_true = zeros(T, pb); Th_true = zeros(T, q); g_true = zeros(T, 1);
b = [0; 1];
% the initial weights th'*V make one cycle over the lags
th = [1; 2; -3]; th = th/(th'*vsum);             % scaled to satisfy the restriction
Pv = eye(q) - (vsum*vsum')/(vsum'*vsum);         % keeps the walk on the hyperplane
gt = 0;
for t = 1:T
    b = b + sqrt(om2_true).*randn(pb,1);
    th = th + Pv*(sqrt(xi2_true)*randn(q,1));
    gt = gt + sqrt(sig2g_true)*randn;
    B_true(t,:) = b'; Th_true(t,:) = th'; g_true(t) = gt;
end
y = B_true(:,1) + B_true(:,2).*sum(Th_true.*VX, 2) + exp(g_true/2).*randn(T,1);
W_true = Th_true*V;                              % row t: the weights B(0:K; theta_t)

H1 = kron(ssm.diffmat(T), speye(pb));
H2 = kron(ssm.diffmat(T), speye(q));
M = kron(speye(T), vsum'); Mt = full(M');
om2 = om2_true; xi2 = xi2_true*ones(q,1); sig2g = sig2g_true;
Th = repmat(Th_true(1,:), T, 1); Bc = zeros(T, pb); Bc(:,2) = 1; g = zeros(T,1);
store_B = zeros(nsim, T, pb);
store_W = zeros(nsim, T, K+1);
store_g = zeros(nsim, T);
store_v = zeros(nsim, 4);
store_res = zeros(nsim, 1);

fprintf('\n-- Section 2: %d periods, %d high-frequency observations each, %d draws\n', ...
    T, m, nsim);
start_time = tic;
for loop = 1:nsim + burnin
    iSig = sparse(1:T, 1:T, exp(-g));

    % the coefficients, given the weights
    X1 = ssm.surform([ones(T,1), sum(Th.*VX, 2)]);
    S1 = sparse(1:T*pb, 1:T*pb, [1/Vb*ones(1,pb), repmat(1./om2', 1, T-1)]);
    Kb = H1'*S1*H1 + X1'*iSig*X1;
    Bc = reshape(ssm.simulate_states(Kb, X1'*iSig*y), pb, T)';

    % the weights, given the coefficients, on the hyperplane theta_t'*V*1 = 1
    X2 = ssm.surform(Bc(:,2).*VX);
    S2 = sparse(1:T*q, 1:T*q, [1/Vth*ones(1,q), repmat(1./xi2', 1, T-1)]);
    Kth = H2'*S2*H2 + X2'*iSig*X2;
    [thtilde, ~, C] = ssm.simulate_states(Kth, X2'*iSig*(y - Bc(:,1)));
    U = C'\(C\Mt);
    theta = thtilde + U*((M*U)\(ones(T,1) - M*thtilde));
    Th = reshape(theta, q, T)';

    % the log-volatility
    err = y - Bc(:,1) - Bc(:,2).*sum(Th.*VX, 2);
    g = ssm.ksc_rw_h0(log(err.^2 + 1e-4), g, sig2g, 0);

    % the state innovation variances
    dB = diff(Bc); dTh = diff(Th); dg = [g(1); diff(g)];
    om2 = 1 ./ gamrnd(nuom + (T-1)/2, 1 ./ (Som + sum(dB.^2)'/2));
    xi2 = 1 ./ gamrnd(nuxi + (T-1)/2, 1 ./ (Sxi + sum(dTh.^2)'/2));
    sig2g = 1 / gamrnd(nug + T/2, 1 / (Sg + sum(dg.^2)/2));

    if loop > burnin
        store_B(loop-burnin,:,:) = Bc;
        store_W(loop-burnin,:,:) = Th*V;
        store_g(loop-burnin,:) = g';
        store_v(loop-burnin,:) = [om2', xi2(2), sig2g];
        store_res(loop-burnin) = norm(M*theta - ones(T,1), inf);
    end
end
fprintf('sampling takes %.1f seconds\n', toc(start_time));
fprintf('restriction residual, every draw: max abs value %.2e\n', max(store_res));

fprintf('\n%-28s %9s %9s %9s\n', 'quantity', 'RMSE', 'post sd', 'coverage');
report('alpha_t', store_B(:,:,1), B_true(:,1));
report('beta_t', store_B(:,:,2), B_true(:,2));
report('weights B(k; theta_t)', reshape(store_W, nsim, []), W_true(:));
report('volatility exp(g_t/2)', exp(store_g/2), exp(g_true/2));

fprintf('\n%-28s %9s %9s\n', 'innovation variance', 'estimate', 'true');
vn = {'omega2_alpha', 'omega2_beta', 'xi2 (cosine term)', 'sigma2_g'};
vt = [om2_true', xi2_true, sig2g_true];
for j = 1:4
    fprintf('%-28s %9.5f %9.5f\n', vn{j}, mean(store_v(:,j)), vt(j));
end

% the weighting function averaged over the sample. Two free coefficients drive all K+1
% weights, so at any single date the errors across lags move together; the coverage above
% is over every date and lag
Wbar = squeeze(mean(store_W, 2));
Wband = quantile(Wbar, [.05 .95], 1)';
fprintf('\nweighting function averaged over the sample, by lag\n');
fprintf('%6s %10s %18s %10s\n', 'lag', 'estimate', '90% band', 'true');
for k = [0 3 6 9 K]
    fprintf('%6d %10.3f %9.3f %8.3f %10.3f\n', k, mean(Wbar(:,k+1)), ...
        Wband(k+1,1), Wband(k+1,2), mean(W_true(:,k+1)));
end
fprintf('%6s %10.3f %27.3f\n', 'sum', sum(mean(Wbar,1)), mean(sum(W_true,2)));

function [X, VX] = midas_regressors(T, m, K, V)
% the K+1 high-frequency observations of each low-frequency period, from an AR(1) as in
% the Monte Carlo design of the paper, and their projection on the basis
xhf = filter(1, [1 -.7], randn(T*m + K, 1));
X = zeros(T, K+1);
for t = 1:T
    X(t,:) = xhf(K + t*m - (0:K))';
end
VX = X*V';
end

function report(name, draws, truth)
% RMSE of the posterior mean, the average posterior standard deviation, and the share of
% true values inside the 90 percent bands
mhat = mean(draws, 1)';
band = quantile(draws, [.05 .95], 1)';
fprintf('%-28s %9.3f %9.3f %8.1f%%\n', name, sqrt(mean((mhat - truth).^2)), ...
    mean(std(draws, 0, 1)), 100*mean(truth >= band(:,1) & truth <= band(:,2)));
end
