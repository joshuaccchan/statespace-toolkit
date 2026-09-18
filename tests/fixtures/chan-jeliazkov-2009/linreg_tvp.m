% linreg_tvp.m
% Time-varying parameter regression of US PCE inflation on the output gap
% and lagged inflation (a TVP Phillips curve):
%
%     y_t    = x_t' * beta_t + eps_t,     eps_t ~ N(0, sig2)
%     beta_t = beta_{t-1} + u_t,          u_t   ~ N(0, Omega)
%
% where x_t = (1, gap_t, y_{t-1})' and Omega = diag(omega_1^2, ..., omega_k^2),
% with priors
%
%     beta_0    ~ N(beta00, Vbeta0)
%     sig2      ~ IG(nu_sig, S_sig)
%     omega_j^2 ~ IG(nu_om,  S_om(j)),   j = 1, ..., k
%
% The full path of coefficients (beta_1, ..., beta_T) is sampled in one
% block using the precision sampler of Chan and Jeliazkov (2009).

clear; clc; rng(42);
nsim = 20000; burnin = 1000;

% load data
data = readmatrix('USPCE_OutputGap.csv', ...
    'Range', 'B2:C241');
infl = data(:,1); % PCE inflation
gap  = data(:,2); % output gap

% construct y and X
y = infl(2:end);
g = gap(2:end);
ylag = infl(1:end-1);  % y_{t-1}
T = length(y);
X = [ones(T,1), g, ylag];
k = size(X,2);

% prior hyperparameters
beta00 = zeros(k,1);
iVbeta0 = 1/100*eye(k);
nu_sig = 3;  S_sig = 1;  % IG prior for sigma^2
nu_om = 3;               % IG prior for omega_j^2
S_om  = [0.125; 0.025; 0.025].^2*(nu_om-1);

% initialize chain
beta0 = zeros(k,1);
beta_ols = (X'*X)\(X'*y);
sig2 = mean((y - X*beta_ols).^2);
omega2 = 0.01^2 * ones(k,1);

% precompute a few things
S1 = sparse(2:T,1:T-1,1,T,T);
H  = speye(T) - S1;
HH = H'*H;
Z = SURform(X);
ZZ = Z'*Z;
Zy = Z'*y;

% storage
store_beta = zeros(nsim, T*k);
    % [beta0', sig2, omega2']
store_theta = zeros(nsim, 2*k + 1);

for isim = 1:nsim + burnin
    % sample beta
    iOmega = sparse(1:k,1:k,1./omega2);
    P = kron(HH, iOmega); % prior precision
    Kbeta = P + ZZ/sig2;
    CKbeta = chol(Kbeta, 'lower');
    beta_hat = Kbeta\(P*repmat(beta0,T,1) + Zy/sig2);
    beta = beta_hat + (CKbeta')\randn(k*T,1);

    % sample sigma^2
    e = y - Z*beta;
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + e'*e/2));

    % sample omega_j^2
    Beta = reshape(beta,k,T)';
    SSE = sum((Beta - [beta0'; Beta(1:T-1,:)]).^2)';
    omega2 = 1./gamrnd(nu_om + T/2, 1./(S_om + 0.5*SSE));

    % sample beta0
    Kbeta0 = iVbeta0 + sparse(1:k,1:k,1./omega2);
    beta0_hat = Kbeta0\(iVbeta0*beta00 ...
        + beta(1:k)./omega2);
    Cbeta0 = chol(Kbeta0,'lower');
    beta0 = beta0_hat + (Cbeta0)'\randn(k,1);

    if isim > burnin
        isave = isim - burnin;
        store_beta(isave,:) = beta';
        store_theta(isave,:) = [beta0', sig2, omega2'];
    end
end
Beta_mean = reshape(mean(store_beta, 1),k,T)';
Beta_q = permute(reshape(quantile(store_beta, [0.05,0.95], 1)',k,T,2),[2,1,3]);

% posterior summary for time-invariant parameters
theta_hat = mean(store_theta);
theta_q   = quantile(store_theta, [0.05 0.95]);
fprintf('Posterior means and 90%% CIs:\n');
fprintf('  beta0(intercept)    = %.4f  [%.4f, %.4f]\n', theta_hat(1), theta_q(1,1), theta_q(2,1));
fprintf('  beta0(output gap)   = %.4f  [%.4f, %.4f]\n', theta_hat(2), theta_q(1,2), theta_q(2,2));
fprintf('  beta0(lagged infl)  = %.4f  [%.4f, %.4f]\n', theta_hat(3), theta_q(1,3), theta_q(2,3));
fprintf('  sig2                = %.4f  [%.4f, %.4f]\n', theta_hat(4), theta_q(1,4), theta_q(2,4));
fprintf('  omega2(intercept)   = %.4f  [%.4f, %.4f]\n', theta_hat(5), theta_q(1,5), theta_q(2,5));
fprintf('  omega2(output gap)  = %.4f  [%.4f, %.4f]\n', theta_hat(6), theta_q(1,6), theta_q(2,6));
fprintf('  omega2(lagged infl) = %.4f  [%.4f, %.4f]\n', theta_hat(7), theta_q(1,7), theta_q(2,7));

% quarterly date axis, 1960Q2 - 2019Q4
tid = 1960.25 + (0:T-1)'/4;

names = {'Intercept', 'Output gap', 'Lagged inflation'};

fig = figure('Color','w');
for j = 1:k
    subplot(k,1,j); hold on;
    lo = Beta_q(:,j,1); hi = Beta_q(:,j,2);
    fill([tid; flipud(tid)], [lo; flipud(hi)], [0.85 0.85 0.85], ...
         'EdgeColor','none', 'HandleVisibility','off');

    plot(tid, Beta_mean(:,j), 'k', 'LineWidth', 1.5);

    box off;
    set(gca,'FontSize',12,'Layer','top');
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    ylabel(names{j});
    if j == k
        xlabel('Time');
    end
end
set(gcf,'Position',[100 100 600 500]);
