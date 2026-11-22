%% guide - the code of guides/tvp_states/README.md: the coefficients of a time-varying parameter
% regression, drawn by the precision sampler given the parameters. The sections called "the code
% to adapt" are all a sampler needs. Saves fig_spy.png, fig_coef.png and fig_mean.png next to this
% file.

here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)), 'setup.m'));
rng(1, 'twister');
rel = @(a, b) max(abs(a(:) - b(:)))/max(abs(b(:)));

%% 1. the TVP regression: the code to adapt
T = 200; k = 2; sig2 = 1;                        % the parameters, known
Omega = diag([.005 .02]); beta0 = [2; 1];
X = [ones(T,1) randn(T,1)];                      % x_t = (1, x_t)'
beta = beta0' + cumsum(randn(T,k)*sqrt(Omega));  % beta_t = beta_{t-1} + eta_t (T x k)
y = sum(X.*beta, 2) + sqrt(sig2)*randn(T,1);     % y_t = x_t'*beta_t + eps_t

Z = ssm.surform(X);                  % Z = diag(x_1', ..., x_T')
H = kron(ssm.diffmat(T), speye(k));  % H = H_T kron I_k
iS = kron(speye(T), inv(Omega));     % S^{-1} = I_T kron Omega^{-1}
btil = [beta0; zeros((T-1)*k,1)];    % betatilde_0 = (beta_0', 0', ..., 0')'
K = H'*iS*H + Z'*Z/sig2;             % K = H'*S^{-1}*H + Z'*Z/sig2
c = H'*iS*btil + Z'*y/sig2;          % c = H'*S^{-1}*betatilde_0 + Z'*y/sig2
[draws, betahat] = ssm.simulate_states(K, c, 5000);  % 5,000 independent draws of the path
B = reshape(draws, k, T, []);                        % B(i,t,d): coefficient i, period t, draw d

%% 1. the checks
fprintf('\nthe TVP regression, T = %d, k = %d\n', T, k);
fprintf('K is %d x %d with lower bandwidth %d\n', size(K), bandwidth(K, 'lower'));
fprintf('the top left corner of K:\n'); disp(full(K(1:4,1:4)));
Hd = full(H); Zd = full(Z);
V = Hd\(full(kron(speye(T), Omega))/Hd');  % the prior covariance of the path
mu0 = repmat(beta0, T, 1);                 % its prior mean
G = Zd*V*Zd' + sig2*eye(T);
mu = mu0 + V*Zd'*(G\(y - Zd*mu0));         % the covariance form
Vpost = V - V*Zd'*(G\(Zd*V));
fprintf('mean and covariance against dense algebra:   %.1e, %.1e (relative)\n', ...
    rel(betahat, mu), rel(inv(full(K)), Vpost));
fprintf('5,000 draws: means within %.2f posterior sd, variances %.3f to %.3f of the posterior''s\n', ...
    max(abs(mean(draws, 2) - mu)./sqrt(diag(Vpost))), min(var(draws, 0, 2)./diag(Vpost)), ...
    max(var(draws, 0, 2)./diag(Vpost)));
q1 = quantile(B, [.05 .95], 3);            % k x T x 2
in1 = beta' >= q1(:,:,1) & beta' <= q1(:,:,2);
fprintf(['this realization: the 90%% intervals contain the true intercept in %d and the true ' ...
    'slope in %d of %d periods\n'], nnz(in1(1,:)), nnz(in1(2,:)), T);
fprintf('average width of the 90%% intervals: intercept %.2f, slope %.2f\n', ...
    mean(q1(1,:,2) - q1(1,:,1)), mean(q1(2,:,2) - q1(2,:,1)));
beta1 = beta; m1 = mean(B, 3);
Tspy = 25;                                 % K for T = 25, for Figure 1
Zs = ssm.surform(X(1:Tspy,:)); Hs = kron(ssm.diffmat(Tspy), speye(k));
Kspy = Hs'*kron(speye(Tspy), inv(Omega))*Hs + Zs'*Zs/sig2;

%% 2. coefficients that revert to a mean: the code to adapt
Phi = diag([.9 .9]); mu = beta0;  % beta_t - mu = Phi*(beta_{t-1} - mu) + eta_t
beta = zeros(T,k); b = beta0;     % beta_0 = mu
for t = 1:T
    b = mu + Phi*(b - mu) + sqrt(Omega)*randn(k,1);
    beta(t,:) = b';
end
y = sum(X.*beta, 2) + sqrt(sig2)*randn(T,1);

L = speye(T) - ssm.diffmat(T);     % the lag matrix: ones below the diagonal
HPhi = speye(T*k) - kron(L, Phi);  % H_Phi = I - L kron Phi
btil = [(eye(k) - Phi)*mu + Phi*beta0; repmat((eye(k) - Phi)*mu, T-1, 1)];
K = HPhi'*iS*HPhi + Z'*Z/sig2;     % K = H_Phi'*S^{-1}*H_Phi + Z'*Z/sig2
c = HPhi'*iS*btil + Z'*y/sig2;     % c = H_Phi'*S^{-1}*betatilde + Z'*y/sig2
[draws, betahat] = ssm.simulate_states(K, c, 5000);
B = reshape(draws, k, T, []);

%% 2. the checks
fprintf('\ncoefficients that revert to a mean, Phi = diag(%.1f, %.1f); K has lower bandwidth %d\n', ...
    diag(Phi), bandwidth(K, 'lower'));
Hd = full(HPhi);
V = Hd\(full(kron(speye(T), Omega))/Hd');
mu0 = Hd\btil;  % the prior mean of the path
G = Zd*V*Zd' + sig2*eye(T);
mpost = mu0 + V*Zd'*(G\(y - Zd*mu0));
Vpost = V - V*Zd'*(G\(Zd*V));
fprintf('mean and covariance against dense algebra:   %.1e, %.1e (relative)\n', ...
    rel(betahat, mpost), rel(inv(full(K)), Vpost));
q2 = quantile(B, [.05 .95], 3);
in2 = beta' >= q2(:,:,1) & beta' <= q2(:,:,2);
fprintf(['this realization: the 90%% intervals contain the true intercept in %d and the true ' ...
    'slope in %d of %d periods\n'], nnz(in2(1,:)), nnz(in2(2,:)), T);
fprintf('average width of the 90%% intervals: intercept %.2f, slope %.2f\n', ...
    mean(q2(1,:,2) - q2(1,:,1)), mean(q2(2,:,2) - q2(2,:,1)));
beta2 = beta; m2 = mean(B, 3);

%% 3. a prior of their own for the first coefficients: the adaptation
a = zeros(k,1); V1 = 10*eye(k);                % beta_1 ~ N(a, V1), beta_0 not needed
iS1 = blkdiag(sparse(inv(V1)), kron(speye(T-1), sparse(inv(Omega))));
btil1 = [a; zeros((T-1)*k,1)];
y = sum(X.*beta1, 2) + sqrt(sig2)*randn(T,1);  % the random-walk coefficients of Section 1
K = H'*iS1*H + Z'*Z/sig2;
c = H'*iS1*btil1 + Z'*y/sig2;
[~, betahat] = ssm.simulate_states(K, c, 1);
Hd = full(H);
V = Hd\(full(blkdiag(V1, kron(eye(T-1), Omega)))/Hd');
mu0 = Hd\btil1;
G = Zd*V*Zd' + sig2*eye(T);
fprintf('\na prior of their own for the first coefficients: mean against dense algebra %.1e\n', ...
    rel(betahat, mu0 + V*Zd'*(G\(y - Zd*mu0))));

%% the figures
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 16 8]);
subplot(1,2,1); spy(Kspy, 'k', 4); title(sprintf('K, T = %d, k = %d', Tspy, k), 'FontWeight', 'normal'); xlabel('');
subplot(1,2,2); spy(sparse(abs(inv(full(Kspy))) > 1e-12), 'k', 4);
title(sprintf('K^{-1}, T = %d, k = %d', Tspy, k), 'FontWeight', 'normal'); xlabel('');
exportgraphics(fig, fullfile(here, 'fig_spy.png'), 'Resolution', 150); close(fig);
figs = {'fig_coef.png', 'fig_mean.png'}; TB = {beta1, beta2}; M = {m1, m2}; Q = {q1, q2};
names = {'intercept, \beta_{1t}', 'slope, \beta_{2t}'};
for f = 1:2
    fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 11]);
    for i = 1:k
        subplot(k, 1, i); hold on
        hb = ssm.shaded_band(1:T, Q{f}(i,:,1), Q{f}(i,:,2));
        ht = plot(1:T, TB{f}(:,i), 'k--', 'LineWidth', 1);
        hm = plot(1:T, M{f}(i,:), 'k', 'LineWidth', 1.3);
        hold off; box off; xlim([1 T]); title(names{i}, 'FontWeight', 'normal');
        if i == 1
            legend([ht hm hb], {'true coefficient', 'posterior mean', '90% pointwise credible interval'}, ...
                'Location', 'best', 'NumColumns', 3); legend boxoff
        end
    end
    exportgraphics(fig, fullfile(here, figs{f}), 'Resolution', 150); close(fig);
end
