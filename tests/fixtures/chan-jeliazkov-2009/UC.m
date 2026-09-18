% UC.m
% Local-level unobserved components model applied to US CPI inflation:
%
%     y_t   = tau_t + eps_t,        eps_t ~ N(0, sig2)
%     tau_t = tau_{t-1} + u_t,      u_t   ~ N(0, omega2)
%
% with priors
%
%     tau_0  ~ N(a0, b0)
%     sig2   ~ IG(nu_sig0, S_sig0)
%     omega2 ~ IG(nu_omega0, S_omega0)
%
% The latent trend (tau_1, ..., tau_T) is sampled in one block using the
% precision sampler of Chan and Jeliazkov (2009).

clear; clc; rng(42);
nsim = 20000;
burnin = 1000;
% load data - US CPI 1948M1 - 2019M12
data = readmatrix('USCPI.csv', 'Range', 'B2:B865');
y = data;
T = length(y);

    % initialize for storage
store_tau = zeros(nsim,T);
store_theta = zeros(nsim,3);  % [sig2,omega2,tau0]

    % prior
a0 = 5; b0 = 100;
nu_sig0 = 3; S_sig0 = 1*(nu_sig0-1);
nu_omega0 = 3; S_omega0 = .25^2*(nu_omega0-1);
% nu_omega0 = 3; S_omega0 = 1*(nu_omega0-1);

    % initialize the Markov chain
sig2 = 1; omega2 = .1; tau0 = 5;

    % compute a few things outside the loop
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
HHiota = HH*ones(T,1);

for isim = 1:nsim+burnin

        % sample tau
    Ktau = HH/omega2 + speye(T)/sig2;
    tau_hat = Ktau\(tau0/omega2*HHiota + y/sig2);
    Ctau = chol(Ktau,'lower');
    tau = tau_hat + Ctau'\randn(T,1);

        % sample sig2
    sig2 = 1/gamrnd(nu_sig0 + T/2,1/(S_sig0 + (y-tau)'*(y-tau)/2));

        % sample omega2
    omega2 = 1/gamrnd(nu_omega0 + T/2, ...
        1/(S_omega0 + (tau-tau0)'*HH*(tau-tau0)/2));

        % sample tau0
    Ktau0 = 1/b0 + 1/omega2;
    tau0_hat = Ktau0\(a0/b0 + tau(1)/omega2);
    tau0 = tau0_hat + sqrt(Ktau0)'\randn;

    if isim>burnin
        isave = isim-burnin;
        store_tau(isave,:) = tau';
        store_theta(isave,:) = [sig2 omega2 tau0];
    end
end

theta_hat = mean(store_theta);
theta_CI = quantile(store_theta,[.025 .975]);
tau_hat = mean(store_tau)';
tau_q = quantile(store_tau, [0.05 0.95], 1);   % 90% pointwise CI for tau
tau_lo = tau_q(1,:)';
tau_hi = tau_q(2,:)';

fprintf('Posterior means:        sig2 = %.2f,  omega2 = %.2f,  tau0 = %.2f\n', theta_hat);
fprintf('Posterior 95%% CI lower: sig2 = %.2f,  omega2 = %.2f,  tau0 = %.2f\n', theta_CI(1,:));
fprintf('Posterior 95%% CI upper: sig2 = %.2f,  omega2 = %.2f,  tau0 = %.2f\n', theta_CI(2,:));

% monthly date axis, 1948M1 - 2019M12
tid = (1948 + (0:T-1)'/12);

figure;
hold on
    h_ci = fill([tid; flipud(tid)], [tau_lo; flipud(tau_hi)], ...
                [0.85 0.85 0.85], 'EdgeColor', 'none');
    h_y  = plot(tid, y, 'k--', 'LineWidth', 0.8);
    h_tr = plot(tid, tau_hat, 'k', 'LineWidth', 1.8);
hold off
xlim([tid(1) tid(end)]); box off;
set(gca, 'Layer', 'top');
legend([h_y, h_tr, h_ci], ...
       {'CPI inflation', 'Posterior mean trend $\widehat{\tau}$', '90\% CI for $\tau$'}, ...
       'Interpreter', 'latex', 'Location', 'best', 'Box', 'off');
set(gcf,'Position',[100 100 800 400]);

