%% ex08 - Stochastic volatility with MA(1) and Student-t errors, on silver returns
%
% Three models of daily returns y_t, each with mean zero and a stationary AR(1)
% log-volatility:
%
%   y_t = e_t + psi*e_{t-1},   e_t ~ N(0, lam_t*exp(h_t)),   e_0 = 0,
%   h_t = mu_h + phi_h*(h_{t-1} - mu_h) + u_t,   u_t ~ N(0, sigma2_h),
%
% with h_1 ~ N(mu_h, sigma2_h/(1 - phi_h^2)) and lam_t ~ IG(nu/2, nu/2). SV sets psi = 0
% and lam_t = 1, SV-MA sets lam_t = 1, and SV-MA-t keeps both, so its errors are
% Student-t with nu degrees of freedom. These are the models of Chan and Hsiao (2014)
% without their constant mean; SV-MA is the MA(1) case of the moving average stochastic
% volatility model of Chan (2013).
%
% Stacked over t, y = H_psi*e, with H_psi lower bidiagonal: ones on the diagonal and psi
% below it. So e = H_psi^{-1}*y is one call to filter, and the likelihood given h, lam
% and psi is exact.
%
% Each sweep of the sampler takes five steps; SV skips steps 3-5, and SV-MA steps 4-5.
%
%   1. h, by ssm.ksc_ar1_mean. Given psi and lam, log(e_t^2/lam_t) is h_t plus a
%      log chi^2_1 error, which the auxiliary mixture sampler of Kim, Shephard and Chib
%      (1998) approximates by a seven-component Gaussian mixture. The function draws the
%      component indicators and then h, by the precision sampler of Chan and Jeliazkov
%      (2009).
%   2. sigma2_h and mu_h from their full conditionals, and phi_h by an independence-chain
%      Metropolis-Hastings (MH) step.
%   3. psi by an independence-chain MH step with a normal proposal: its mean is the mode
%      of the conditional density and its variance the inverse of the negative Hessian
%      there, both from ssm.mode_newton.
%   4. lam_t from its inverse-gamma full conditional.
%   5. nu by an MH step of the same form as step 3.
%
% Steps 3-5 do not condition on the component indicators, so they come after h and
% before the next draw of the indicators, the order of Del Negro and Primiceri (2015);
% see Section 10.1.1 of the book Bayesian Macroeconometrics (Chan, forthcoming).
%
% On the daily silver returns of Chan and Hsiao (2014), January 2005 to December 2012,
% both extensions matter. Under SV-MA-t the 90% interval of psi is about (-0.12, -0.05),
% nu is about 7, and sigma2_h is about half its value under SV: with Student-t errors the
% largest returns come from the tails of e_t, so the volatility path is smoother. The
% estimates are close to those Chan and Hsiao (2014) report for SV-MA-t with a constant
% mean.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 10.1.1.
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with Application to
% Inflation Forecast, Journal of Econometrics, 176(2): 162-172.
% Chan, J.C.C. and Hsiao, C.Y.L. (2014). Estimation of Stochastic Volatility Models
% with Heavy Tails and Serial Dependence. In: I. Jeliazkov and X.-S. Yang (Eds.),
% Bayesian Inference in the Social Sciences, 155-176, John Wiley & Sons, Hoboken, New
% Jersey.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
% Del Negro, M. and Primiceri, G.E. (2015). Time Varying Structural Vector
% Autoregressions and Monetary Policy: A Corrigendum, Review of Economic Studies, 82(4):
% 1342-1345.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
% Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
% 361-393.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex08: stochastic volatility with MA(1) and Student-t errors ===\n');

rng(42);
y = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', 'silver.csv'));
T = length(y);
nsim = 5000; burnin = 1000;

% the priors of MASVt.m, in replications/chan_hsiao2014_wiley_sv
pri.muh0 = 0; pri.Vmuh = 5;                % mu_h ~ N(0, 5)
pri.phih0 = .95; pri.Vphih = 1;            % phi_h ~ N(0.95, 1) on (-1, 1)
pri.nuh = 10; pri.Sh = .02*(pri.nuh - 1);  % sigma2_h ~ IG(10, 0.18), mean 0.02
pri.Vpsi = 1;                              % psi ~ N(0, 1) on (-1, 1)
pri.nuub = 50;                             % nu ~ U(0, 50)

names = {'SV', 'SV-MA', 'SV-MA-t'};
res = cell(1,3);
for m = 1:3
    tic;
    res{m} = sampler(y, m >= 2, m == 3, nsim, burnin, pri);
    res{m}.time = toc;
end

%% Results
fprintf('\nSilver spot price, daily returns, T = %d, %d draws after %d burn-in for each model\n', ...
    T, nsim, burnin);
pnames = {'mu_h', 'phi_h', 'sigma2_h', 'psi', 'nu'};
for m = 1:3
    r = res{m};
    fprintf('\n%s (%.1f s)\n', names{m}, r.time);
    fprintf('   %-9s %9s   %s\n', '', 'mean', '90% interval');
    for j = find(r.used)
        q = quantile(r.theta(:,j), [.05 .95]);
        fprintf('   %-9s %9.3f   (%.3f, %.3f)\n', pnames{j}, mean(r.theta(:,j)), q);
    end
    fprintf('   acceptance: phi_h %.2f', r.acc(1));
    if r.used(4), fprintf(', psi %.2f', r.acc(2)); end
    if r.used(5), fprintf(', nu %.2f', r.acc(3)); end
    fprintf('\n');
end

%% Figure: the volatility under each model, and the posteriors of psi and nu
tid = linspace(2005, 2013, T)';
figure('Name', 'ex08 SV with MA(1) and t errors');
subplot(2,1,1); hold on
hb = ssm.shaded_band(tid, res{3}.volq(:,1), res{3}.volq(:,2));
h1 = plot(tid, res{1}.vol, 'Color', [.6 .6 .6], 'LineWidth', 1);
h2 = plot(tid, res{2}.vol, 'b', 'LineWidth', 1);
h3 = plot(tid, res{3}.vol, 'k', 'LineWidth', 1.2);
hold off; box off; xlim([tid(1) tid(end)])
title('silver returns: posterior mean of the standard deviation of e_t')
legend([h1 h2 h3 hb], {'SV', 'SV-MA', 'SV-MA-t', 'SV-MA-t 90% band'}, ...
    'Location', 'best'); legend boxoff
subplot(2,2,3); hold on
histogram(res{2}.theta(:,4), 50, 'Normalization', 'pdf', 'EdgeColor', 'none');
histogram(res{3}.theta(:,4), 50, 'Normalization', 'pdf', 'EdgeColor', 'none');
hold off; box off
title('posterior of \psi'); legend({'SV-MA', 'SV-MA-t'}, 'Location', 'best'); legend boxoff
subplot(2,2,4);
histogram(res{3}.theta(:,5), 50, 'Normalization', 'pdf', 'EdgeColor', 'none');
box off; title('posterior of \nu, SV-MA-t')
drawnow

fprintf('\nex08 done.\n');

function out = sampler(y, has_ma, has_t, nsim, burnin, pri)
% one run of the sampler; has_ma and has_t switch on psi and lam
T = length(y);
c = 1e-4;
sigh2 = .05; phih = .95; muh = 1;          % the start of MASVt.m
h = log(var(y)*.8)*ones(T,1);
psi = 0; nu = 5; lam = ones(T,1);
if has_t, lam = 1./gamrnd(nu/2, 2/nu, T, 1); end
if has_ma                                   % psi starts at its mode, as in MASVt.m
    psi = fminbnd(@(q) sum(filter(1, [1 q], y).^2./lam), -.99, .99);
end
psihat = psi; nuhat = nu;
theta = zeros(nsim, 5);                     % [mu_h phi_h sigma2_h psi nu]
volsum = zeros(T,1); voldraws = zeros(T, nsim/10);
acc = zeros(1,3);
for isim = 1:nsim + burnin
    % h, with the mixture indicators, given psi and lam
    e = filter(1, [1 psi], y);              % H_psi\y
    h = ssm.ksc_ar1_mean(log(e.^2./lam + c), h, muh, phih, sigh2);

    % sigma2_h, phi_h and mu_h given h
    ht = h - muh;
    Sh = pri.Sh + ((1 - phih^2)*ht(1)^2 + sum((ht(2:T) - phih*ht(1:T-1)).^2))/2;
    sigh2 = 1/gamrnd(pri.nuh + T/2, 1/Sh);
    Kphi = 1/pri.Vphih + (ht(1:T-1)'*ht(1:T-1))/sigh2;
    phihat = (pri.phih0/pri.Vphih + (ht(1:T-1)'*ht(2:T))/sigh2)/Kphi;
    phic = phihat + randn/sqrt(Kphi);
    g = @(x) .5*log(1 - x^2) - .5*(1 - x^2)/sigh2*ht(1)^2;
    if abs(phic) < .9999 && g(phic) - g(phih) > log(rand)
        phih = phic;
        if isim > burnin, acc(1) = acc(1) + 1; end
    end
    Kmu = 1/pri.Vmuh + ((T-1)*(1 - phih)^2 + (1 - phih^2))/sigh2;
    muhat = (pri.muh0/pri.Vmuh + (1 - phih^2)/sigh2*h(1) ...
        + (1 - phih)/sigh2*sum(h(2:T) - phih*h(1:T-1)))/Kmu;
    muh = muhat + randn/sqrt(Kmu);

    % psi given h and lam, from the exact likelihood
    if has_ma
        iv = exp(-h)./lam;
        f = @(q) -.5*(iv'*filter(1, [1 q], y).^2) - .5*q^2/pri.Vpsi;
        [psihat, Kpsi] = ssm.mode_newton(psihat, @(q) grad_psi(q, y, iv, pri.Vpsi));
        psic = psihat + randn/sqrt(Kpsi);
        if abs(psic) < 1 && f(psic) - f(psi) ...
                + .5*Kpsi*((psic - psihat)^2 - (psi - psihat)^2) > log(rand)
            psi = psic;
            if isim > burnin, acc(2) = acc(2) + 1; end
        end
    end

    % lam and nu given h and psi
    if has_t
        e = filter(1, [1 psi], y);
        lam = 1./gamrnd((nu + 1)/2, 1./(nu/2 + e.^2.*exp(-h)/2));
        s1 = sum(log(lam)); s2 = sum(1./lam);
        fnu = @(x) T*(x/2*log(x/2) - gammaln(x/2)) - (x/2 + 1)*s1 - x/2*s2;
        [nuhat, Knu] = ssm.mode_newton(nuhat, @(x) grad_nu(x, T, s1, s2));
        nuc = nuhat + randn/sqrt(Knu);
        if nuc > 0 && nuc < pri.nuub && fnu(nuc) - fnu(nu) ...
                + .5*Knu*((nuc - nuhat)^2 - (nu - nuhat)^2) > log(rand)
            nu = nuc;
            if isim > burnin, acc(3) = acc(3) + 1; end
        end
    end

    if isim > burnin
        i = isim - burnin;
        theta(i,:) = [muh phih sigh2 psi nu];
        sd = exp(h/2);                      % the standard deviation of e_t
        if has_t, sd = sd*sqrt(nu/max(nu - 2, 0)); end   % Inf when nu <= 2
        volsum = volsum + sd;
        if mod(i, 10) == 0, voldraws(:, i/10) = sd; end
    end
end
out.theta = theta;
out.used = [true true true has_ma has_t];
out.acc = acc/nsim;
out.vol = volsum/nsim;
out.volq = quantile(voldraws, [.05 .95], 2);
end

function [g, K] = grad_psi(q, y, iv, Vpsi)
% gradient and negative Hessian of the log conditional density of psi; the
% derivatives of e = H_psi\y follow the same recursion as e
e = filter(1, [1 q], y);
d1 = filter(1, [1 q], -[0; e(1:end-1)]);
d2 = filter(1, [1 q], -2*[0; d1(1:end-1)]);
g = -(iv'*(e.*d1)) - q/Vpsi;
K = iv'*(d1.^2 + e.*d2) + 1/Vpsi;
end

function [g, K] = grad_nu(x, T, s1, s2)
% gradient and negative Hessian of the log density of lam given nu, as in MASVt.m
g = T/2*(log(x/2) + 1 - psi(x/2)) - .5*(s1 + s2);
K = -(T/(2*x) - T/4*psi(1, x/2));
end
