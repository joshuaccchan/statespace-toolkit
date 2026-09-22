%% ex04 - The output gap: a local linear trend with an AR(2) cycle
%
% y_t = tau_t + c_t, where y_t is 100 times log US real GDP. The cycle, the output gap,
% is AR(2), c_t = phi_1*c_{t-1} + phi_2*c_{t-2} + u_t^c with u_t^c ~ N(0, sigc2) and
% c_0 = c_{-1} = 0. Trend growth is a random walk,
% tau_t - tau_{t-1} = tau_{t-1} - tau_{t-2} + u_t^tau with u_t^tau ~ N(0, sigtau2),
% and tau_0 and tau_{-1} are unknown. Given y and the parameters, the trend has the
% banded precision K = H2'*H2/sigtau2 + Hphi'*Hphi/sigc2, where
% H2 = ssm.lagpolymat(T, [2 -1]) is the second-difference matrix and
% Hphi = ssm.lagpolymat(T, phi), and ssm.simulate_states draws the whole path at once.
%
% The example estimates the output gap and trend growth of US real GDP, 1947Q1-2019Q4,
% with the sampler of the book's chapter09/UC_output_gap.m and the means and draws of tau,
% phi and (tau_0, tau_{-1}) by ssm.simulate_states. The AR coefficients
% are drawn from their Gaussian full conditional and kept when stationary, and sigtau2,
% under a uniform prior on (0, 0.01), by griddy Gibbs. The specification is similar to
% Grant and Chan (2017), with the AR(2) cycle of Morley, Nelson and Zivot (2003). For
% output gaps from published models, re-estimated every quarter, see
% trend-cycle-toolkit (github.com/joshuaccchan/trend-cycle-toolkit).
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.1.3.
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling Output Gaps: Unobserved Components
% Model and Hodrick-Prescott Filter, Journal of Economic Dynamics and Control, 75:
% 114-121.
% Morley, J.C., Nelson, C.R. and Zivot, E. (2003). Why Are the Beveridge-Nelson and
% Unobserved-Components Decompositions of GDP So Different?, Review of Economics and
% Statistics, 85(2): 235-243.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex04: the output gap, a local linear trend with an AR(2) cycle ===\n');

rng(42);
nsim = 20000; burnin = 1000;
data_raw = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', 'USGDP.csv'), ...
    'Range', 'B2:B293');                                   % 1947Q1-2019Q4
data = 100*log(data_raw);
y = data;
T = length(y);

% prior hyperparameters
a0 = [750;750]; B0 = 100*eye(2);                           % (tau_0, tau_{-1}) ~ N(a0, B0)
phi0 = [1.3 -.7]'; iVphi = speye(2);                       % phi ~ N(phi0, I), stationary
nu_sigc2 = 3; S_sigc2 = 1*(nu_sigc2-1);                    % sigc2 ~ IG(nu_sigc2, S_sigc2)
sigtau2_ub = .01;                                          % sigtau2 ~ U(0, sigtau2_ub)

% storage
store_theta = zeros(nsim,6);                               % [phi, sigc2, sigtau2, tau0]
store_tau = zeros(nsim,T);
store_mu = zeros(nsim,T);                                  % annualized trend growth

% initialize
phi = [1.34 -.7]';
tau0 = [y(1) y(1)]';                                       % [tau_{0}, tau_{-1}]
sigc2 = .5;
sigtau2 = .001;

% construct a few things
H2 = ssm.lagpolymat(T, [2 -1]);                            % second differences
H2H2 = H2'*H2;
Hphi = ssm.lagpolymat(T, phi);
Xtau0 = [(2:T+1)' -(1:T)'];
n_grid = 500;
count_phi = 0;

for isim = 1:nsim+burnin
    % sample tau
    alp_tau = H2\[2*tau0(1)-tau0(2);-tau0(1);sparse(T-2,1)];
    Ktau = H2H2/sigtau2 + Hphi'*Hphi/sigc2;
    tau = ssm.simulate_states(Ktau, H2H2*alp_tau/sigtau2 + Hphi'*Hphi*y/sigc2);

    % sample phi
    c = y-tau;
    Xphi = [[0;c(1:T-1)] [0;0;c(1:T-2)]];
    Kphi = iVphi + Xphi'*Xphi/sigc2;
    phic = ssm.simulate_states(Kphi, iVphi*phi0 + Xphi'*c/sigc2);
    if sum(phic) < .99 && phic(2) - phic(1) < .99 ...
            && phic(2) > -.99
        phi = phic;
        Hphi = ssm.lagpolymat(T, phi);
        count_phi = count_phi + 1;
    end

    % sample sigc2
    sigc2 = 1/gamrnd(nu_sigc2 + T/2,1/(S_sigc2 ...
        + (c-Xphi*phi)'*(c-Xphi*phi)/2));

    % sample sigtau2 via Griddy-Gibbs on (0, sigtau2_ub)
    del_tau = [tau0(1); tau] ...
        - [tau0(2); tau0(1); tau(1:end-1)];                % first differences of tau
    ddel_tau = del_tau(2:end) - del_tau(1:end-1);          % second differences of tau
    logf_sigtau2 = @(x) -(T/2)*log(x) ...
        - (ddel_tau'*ddel_tau)./(2*x);
    sigtau2 = griddy_gibbs(logf_sigtau2, 1e-12, ...
        sigtau2_ub, n_grid);

    % sample tau0
    Ktau0 = B0\speye(2) + Xtau0'*H2H2*Xtau0/sigtau2;
    tau0 = ssm.simulate_states(Ktau0, B0\a0 + Xtau0'*H2H2*tau/sigtau2);

    if isim > burnin
        i = isim-burnin;
        store_tau(i,:) = tau';
        store_theta(i,:) = [phi' sigc2 sigtau2 tau0'];
        store_mu(i,:) = 4*(tau-[tau0(1);tau(1:end-1)])';
    end
end
tau_mean = mean(store_tau)';
theta_mean = mean(store_theta)';
theta_CI = quantile(store_theta,[.025 .975]);
mu_mean = mean(store_mu)';
gap_q = quantile(y' - store_tau, [.05 .95])';              % output gap, 90% band
mu_q = quantile(store_mu, [.05 .95])';

names = {'phi_1', 'phi_2', 'sigc2', 'sigtau2', 'tau_0', 'tau_{-1}'};
fmt = {'%.3f', '%.3f', '%.3f', '%.5f', '%.2f', '%.2f'};
fprintf('\nUS real GDP, 1947Q1-2019Q4, T = %d, %d draws after %d burn-in\n', T, nsim, burnin);
fprintf('   posterior means and 95%% CIs\n');
for j = 1:numel(names)
    fprintf(['   %-9s = ' fmt{j} '  [' fmt{j} ', ' fmt{j} ']\n'], names{j}, ...
        theta_mean(j), theta_CI(:,j));
end
fprintf('   phi accepted in %.1f%% of sweeps\n', 100*count_phi/(nsim + burnin));
fprintf(['   2019Q4: output gap %.2f, 90%% band [%.2f, %.2f]; annualized trend growth ' ...
    '%.2f, 90%% band [%.2f, %.2f]\n'], ...
    y(T) - tau_mean(T), gap_q(T,:), mu_mean(T), mu_q(T,:));

%% Figures: the output gap and trend growth, NBER recessions shaded
tt = (1947:.25:2019.75)';
figure('Name', 'ex04 output gap');
subplot(2,1,1); hold on
yl = [min(gap_q(:,1))-1, max(gap_q(:,2))+1];
shade_nber_recessions(yl(1), yl(2));
hb = ssm.shaded_band(tt, gap_q(:,1), gap_q(:,2));
hm = plot(tt, y - tau_mean, 'k', 'LineWidth', 1.5);
plot(tt, zeros(T,1), '--k', 'LineWidth', 1);
hold off; box off; xlim([1947 2020]); ylim(yl)
title('output gap: posterior mean and 90% band')
legend([hm hb], {'posterior mean', '90% band'}, 'Location', 'best'); legend boxoff
subplot(2,1,2); hold on
yl = [min(mu_q(:,1))-.5, max(mu_q(:,2))+.5];
shade_nber_recessions(yl(1), yl(2));
ssm.shaded_band(tt, mu_q(:,1), mu_q(:,2));
plot(tt, mu_mean, 'k', 'LineWidth', 1.5);
hold off; box off; xlim([1947 2020]); ylim(yl)
title('trend output growth, annualized: posterior mean and 90% band')
drawnow

fprintf('\nex04 done.\n');

function [x_draw, x_grid, logf_grid] = griddy_gibbs(logf, a, b, n_grid)
% The book's griddy_gibbs.m: one draw from the Griddy-Gibbs approximation to a
% univariate target on (a, b), given its unnormalized log density logf, evaluated on
% a jittered grid of n_grid points

% jittered grid to avoid repeated draws
step = (b - a) / (n_grid + 1); % grid spacing
x_grid = a + step*(1:n_grid)';
x_grid = x_grid + (rand(n_grid,1) - 0.5)*step; % small jitter
    % enforce strict bounds
x_grid = min(max(x_grid, a + eps), b - eps);

% evaluate logf on the grid and normalize
logf_grid = logf(x_grid);
w = exp(logf_grid - max(logf_grid));
w = w / sum(w); % normalize to sum to 1

cdf_grid = cumsum(w);
u = rand;
idx = find(cdf_grid >= u, 1, 'first');
x_draw = x_grid(idx);
end

function shade_nber_recessions(ymin, ymax)
% The book's shade_nber_recessions.m without its closing uistack and set(ax,'Layer','top')
% lines: shades the NBER recessions from peak to trough on the current axes, whose x-axis
% is in decimal years

% NBER peak-to-trough dates (month precision)
% [peak_year peak_month  trough_year trough_month]
R = [ ...
    1948 11  1949 10;
    1953  7  1954  5;
    1957  8  1958  4;
    1960  4  1961  2;
    1969 12  1970 11;
    1973 11  1975  3;
    1980  1  1980  7;
    1981  7  1982 11;
    1990  7  1991  3;
    2001  3  2001 11;
    2007 12  2009  6;
    2020  2  2020  4];

dec = @(yy,mm) yy + (mm-1)/12;

ax = gca;
hold(ax,'on');

for i = 1:size(R,1)
    x1 = dec(R(i,1), R(i,2));
    x2 = dec(R(i,3), R(i,4));

    patch(ax, [x1 x2 x2 x1], [ymin ymin ymax ymax], [0.9 0.9 0.9], ...
        'EdgeColor','none', ...
        'HandleVisibility','off', ...
        'Clipping','on');
end
end
