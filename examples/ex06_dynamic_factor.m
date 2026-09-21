%% ex06 - A dynamic factor model: a business-cycle indicator from FRED-MD
%
% y_it = a_i*f_t + eps_it, eps_it ~ N(0, sig2_i), i = 1, ..., n, with one factor
% f_t = phi*f_{t-1} + u_t, u_t ~ N(0, omega2), f_0 = 0, and the loading a_1 of the first
% series fixed at 1. Given the loadings and variances, the factor path
% f = (f_1, ..., f_T)' has a tridiagonal precision, and ssm.simulate_states draws it at
% once: the precision sampler of Chan and Jeliazkov (2009). Given the factor, the
% loadings are independent, so one call to ssm.simulate_states with a diagonal
% precision draws them all.
%
% FRED-MD, 92 monthly US series from March 1959 to November 2025, each transformed as
% McCracken and Ng (2016) recommend and standardized, with industrial production first.
% The sampler is that of DFM.m in the chan-jeliazkov-2009 repository and of the book's
% chapter11/DFM.m, written here for one factor. The chan-jeliazkov-2009 copy reads each
% date one month late; ex06 reads the dates as the book does. The book reports a
% posterior mean of phi of about 0.27, with a 95% credible interval of (0.20, 0.33),
% and a factor near -16.5 in April 2020.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 11.2.3.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
% McCracken, M.W. and Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic
% Research, Journal of Business and Economic Statistics, 34(4): 574-589.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex06: a dynamic factor model, a business-cycle indicator from FRED-MD ===\n');

rng(42);
nsim = 20000;
burnin = 1000;

% load data; the first column holds dates as year + month/12
raw = readtable(fullfile(fileparts(mfilename('fullpath')), 'data', 'FRED-MD.csv'), ...
    'VariableNamingRule', 'preserve');
month_idx = round(12*raw{:,1});
tid = (month_idx - 1)/12;                                  % decimal years, January = .0
data = raw{:,2:end};
varnames = raw.Properties.VariableNames(2:end);

% move the 6th column (INDPRO) to the first column
perm = [6, 1:5, 7:size(data,2)];
data = data(:,perm);
varnames = varnames(perm);

% remove columns with missing values
idx = all(~isnan(data),1);
data = data(:,idx);
varnames = varnames(idx);

% standardize the data
Y = (data - mean(data,1)) ./ std(data,0,1);
[T,n] = size(Y);

% prior hyperparameters
a0 = 0; Va = 1;                                            % a_i ~ N(a0, Va), i > 1
phi0 = 0; Vphi = 1;                                        % phi ~ N(phi0, Vphi) on (-1, 1)
nusig2 = 3; Ssig2 = (nusig2-1)*ones(n,1);                  % sig2_i ~ IG(nusig2, Ssig2_i)
nuomega2 = 3; Somega2 = nuomega2 - 1;                      % omega2 ~ IG(nuomega2, Somega2)

% initialize the Markov chain
sig2 = var(Y)';
omega2 = 1;
phi = 0.5;
A = [1; zeros(n-1,1)];                                     % loading of INDPRO fixed at 1

store_F = zeros(nsim,T);
store_A = zeros(nsim,n);
store_theta = zeros(nsim,2);                               % [phi, omega2]
for isim = 1:(nsim + burnin)
    % sample the factor
    Hphi = ssm.diffmat(T, phi);
    Kf = Hphi'*Hphi/omega2 + sum(A.^2./sig2)*speye(T);
    F = ssm.simulate_states(Kf, Y*(A./sig2));

    % sample the loadings a_2, ..., a_n
    Ka = 1/Va + (F'*F)./sig2(2:n);
    A(2:n) = ssm.simulate_states(spdiags(Ka,0,n-1,n-1), a0/Va + (F'*Y(:,2:n))'./sig2(2:n));

    % sample sig2
    E_y = Y - F*A';
    sig2 = 1 ./ gamrnd(nusig2 + T/2, 1 ./ (Ssig2 + sum(E_y.^2)'/2));

    % sample omega2
    E_f = [F(1); F(2:end) - phi*F(1:end-1)];
    omega2 = 1 / gamrnd(nuomega2 + T/2, 1 / (Somega2 + sum(E_f.^2)/2));

    % sample phi from its normal full conditional, truncated to (-1, 1)
    Zf = [0; F(1:end-1)];
    Kphi = 1/Vphi + sum(Zf.^2)/omega2;
    phi_hat = (phi0/Vphi + sum(Zf.*F)/omega2)/Kphi;
    accepted = false;
    while ~accepted
        phi_prop = phi_hat + sqrt(1/Kphi)*randn;
        if abs(phi_prop) < 1
            phi = phi_prop;
            accepted = true;
        end
    end

    if isim > burnin
        isave = isim - burnin;
        store_F(isave,:) = F';
        store_A(isave,:) = A';
        store_theta(isave,:) = [phi omega2];
    end
end
F_mean = mean(store_F)';
F_q = quantile(store_F, [.05 .95])';
theta_mean = mean(store_theta);
theta_CI = quantile(store_theta, [.025 .975]);
A_mean = mean(store_A)';

fprintf(['\nFRED-MD, n = %d series, March 1959-November 2025 (T = %d), ' ...
    '%d draws after %d burn-in\n'], n, T, nsim, burnin);
fprintf('   phi     posterior mean %.3f, 95%% CI [%.3f, %.3f]\n', theta_mean(1), theta_CI(:,1));
fprintf('   omega2  posterior mean %.3f, 95%% CI [%.3f, %.3f]\n', theta_mean(2), theta_CI(:,2));
iapr = find(month_idx == round(12*(2020 + 4/12)));
fprintf('   factor in April 2020: posterior mean %.2f, 90%% band [%.2f, %.2f]\n', ...
    F_mean(iapr), F_q(iapr,:));
[~, order] = sort(abs(A_mean), 'descend');
top = order(1:5);
fprintf('   largest loadings: %s\n', strjoin(cellfun(@(s, a) sprintf('%s %.2f', s, a), ...
    varnames(top), num2cell(A_mean(top))', 'UniformOutput', false), ', '));

%% Figure: the factor over the full sample and before 2020, NBER recessions shaded
figure('Name', 'ex06 dynamic factor');
subsets = {true(T,1), tid < 2020};
titles = {'posterior mean of the business-cycle factor', 'the same, before 2020'};
for p = 1:2
    s = subsets{p};
    subplot(2,1,p); hold on
    yl = [min(F_mean(s)) - .5, max(F_mean(s)) + .5];
    shade_nber_recessions(yl(1), yl(2));
    plot(tid(s), F_mean(s), 'k', 'LineWidth', 1.2);
    plot(tid(s), zeros(nnz(s),1), '--k');
    hold off; box off; xlim([tid(find(s,1)) tid(find(s,1,'last'))]); ylim(yl)
    title(titles{p})
end
drawnow

fprintf('\nex06 done.\n');

function shade_nber_recessions(ymin, ymax)
% The book's shade_nber_recessions.m: shades the NBER recessions from peak to trough
% on the current axes, whose x-axis is in decimal years

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
