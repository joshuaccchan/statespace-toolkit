%% ex05 - Missing data and mixed frequencies: a ragged edge, and monthly GDP
%
% In a conditionally Gaussian state space model, the missing values are jointly Gaussian
% given the observed values, the states and the parameters, and their precision matrix is
% banded, so they are drawn in one block from a single Cholesky factor; see Chan, Poon and
% Zhu (2023). Write the stacked data as y = So*yo + Sm*ym, with the selection matrices
% ssm.select_obs returns. Stacking the measurement equation gives
% Go*yo + Gm*ym = W*alpha + X*beta + e with e ~ N(0, Sigma), where Go and Gm include any
% difference matrices the model applies to y, and then
%
%   (ym | yo, alpha, theta) ~ N(ymhat, K^{-1}),   K = Gm'*Sigma^{-1}*Gm,
%   K*ymhat = Gm'*Sigma^{-1}*(W*alpha + X*beta - Go*yo),
%
% which ssm.simulate_states draws without forming K^{-1}.
%
% Section 1 checks that the missing values are drawn from the right distribution. It
% removes values from a generated VAR(1) in three patterns at once, a series that starts
% late, a hole in the middle and a ragged edge at the end, and compares the draws with the
% values removed and with the dense conditional normal.
%
% Section 2 estimates monthly real GDP growth, which no statistical agency publishes. It
% enters a monthly VAR as a fourth variable that is missing in every month. Each observed
% quarterly growth rate is tied to five consecutive monthly values by the log-linear
% aggregation of Mariano and Murasawa (2003),
%
%   z = (y_t + 2*y_{t-1} + 3*y_{t-2} + 2*y_{t-3} + y_{t-4})/3,
%
% which stacks over the sample into the hard constraint M*ym = z. The quarterly rates fix
% the weighted sum of the monthly values in each quarter, and the three monthly indicators
% fix their timing within the quarter, through the VAR dynamics and the contemporaneous
% covariance. One update of an unconstrained draw imposes the constraint exactly,
% ym = u + K^{-1}*M'*(M*K^{-1}*M')^{-1}*(z - M*u), reusing the Cholesky factor
% ssm.simulate_states returns. The result is a set of model implied monthly GDP
% estimates, and their accuracy depends on the indicators and on the VAR. Mariano and
% Murasawa (2003, 2010) and Schorfheide and Song (2015) estimate monthly GDP this way.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Section 2 and Algorithm 2.
% Mariano, R.S. and Murasawa, Y. (2003). A New Coincident Index of Business Cycles
% Based on Monthly and Quarterly Series, Journal of Applied Econometrics, 18(4):
% 427-443.
% Mariano, R.S. and Murasawa, Y. (2010). A Coincident Index, Common Factors, and
% Monthly Real GDP, Oxford Bulletin of Economics and Statistics, 72(1): 27-46.
% McCracken, M.W. and Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic
% Research, Journal of Business and Economic Statistics, 34(4): 574-589.
% Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
% Chapman & Hall/CRC, Algorithm 2.6.
% Schorfheide, F. and Song, D. (2015). Real-Time Forecasting with a Mixed-Frequency
% VAR, Journal of Business and Economic Statistics, 33(3): 366-380.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex05: missing data and mixed frequencies ===\n');

%% Section 1: a late start, a hole and a ragged edge, in a generated VAR(1)
rng(42);
n = 4; T = 200;
Phi = [.5 .1 0 0; .2 .4 .1 0; 0 .2 .5 .1; .1 0 .2 .3];
CSig = [1 0 0 0; .5 1 0 0; .3 .4 1 0; .2 .3 .5 1]/sqrt(2);
Sig = CSig*CSig';

Ytrue = zeros(T,n);
yt = zeros(n,1);
for t = 1:T
    yt = Phi*yt + CSig*randn(n,1);
    Ytrue(t,:) = yt';
end

Y = Ytrue;
Y(1:24,4) = NaN;                                 % a series that starts two years late
Y(91:93,2) = NaN;                                % a hole
Y(T,1) = NaN; Y(T-1:T,3) = NaN; Y(T-2:T,4) = NaN;   % the ragged edge at the end

[So, Sm, yo] = ssm.select_obs(Y);
L = sparse(2:T,1:T-1,1,T,T);                     % the lag operator on the stacked path
H = speye(T*n) - kron(L, Phi);                   % H*y = e, with y_0 = 0
iSig = kron(speye(T), inv(Sig));
Gm = H*Sm; Go = H*So;
K = Gm'*iSig*Gm;
c = -Gm'*iSig*(Go*yo);                           % no states or regressors in this model
ndraws = 20000;
[ym, ymhat] = ssm.simulate_states(K, c, ndraws);

ytrue_m = Sm'*reshape(Ytrue', T*n, 1);
sd = std(ym, 0, 2);
band = quantile(ym, [.05 .95], 2);
inside = ytrue_m >= band(:,1) & ytrue_m <= band(:,2);
[blo, bup] = bandwidth(full(K));

fprintf('\n-- Section 1: %d of %d values blanked out, VAR(1) with n = %d, T = %d\n', ...
    numel(ytrue_m), T*n, n, T);
fprintf('precision of the missing data: %d x %d, %d nonzeros, bandwidth %d\n', ...
    size(K,1), size(K,2), nnz(K), max(blo,bup));

im = find(isnan(reshape(Y', T*n, 1)));
tm = ceil(im/n); vm = im - (tm-1)*n;
grp = {'series starting late', vm == 4 & tm <= 24; ...
       'hole in the middle',   vm == 2 & tm >= 91 & tm <= 93; ...
       'ragged edge',          tm > T-3};
fprintf('\n%-22s %5s %9s %9s %9s\n', 'pattern', 'count', 'RMSE', 'post sd', 'coverage');
for g = 1:size(grp,1)
    j = grp{g,2};
    fprintf('%-22s %5d %9.3f %9.3f %8.1f%%\n', grp{g,1}, sum(j), ...
        sqrt(mean((ymhat(j) - ytrue_m(j)).^2)), mean(sd(j)), 100*mean(inside(j)));
end
fprintf('%-22s %5d %9.3f %9.3f %8.1f%%\n', 'all', numel(ytrue_m), ...
    sqrt(mean((ymhat - ytrue_m).^2)), mean(sd), 100*mean(inside));

% the same moments, from conditioning the joint normal of y with dense algebra
V = inv(full(H'*iSig*H));
io = find(~isnan(reshape(Y', T*n, 1)));
mudense = V(im,io)*(V(io,io)\yo);
fprintf('\nmean against dense conditioning:   max abs difference %.2e\n', ...
    norm(ymhat - mudense, inf));
fprintf('draws against that mean:           max abs difference %.3f posterior sd\n', ...
    max(abs(mean(ym,2) - ymhat)./sd));

%% Section 2: estimating monthly GDP
rng(42);
nsim = 4000; burnin = 1000;

% three monthly indicators, September 1989 to December 2024, as percent growth
raw = readtable(fullfile(fileparts(mfilename('fullpath')), 'data', 'FRED-MD.csv'), ...
    'VariableNamingRule', 'preserve');
month_idx = round(12*raw{:,1});
keep = month_idx >= 1989*12+9 & month_idx <= 2024*12+12;
ind = {'INDPRO', 'PAYEMS', 'RPI'};
[~, cols] = ismember(ind, raw.Properties.VariableNames);
Xm = 100*raw{keep, cols};
assert(all(~isnan(Xm(:))), 'the indicators must be complete over this sample');
mid = month_idx(keep);
T = size(Xm,1);
n = numel(ind) + 1;                              % the indicators and monthly GDP growth

% quarterly real GDP growth, 1990Q1 to 2024Q4, in percent
gdp = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', 'USGDP.csv'), ...
    'Range', 'B173:B313');                       % 1989Q4-2024Q4
zq = 100*diff(log(gdp));
qend = (7:3:T)';                                 % the last month of each quarter
assert(numel(zq) == numel(qend), 'the quarterly and monthly samples must line up');

% the aggregation z = (y_t + 2*y_{t-1} + 3*y_{t-2} + 2*y_{t-3} + y_{t-4})/3, as M*ym = z
w = [1 2 3 2 1]'/3;
M = sparse(repmat((1:numel(qend))',5,1), reshape(qend - (0:4), [], 1), ...
    kron(w, ones(numel(qend),1)), numel(qend), T);

% the means are fixed at sample values and taken out: the aggregation weights sum to 3,
% so the mean of monthly GDP growth implied by the quarterly mean is mean(zq)/3
mug = mean(zq)/3;
Y = [Xm - mean(Xm,1), nan(T,1)];
zd = zq - 3*mug;
[So, Sm, yo] = ssm.select_obs(Y);
L = sparse(2:T,1:T-1,1,T,T);

% prior: vec(A) | Sigma ~ N(0, Sigma kron V0) for y_t = A'*y_{t-1} + e_t, Sigma ~ IW(nu0, S0)
iV0 = eye(n)/.5^2; nu0 = n + 3; S0 = eye(n);
A = zeros(n); Sigma = cov(Xm(:,1))*eye(n);
store_g = zeros(nsim, T);
store_res = zeros(nsim, 1);

fprintf('\n-- Section 2: %d months, %d quarterly observations, n = %d, %d draws\n', ...
    T, numel(zq), n, nsim);
start_time = tic;
for loop = 1:nsim + burnin
    % (1) monthly GDP growth, drawn under the aggregation constraint
    H = speye(T*n) - kron(L, A');
    iSig = kron(speye(T), inv(Sigma));
    Gm = H*Sm; Go = H*So;
    K = Gm'*iSig*Gm;
    [u, ~, C] = ssm.simulate_states(K, -Gm'*iSig*(Go*yo));
    U = C'\(C\full(M'));                         % K^{-1}*M', by the factor already formed
    ymd = u + U*((M*U)\(zd - M*u));

    % (2) the VAR coefficients and covariance, from the completed data
    Ycomp = reshape(So*yo + Sm*ymd, n, T)';
    Xreg = [zeros(1,n); Ycomp(1:T-1,:)];
    Kpost = iV0 + Xreg'*Xreg;
    Apost = Kpost\(Xreg'*Ycomp);
    Spost = S0 + Ycomp'*Ycomp - Apost'*Kpost*Apost;
    Sigma = iwishrnd((Spost + Spost')/2, nu0 + T);
    A = Apost + chol(Kpost,'lower')'\randn(n)*chol(Sigma,'lower')';

    if loop > burnin
        store_g(loop-burnin,:) = ymd' + mug;
        store_res(loop-burnin) = norm(M*ymd - zd, inf);
    end
end
fprintf('sampling takes %.1f seconds\n', toc(start_time));

ghat = mean(store_g, 1)';
gband = quantile(store_g, [.05 .95], 1)';
fprintf('the aggregation constraint holds in every draw: max residual %.2e\n', ...
    max(store_res));
fprintf('the posterior mean path aggregates to the observed quarters: max gap %.2e\n', ...
    norm(M*(ghat - mug) - zd, inf));
fprintf('correlation of estimated monthly GDP growth with industrial production: %.2f\n', ...
    corr(ghat, Xm(:,1)));

fprintf('\nestimated monthly real GDP growth, percent per month, a latent variable\n');
fprintf('of the model: posterior mean (90%% band)\n');
show = find(mid >= 2020*12+2 & mid <= 2020*12+7);
for j = show'
    fprintf('  %4d M%02d %7.2f  (%6.2f, %6.2f)\n', floor((mid(j)-1)/12), ...
        mid(j) - 12*floor((mid(j)-1)/12), ghat(j), gband(j,1), gband(j,2));
end
k2020q2 = find(qend == find(mid == 2020*12+6));
fprintf('  2020Q2 from these months %7.2f, observed %7.2f\n', ...
    w'*ghat(qend(k2020q2) - (0:4)'), zq(k2020q2));
