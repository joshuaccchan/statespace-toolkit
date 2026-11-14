% ssm.models.sv - posterior sampler for a stochastic volatility model whose log-volatility is
% a stationary AR(1) or a random walk, and whose errors can have an MA(1) or ARMA(1,1) term
% and Student-t tails. Its special cases include SV, SV-t, the MA(1) case of the moving
% average SV model of Chan (2013) and the ARMA(1,1) case of the model of Zhang, Chan and
% Cross (2020). Written for this toolkit.
%
%   res = ssm.models.sv(y)
%   res = ssm.models.sv(y, 'vol', 'ar1', 'errors', 'arma', 'dist', 't', ...
%                       'nsim', 20000, 'burnin', 1000)
%
%   y : T x 1 series
%
%   'vol'    : the log-volatility h_t, 'ar1' (a stationary AR(1)) or 'rw' (a random walk);
%              default 'ar1'
%   'errors' : 'iid', 'ma' (an MA(1) term) or 'arma' (ARMA(1,1)); default 'iid'
%   'dist'   : 'normal' or 't'; default 'normal'
%   'mean'   : false fixes mu at 0; default true
%   'prior'  : struct with any of the hyperparameters below that the model uses
%   'nsim'   : draws kept, default 10000;  'burnin' : draws discarded first, default 1000
%   'seed'   : if given, rng(seed, 'twister') is set before the first draw
%   'thin'   : k > 0 also keeps every kth kept draw of h in res.draws.h; default 0
%
%   res.params   : nsim x k draws, named in res.param_names, of those of mu, rho, psi, mu_h,
%                  phi, sig2_h and nu that the model has, in that order
%   res.sdh_mean : T x 1 posterior mean of exp(h_t/2)
%   res.lam_mean : with 't', T x 1 posterior mean of lam_t
%   res.acc      : acceptance rates over all sweeps of the Metropolis-Hastings steps for
%                  phi, psi and nu that the model has
%   res.draws.h  : with thin > 0, draws x T
%   res.spec, and the settings
%
% The model is
%   y_t = mu + e_t,  e_t = rho*e_{t-1} + u_t + psi*u_{t-1},  e_0 = u_0 = 0,
%   u_t = sqrt(lam_t)*exp(h_t/2)*eps_t,  eps_t ~ N(0, 1),  lam_t ~ IG(nu/2, nu/2),
% with rho = 0 unless 'arma', psi = 0 under 'iid', and lam_t = 1 under 'normal'. The
% log-volatility is either
%   h_t = mu_h + phi*(h_{t-1} - mu_h) + N(0, sig2_h),  h_1 ~ N(mu_h, sig2_h/(1 - phi^2)),
% or h_t = h_{t-1} + N(0, sig2_h), h_1 ~ N(h0, Vh). The priors are mu ~ N(mu0, Vmu);
% rho ~ N(rho0, Vrho) and psi ~ N(psi0, Vpsi), each on (-1, 1); mu_h ~ N(muh0, Vmuh);
% phi ~ N(phi0, Vphi) on (-1, 1); sig2_h ~ IG(nuh, Sh), with density proportional to
% x^(-nuh-1)*exp(-Sh/x); and nu ~ U(2, nuub). The defaults are mu0 = 0, Vmu = 10,
% rho0 = psi0 = 0, Vrho = Vpsi = 1, muh0 = 1, Vmuh = 10, phi0 = .97, Vphi = .1^2, nuh = 5,
% h0 = 0, Vh = 10 and nuub = 100, with Sh = .16 for an AR(1) and Sh = .04 for a random walk.
% Those that the SV guides set have the guides' values.
%
% rho and psi start at the point of a grid on [-0.975, 0.975] that minimizes the sum of
% squared errors u_t, since a chain of ssm.ma_params started far from the mode can keep its
% starting value (see its trap).
%
% Each sweep draws
%   1. mu from its normal conditional, given the errors' coefficients, h and lam;
%   2. psi by ssm.ma_params, or rho and psi by ssm.arma_params;
%   3. h by the auxiliary mixture sampler of Kim, Shephard and Chib (1998), applied to
%      log(u_t^2/lam_t + 1e-4): ssm.ksc_ar1_mean, or ssm.ksc_rw_diffuse on the data and the
%      path less h0;
%   4. with 't', lam_t from its inverse-gamma conditional, and nu by ssm.nu_studentt;
%   5. mu_h, phi and sig2_h by ssm.sv_params, or sig2_h of a random walk from its
%      inverse-gamma conditional.
% No step after 3 conditions on the mixture indicators, which ssm.ksc_ar1_mean and
% ssm.ksc_rw_diffuse draw just before h (Del Negro and Primiceri, 2015).
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with Application to
% Inflation Forecast, Journal of Econometrics, 176(2): 162-172.
% Del Negro, M. and Primiceri, G.E. (2015). Time Varying Structural Vector Autoregressions
% and Monetary Policy: A Corrigendum, Review of Economic Studies, 82(4): 1342-1345.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference
% and Comparison with ARCH Models, Review of Economic Studies, 65(3): 361-393.
% Zhang, B., Chan, J.C.C. and Cross, J.L. (2020). Stochastic Volatility Models with ARMA
% Innovations: An Application to G7 Inflation Forecasts, International Journal of
% Forecasting, 36(4): 1318-1328.

function res = sv(y, varargin)
vol = 'ar1'; errors = 'iid'; dist = 'normal'; has_mean = true; pr = struct();
nsim = 10000; burnin = 1000; seed = []; thin = 0;
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('ssm:models:sv:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    val = varargin{iv+1};
    switch lower(char(string(varargin{iv})))
        case 'vol',    vol = char(val);
        case 'errors', errors = char(val);
        case 'dist',   dist = char(val);
        case 'mean',   has_mean = logical(val);
        case 'prior',  pr = val;
        case 'nsim',   nsim = val;
        case 'burnin', burnin = val;
        case 'seed',   seed = val;
        case 'thin',   thin = val;
        otherwise, error('ssm:models:sv:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end
y = y(:); T = numel(y);
if T < 5 || any(~isfinite(y))
    error('ssm:models:sv:badData', 'y must hold at least five finite values');
end
if ~(isscalar(nsim) && nsim >= 1 && nsim == fix(nsim) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('ssm:models:sv:badOption', 'nsim must be a positive integer and burnin a nonnegative integer');
end
m = spec(vol, errors, dist, has_mean, pr);
p = m.prior;
is_ar1 = strcmp(vol, 'ar1'); is_t = strcmp(dist, 't');
if ~isempty(seed), rng(seed, 'twister'); end

% starting values, those of the SV guides
offset = 1e-4;
muh = log(var(y));
if muh == 0, muh = 1e-8; end           % ssm.sv_params skips the draw of mu_h when it is 0
phi = .98; sig2h = .04;
if ~is_ar1, sig2h = .01; end
h = muh*ones(T,1); lam = ones(T,1); nu = 5;
mu = 0; rho = 0; psi = 0;
if ~strcmp(errors, 'iid')
    e0 = y - has_mean*mean(y);
    psig = linspace(-.975, .975, 40);
    if strcmp(errors, 'arma'), rhog = psig; else, rhog = 0; end
    best = Inf;
    for a = rhog
        ea = e0 - a*[0; e0(1:end-1)];
        for b = psig
            ssr = sum(filter(1, [1 b], ea).^2);
            if ssr < best, best = ssr; rho = a; psi = b; end
        end
    end
end
hyp = struct('nuh', p.nuh, 'Sh', p.Sh);
if is_ar1
    hyp.phi0 = p.phi0; hyp.Vphi = p.Vphi; hyp.mu0 = p.muh0; hyp.Vmu = p.Vmuh;
end
if ~strcmp(errors, 'iid')
    hyp.psi0 = p.psi0; hyp.Vpsi = p.Vpsi;
    if strcmp(errors, 'arma'), hyp.rho0 = p.rho0; hyp.Vrho = p.Vrho; end
end
params = zeros(nsim, numel(m.names));
sdh_sum = zeros(T,1); lam_sum = zeros(T,1);
if thin > 0, Dh = zeros(floor(nsim/thin), T); end
acc = struct('phi', 0, 'psi', 0, 'nu', 0);
ones_T = ones(T,1);

for loop = 1:nsim + burnin
    if is_t, w = 1./(lam.*exp(h)); else, w = exp(-h); end

    % 1. mu: y filtered by the errors' lag polynomials is mu times the filtered ones
    if has_mean
        switch errors
            case 'iid',  z = y; x1 = ones_T;
            case 'ma',   z = filter(1, [1 psi], y); x1 = filter(1, [1 psi], ones_T);
            case 'arma', z = filter([1 -rho], [1 psi], y); x1 = filter([1 -rho], [1 psi], ones_T);
        end
        Kmu = 1/p.Vmu + sum(w.*x1.^2);
        mu = (p.mu0/p.Vmu + w'*(x1.*z))/Kmu + randn/sqrt(Kmu);
    end

    % 2. the coefficients of the errors
    e = y - mu;
    switch errors
        case 'iid',  u = e;
        case 'ma',   [psi, u, a] = ssm.ma_params(e, w, psi, hyp); acc.psi = acc.psi + a;
        case 'arma', [rho, psi, u, a] = ssm.arma_params(e, w, rho, psi, hyp); acc.psi = acc.psi + a;
    end

    % 3. h
    if is_t, ystar = log(u.^2./lam + offset); else, ystar = log(u.^2 + offset); end
    if is_ar1
        h = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h);
    else
        h = p.h0 + ssm.ksc_rw_diffuse(ystar - p.h0, h - p.h0, sig2h, p.Vh);
    end

    % 4. lam and nu
    if is_t
        e2 = u.^2.*exp(-h);
        lam = 1./gamrnd((nu + 1)/2, 2./(nu + e2));
        [nu, a] = ssm.nu_studentt(lam, nu, p.nuub);
        acc.nu = acc.nu + a;
    end

    % 5. the parameters of h
    if is_ar1
        [muh, phi, sig2h, a] = ssm.sv_params(h, muh, phi, hyp, 1);
        acc.phi = acc.phi + a;
    else
        dh = diff(h);
        sig2h = 1/gamrnd(p.nuh + (T-1)/2, 1/(p.Sh + dh'*dh/2));
    end

    if loop > burnin
        i = loop - burnin;
        v = [mu rho psi muh phi sig2h nu];
        params(i,:) = v(m.cols);
        sdh_sum = sdh_sum + exp(h/2);
        if is_t, lam_sum = lam_sum + lam; end
        if thin > 0 && mod(i, thin) == 0, Dh(i/thin,:) = h'; end
    end
end
f = {'phi', 'psi', 'nu'};
keep = [is_ar1, ~strcmp(errors, 'iid'), is_t];
acc = rmfield(acc, f(~keep));
for j = find(keep), acc.(f{j}) = acc.(f{j})/(nsim + burnin); end

res.params = params; res.param_names = m.names;
res.sdh_mean = sdh_sum/nsim;
if is_t, res.lam_mean = lam_sum/nsim; end
res.acc = acc;
if thin > 0, res.draws = struct('h', Dh); end
res.spec = rmfield(m, {'names', 'cols'});
res.nsim = nsim; res.burnin = burnin; res.seed = seed; res.thin = thin;
end

function m = spec(vol, errors, dist, has_mean, pr)
% the model: its options, its prior and the names and positions of its parameters
if ~any(strcmp(vol, {'ar1', 'rw'}))
    error('ssm:models:sv:badOption', 'vol must be ''ar1'' or ''rw''');
end
if ~any(strcmp(errors, {'iid', 'ma', 'arma'}))
    error('ssm:models:sv:badOption', 'errors must be ''iid'', ''ma'' or ''arma''');
end
if ~any(strcmp(dist, {'normal', 't'}))
    error('ssm:models:sv:badOption', 'dist must be ''normal'' or ''t''');
end
p = struct('nuh', 5);
if has_mean, p.mu0 = 0; p.Vmu = 10; end
if strcmp(errors, 'arma'), p.rho0 = 0; p.Vrho = 1; end
if ~strcmp(errors, 'iid'), p.psi0 = 0; p.Vpsi = 1; end
if strcmp(vol, 'ar1')
    p.muh0 = 1; p.Vmuh = 10; p.phi0 = .97; p.Vphi = .1^2; p.Sh = .16;
else
    p.h0 = 0; p.Vh = 10; p.Sh = .04;
end
if strcmp(dist, 't'), p.nuub = 100; end
fn = fieldnames(pr);
for i = 1:numel(fn)
    if ~isfield(p, fn{i})
        error('ssm:models:sv:badOption', 'prior has no field ''%s'' for this model', fn{i});
    end
    p.(fn{i}) = pr.(fn{i});
end
m.vol = vol; m.errors = errors; m.dist = dist; m.mean = has_mean; m.prior = p;
all_names = {'mu', 'rho', 'psi', 'mu_h', 'phi', 'sig2_h', 'nu'};
used = [has_mean, strcmp(errors, 'arma'), ~strcmp(errors, 'iid'), ...
    strcmp(vol, 'ar1'), strcmp(vol, 'ar1'), true, strcmp(dist, 't')];
m.cols = find(used);
m.names = all_names(used);
end
