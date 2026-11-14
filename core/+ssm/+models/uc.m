% ssm.models.uc - posterior sampler for an unobserved components model, a random-walk trend
% plus a gap whose log variances can each be constant or vary over time, with an optional
% MA(1) term in the gap, and its forecasts. Its special cases include UC and UCSV (Stock and
% Watson, 2007), and its gap can have the MA(1) errors with stochastic volatility of Chan
% (2013). Written for this toolkit.
%
%   res = ssm.models.uc(y)
%   res = ssm.models.uc(y, 'gap', 'rw', 'trend', 'const', 'ma', false, ...
%                       'nsim', 20000, 'burnin', 5000, 'horizon', 4)
%
%   y : T x 1 series
%
%   'gap'     : the log variance h_t of the gap: 'const', 'rw' (a random walk) or 'ar1' (a
%               stationary AR(1)); default 'rw'
%   'trend'   : the log variance g_t of the trend innovations, the same choices; default 'rw'
%   'ma'      : true adds the term psi*u_{t-1} to the gap; default false
%   'prior'   : struct with any of the fields gap and trend, each a struct with any of nu and
%               S, m0 and V0, phi0 and Vphi (see the model below), and tau1, a struct with any
%               of a0, b0, V1a and V1b
%   'nsim'    : draws kept, default 10000;  'burnin' : draws discarded first, default 1000
%   'seed'    : if given, rng(seed, 'twister') is set before the first draw
%   'thin'    : k > 0 also keeps every kth kept draw of the paths in res.draws; default 0
%   'horizon' : H > 0 also forecasts y_{T+1}, ..., y_{T+H} and their average with
%               ssm.models.uc_forecast; default 4; 0 for no forecasts
%   'quantiles' : probabilities of the forecast quantiles, default [.05 .1 .5 .9 .95];
%               [] for none
%
%   res.tau_mean, res.sdh_mean, res.sdg_mean : T x 1 posterior means of the trend and of
%               the standard deviations exp(h_t/2) and exp(g_t/2)
%   res.params : nsim x k draws of the free parameters, named in res.param_names: for each
%               component, log s2 of a constant variance; x0 and log s2 of a random walk;
%               mu, atanh(phi) and log s2 of an AR(1); then atanh(psi) with the MA term
%   res.last   : nsim x 4 draws of [tau_T h_T g_T u_T], where the forecasts start
%   res.acc    : acceptance rates over all sweeps of the Metropolis-Hastings steps for psi and
%               for the AR(1) coefficients of the gap and of the trend
%   res.forecast : with horizon > 0, the fields m, v, mavg and vavg of ssm.models.uc_forecast;
%               mean (1 x H) and avg_mean, the means of the predictive mixtures of y_{T+h} and
%               of the average; and, unless quantiles is empty, their quantiles q
%               (numel(quantiles) x H) and avg_q, and probs
%   res.draws  : with thin > 0, tau, sdh and sdg, each draws x T
%   res.spec, and the settings
%
% The model is
%   y_t = tau_t + u_t + psi*u_{t-1},  u_t ~ N(0, exp(h_t)),  u_0 = 0,
%   tau_t = tau_{t-1} + e_t,  e_t ~ N(0, exp(g_t)),  tau_1 ~ N(tau0, V1a + V1b*exp(g_1)),
% with tau0 ~ N(a0, b0), or fixed at a0 when b0 = 0, and psi ~ N(0, 1) on (-1, 1). A log
% variance is a constant log(s2), s2 ~ IG(nu, S); a random walk x_t = x_{t-1} + N(0, s2),
% x_1 ~ N(x0, s2), x0 ~ N(m0, V0), s2 ~ IG(nu, S); or an AR(1)
% x_t = mu + phi*(x_{t-1} - mu) + N(0, s2) started from its stationary distribution,
% mu ~ N(m0, V0), phi ~ N(phi0, Vphi) on (-1, 1), s2 ~ IG(nu, S). IG(nu, S) has density
% proportional to x^(-nu-1)*exp(-S/x). The default priors are a constant variance IG(3, 2)
% for the gap and IG(10, .18) for the trend; a random walk with
% m0 = 0, V0 = 5 and IG(10, .45); an AR(1) with m0 = 0, V0 = 5, phi0 = .9, Vphi = 1 and
% IG(10, .45); and tau1 with a0 = 0, b0 = 5, V1a = 0 and V1b = 1 when the trend's log
% variance is a random walk, as in UCSV, and with a0 = 0, b0 = 0, V1a = 5 and V1b = 0
% otherwise.
%
% Each sweep draws the trend in one block by ssm.simulate_states (with the MA term, through
% s = Hpsi\tau, whose gap errors are independent); each log variance and its parameters, a
% random walk by ssm.ksc_rw_h0 and an AR(1) by ssm.ksc_ar1_mean, its AR coefficient by
% independence Metropolis-Hastings; tau0; and psi by ssm.ma_params. The auxiliary mixture
% works on log(e.^2 + 1e-4).
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with Application to
% Inflation Forecast, Journal of Econometrics, 176(2): 162-172.
% Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications,
% Chapman & Hall/CRC.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
% Estimation in State Space Models, International Journal of Mathematical Modelling and
% Numerical Optimisation, 1(1/2): 101-120.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference
% and Comparison with ARCH Models, Review of Economic Studies, 65(3): 361-393.
% Stock, J.H. and Watson, M.W. (2007). Why Has U.S. Inflation Become Harder to Forecast?,
% Journal of Money, Credit and Banking, 39(s1): 3-33.

function res = uc(y, varargin)
gap = 'rw'; trend = 'rw'; ma = false; pr = struct(); seed = []; thin = 0;
nsim = 10000; burnin = 1000; horizon = 4; probs = [.05 .1 .5 .9 .95];
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('ssm:models:uc:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    val = varargin{iv+1};
    switch lower(char(string(varargin{iv})))
        case 'gap',       gap = char(val);
        case 'trend',     trend = char(val);
        case 'ma',        ma = logical(val);
        case 'prior',     pr = val;
        case 'nsim',      nsim = val;
        case 'burnin',    burnin = val;
        case 'seed',      seed = val;
        case 'thin',      thin = val;
        case 'horizon',   horizon = val;
        case 'quantiles', probs = val;
        otherwise, error('ssm:models:uc:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end
y = y(:); T = numel(y);
if T < 5 || any(~isfinite(y))
    error('ssm:models:uc:badData', 'y must hold at least five finite values');
end
if ~(isscalar(nsim) && nsim >= 1 && nsim == fix(nsim) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('ssm:models:uc:badOption', 'nsim must be a positive integer and burnin a nonnegative integer');
end
m = spec(gap, trend, ma, pr);
if ~isempty(seed), rng(seed, 'twister'); end

H = ssm.diffmat(T);
% initial values
st.gap = init(m.gap, log(var(y)/2), T);
st.trend = init(m.trend, log(var(y)/20), T);
tau0 = m.tau1.a0; psi = 0;
params = zeros(nsim, numel(m.free)); last = zeros(nsim, 4);
tau_sum = zeros(T,1); sdh_sum = zeros(T,1); sdg_sum = zeros(T,1);
if thin > 0
    nsave = floor(nsim/thin);
    Dtau = zeros(nsave, T); Dsdh = zeros(nsave, T); Dsdg = zeros(nsave, T);
end
acc = struct('psi', 0, 'gap_phi', 0, 'trend_phi', 0);

for loop = 1:nsim + burnin
    % tau: yt = s + u, and b = G*s - c ~ N(0, diag(v)), with b_1 = tau_1 - tau0
    if m.ma, Hpsi = ssm.diffmat(T, -psi); yt = Hpsi\y; else, Hpsi = speye(T); yt = y; end
    G = H*Hpsi;
    eh = exp(-st.gap.x);
    v = exp(st.trend.x); v(1) = m.tau1.V1a + m.tau1.V1b*v(1);
    iv = 1./v;
    c = [tau0; zeros(T-1,1)];
    K = G'*spdiags(iv, 0, T, T)*G + spdiags(eh, 0, T, T);
    s = ssm.simulate_states(K, G'*(iv.*c) + eh.*yt);
    tau = Hpsi*s;
    u = yt - s;
    b = [tau(1) - tau0; diff(tau)];

    % the gap: its errors u
    [st.gap, a] = draw_logvar(m.gap, st.gap, u, T);
    acc.gap_phi = acc.gap_phi + a;
    % the trend: its innovations b, from t = 2 when tau_1 has its own variance V1a
    if m.tau1.V1a == 0
        [st.trend, a] = draw_logvar(m.trend, st.trend, b/sqrt(m.tau1.V1b), T);
    else
        [st.trend, a] = draw_logvar(m.trend, st.trend, b(2:end), T);
        st.trend.x = [st.trend.x(1); st.trend.x];      % x(1) is not used
    end
    acc.trend_phi = acc.trend_phi + a;

    % tau0 | tau_1
    if m.tau1.b0 > 0
        v1 = m.tau1.V1a + m.tau1.V1b*exp(st.trend.x(1));
        K0 = 1/m.tau1.b0 + 1/v1;
        tau0 = (m.tau1.a0/m.tau1.b0 + tau(1)/v1)/K0 + randn/sqrt(K0);
    end

    % psi | y, tau, h, and the errors under the retained psi
    if m.ma
        [psi, u, a] = ssm.ma_params(y - tau, exp(-st.gap.x), psi, struct('psi0', 0, 'Vpsi', 1));
        acc.psi = acc.psi + a;
    end

    if loop > burnin
        i = loop - burnin;
        params(i,:) = free_vector(m, st, psi);
        last(i,:) = [tau(T) st.gap.x(T) st.trend.x(T) u(T)];
        tau_sum = tau_sum + tau; sdh_sum = sdh_sum + exp(st.gap.x/2); sdg_sum = sdg_sum + exp(st.trend.x/2);
        if thin > 0 && mod(i, thin) == 0
            k = i/thin;
            Dtau(k,:) = tau'; Dsdh(k,:) = exp(st.gap.x/2)'; Dsdg(k,:) = exp(st.trend.x/2)';
        end
    end
end
f = fieldnames(acc);
for j = 1:numel(f), acc.(f{j}) = acc.(f{j})/(nsim + burnin); end

res.tau_mean = tau_sum/nsim; res.sdh_mean = sdh_sum/nsim; res.sdg_mean = sdg_sum/nsim;
res.params = params; res.param_names = m.free; res.last = last; res.acc = acc;
res.spec = rmfield(m, 'free');
if horizon > 0
    fc = ssm.models.uc_forecast(res, horizon);
    fc.mean = mean(fc.m, 1); fc.avg_mean = mean(fc.mavg);
    if ~isempty(probs)
        fc.q = zeros(numel(probs), horizon);
        for h = 1:horizon
            fc.q(:,h) = arrayfun(@(p) mixquantile(p, fc.m(:,h), fc.v(:,h)), probs(:));
        end
        fc.avg_q = arrayfun(@(p) mixquantile(p, fc.mavg, fc.vavg), probs(:));
        fc.probs = probs;
    end
    res.forecast = fc;
end
if thin > 0, res.draws = struct('tau', Dtau, 'sdh', Dsdh, 'sdg', Dsdg); end
res.nsim = nsim; res.burnin = burnin; res.seed = seed; res.thin = thin; res.horizon = horizon;
end

function m = spec(gap, trend, ma, pr)
% the model: each component's type and prior, tau1, the MA term and the free parameters
types = {'const', 'rw', 'ar1'};
if ~any(strcmp(gap, types)) || ~any(strcmp(trend, types))
    error('ssm:models:uc:badOption', 'gap and trend must each be ''const'', ''rw'' or ''ar1''');
end
known = {'gap', 'trend', 'tau1'};
fn = fieldnames(pr);
for i = 1:numel(fn)
    if ~any(strcmp(fn{i}, known))
        error('ssm:models:uc:badOption', 'unknown prior field ''%s''', fn{i});
    end
end
c = {'gap', gap; 'trend', trend};
for i = 1:2
    switch c{i,2}
        case 'const'
            if i == 1, k = struct('type', 'const', 'nu', 3, 'S', 2);
            else, k = struct('type', 'const', 'nu', 10, 'S', .18); end
        case 'rw',  k = struct('type', 'rw', 'm0', 0, 'V0', 5, 'nu', 10, 'S', .45);
        case 'ar1', k = struct('type', 'ar1', 'm0', 0, 'V0', 5, 'phi0', .9, 'Vphi', 1, 'nu', 10, 'S', .45);
    end
    m.(c{i,1}) = override(k, pr, c{i,1});
end
if strcmp(trend, 'rw')
    m.tau1 = override(struct('a0', 0, 'b0', 5, 'V1a', 0, 'V1b', 1), pr, 'tau1');
else
    m.tau1 = override(struct('a0', 0, 'b0', 0, 'V1a', 5, 'V1b', 0), pr, 'tau1');
end
m.ma = double(ma);
m.free = {};
for ci = {'gap', 'trend'}
    switch m.(ci{1}).type
        case 'const', m.free = [m.free {[ci{1} ' log s2']}];
        case 'rw',    m.free = [m.free {[ci{1} ' x0'], [ci{1} ' log s2']}];
        case 'ar1',   m.free = [m.free {[ci{1} ' mu'], [ci{1} ' atanh phi'], [ci{1} ' log s2']}];
    end
end
if m.ma, m.free = [m.free {'atanh psi'}]; end
end

function k = override(k, pr, name)
% the default prior k with the fields of pr.(name), each of which k must have
if isfield(pr, name)
    fn = fieldnames(pr.(name));
    for i = 1:numel(fn)
        if ~isfield(k, fn{i}) || strcmp(fn{i}, 'type')
            error('ssm:models:uc:badOption', 'prior.%s has no field ''%s'' for this model', name, fn{i});
        end
        k.(fn{i}) = pr.(name).(fn{i});
    end
end
end

function s = init(k, lv, T)
s = struct('x', lv*ones(T,1), 's2', .05, 'x0', lv, 'mu', lv, 'phi', .9);
if strcmp(k.type, 'const'), s.s2 = exp(lv); end
end

function [s, accepted] = draw_logvar(k, s, e, ~)
% one component: e are its errors (the gap or the trend innovations); s.x is its
% log variance path of length numel(e)
n = numel(e); accepted = 0;
switch k.type
    case 'const'
        s.s2 = 1/gamrnd(k.nu + n/2, 1/(k.S + e'*e/2));
        s.x = log(s.s2)*ones(n,1);
    case 'rw'
        Ystar = log(e.^2 + .0001);
        s.x = ssm.ksc_rw_h0(Ystar, s.x, s.s2, s.x0);
        d = [s.x(1) - s.x0; diff(s.x)];
        s.s2 = 1/gamrnd(k.nu + n/2, 1/(k.S + d'*d/2));
        K0 = 1/k.V0 + 1/s.s2;
        s.x0 = (k.m0/k.V0 + s.x(1)/s.s2)/K0 + randn/sqrt(K0);
    case 'ar1'
        Ystar = log(e.^2 + .0001);
        s.x = ssm.ksc_ar1_mean(Ystar, s.x, s.mu, s.phi, s.s2);
        x = s.x; mu = s.mu; phi = s.phi;
        r = [(x(1) - mu)*sqrt(1 - phi^2); x(2:end) - phi*x(1:end-1) - mu*(1 - phi)];
        s.s2 = 1/gamrnd(k.nu + n/2, 1/(k.S + r'*r/2));
        % phi: the regression of x_t - mu on x_{t-1} - mu as proposal, the stationary
        % density of x_1 in the acceptance ratio
        X = x(1:end-1) - mu; z = x(2:end) - mu;
        D = 1/(1/k.Vphi + X'*X/s.s2);
        phic = D*(k.phi0/k.Vphi + X'*z/s.s2) + sqrt(D)*randn;
        g = @(q) -.5*log(s.s2/(1 - q^2)) - .5*(1 - q^2)/s.s2*(x(1) - mu)^2;
        if abs(phic) < .9999 && exp(g(phic) - g(phi)) > rand
            s.phi = phic; accepted = 1;
        end
        phi = s.phi;
        Dmu = 1/(1/k.V0 + ((n-1)*(1 - phi)^2 + (1 - phi^2))/s.s2);
        muhat = Dmu*(k.m0/k.V0 + (1 - phi^2)/s.s2*x(1) + (1 - phi)/s.s2*sum(x(2:end) - phi*x(1:end-1)));
        s.mu = muhat + sqrt(Dmu)*randn;
end
end

function v = free_vector(m, st, psi)
v = [];
for c = {'gap', 'trend'}
    k = m.(c{1}); s = st.(c{1});
    switch k.type
        case 'const', v = [v log(s.s2)]; %#ok<AGROW>
        case 'rw',    v = [v s.x0 log(s.s2)]; %#ok<AGROW>
        case 'ar1',   v = [v s.mu atanh(s.phi) log(s.s2)]; %#ok<AGROW>
    end
end
if m.ma, v = [v atanh(psi)]; end
end

function x = mixquantile(p, m, v)
% the p-quantile of the equally weighted mixture of N(m_i, v_i)
s = sqrt(v);
x = fzero(@(q) mean(normcdf((q - m)./s)) - p, [min(m - 8*s) max(m + 8*s)]);
end
