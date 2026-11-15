%% build - regenerate the figures and numbers of tutorials/sv_volatility/README.md
%
% Five stochastic volatility models of the daily return on the dollar-euro exchange rate,
% 100 times the log change in the ECB reference rate (examples/data/USDEUR_daily.csv),
% against the exponentially weighted moving average (EWMA) of RiskMetrics. SV, SV-t and
% SV-MA are models of Chan and Grant (2016), with their priors, which are the defaults of
% ssm.models.sv; SV-RW and SV-RW-t have a random-walk log-volatility. At each origin from
% the last day of 2019 to the day before the last, every model is estimated on all the
% returns up to the origin, an expanding window as in the forecasting code of Chan and Grant
% (2016), and forecasts the next day's return by ssm.models.sv_forecast.
% The forecasts are scored by the log predictive likelihood, as in Chan and Grant (2016),
% and the variance forecasts by QLIKE and the squared error against the squared return
% (Patton, 2011). Part 0 checks the predictive distributions; Part 1 runs one chain per
% model on the whole sample; Part 2 runs the recursive forecasts, on parallel workers when
% the Parallel Computing Toolbox is installed, saved in runs/ so that an interrupted build
% resumes; Part 3 prints the tables and draws the figures, written next to this file. For
% Figure 2 it reruns the SV and SV-t chains of one origin, with their seeds of Part 2.
% Everything printed goes to a log in tempdir.
%
% Usage, from anywhere:  run tutorials/sv_volatility/build.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
logf = fullfile(tempdir, 'ssm_sv_volatility_build_log.txt');
if exist(logf, 'file'), delete(logf); end
diary(logf);
fprintf('build.m of tutorials/sv_volatility, %s, MATLAB %s\n', char(datetime('now')), version);
t0 = tic;
run(fullfile(repo, 'setup.m'));

% settings
seed = 20261114;
names = {'EWMA', 'SV', 'SV-t', 'SV-RW', 'SV-RW-t', 'SV-MA'};
nm = numel(names); issv = 2:nm;
W = Inf;                                        % estimation window: every return up to the origin
whole = struct('nsim', 10000, 'burnin', 1000);  % whole sample, the run length of Chan and Grant (2016)
fc = struct('nsim', 5000, 'burnin', 1000);      % each origin
first_target = datetime(2020, 1, 1);            % the first day forecast is the first after this
blk = 50;                                       % origins per file in runs/
cver = 1;                                       % cache version, in every key

data = readtable(fullfile(repo, 'examples', 'data', 'USDEUR_daily.csv'));
ok = ~isnan(data.USD_EUR);
p = data.USD_EUR(ok); d = data.observation_date(ok);
y = 100*diff(log(p)); dates = d(2:end); T = numel(y);
assert(T == 7135 && dates(1) == datetime(1999,1,5) && dates(end) == datetime(2026,11,13) && ...
    p(1) == 1.1789 && p(end) == 1.1052, ...
    'build: examples/data/USDEUR_daily.csv is not the file this page was built from');
fprintf('%d daily returns, %s to %s: mean %.4f, sd %.3f, kurtosis %.2f, first autocorrelation %.3f\n', ...
    T, char(dates(1)), char(dates(end)), mean(y), std(y), kurtosis(y), corr(y(1:end-1), y(2:end)));
fprintf('mean squared return %.4f; %d returns are zero\n', mean(y.^2), sum(y == 0));
lme = @(x) max(x) + log(mean(exp(x - max(x))));

%% Part 0: checks
fprintf('\n=== Part 0: checks ===\n');
% (a) the EWMA recursion against its loop
yw = y(max(1, T-W+1):T); s2 = mean(yw.^2);
for k = 1:numel(yw), s2 = .94*s2 + .06*yw(k)^2; end
assert(abs(ewma(yw, .94)/s2 - 1) < 1e-12, 'build: the EWMA variance differs from its recursion');
fprintf('(a) EWMA variance for the day after the sample: %.4f, the recursion %.4f\n', ewma(yw, .94), s2);

% (b) each stochastic volatility model, on the window of the last origin: the predictive distribution
% of ssm.models.sv_forecast against y_{T+1} simulated forward from the same draws, compared
% by the distribution functions at the 5th, ..., 95th percentiles of the simulated values
R = 20; pr = (.05:.05:.95)';
for k = issv
    o = sv_options(names{k});
    res = ssm.models.sv(yw, o{:}, 'nsim', 2000, 'burnin', 1000, 'seed', seed + k);
    F = cell(R, 1); ys = zeros(2000, R);
    for r = 1:R, F{r} = ssm.models.sv_forecast(res); ys(:,r) = sim_next(res); end
    q = quantile(ys(:), pr);
    Fq = mean(cell2mat(cellfun(@(f) f.cdf(q), F', 'UniformOutput', false)), 2);
    dk = max(abs(Fq - pr));
    assert(dk < .01, 'build: the predictive distribution of %s differs from simulated values', names{k});
    fprintf('(b) %-8s largest difference in the distribution function: %.4f\n', names{k}, dk);
end

%% Part 1: the whole sample
fprintf('\n=== Part 1: the whole sample ===\n');
rdir = fullfile(tdir, 'runs');
if ~exist(rdir, 'dir'), mkdir(rdir); end
ch = cell(1, nm);
for k = issv
    ch{k} = cached(fullfile(rdir, [strrep(names{k}, '-', '_') '_chain.mat']), ...
        [whole.nsim whole.burnin seed + k cver], @() run_chain(y, names{k}, whole, seed + k));
    P = ch{k}.params; q = quantile(P, [.05 .95]);
    fprintf('%-8s %d draws after %d burn-in; posterior mean (90%% interval):\n', names{k}, ...
        whole.nsim, whole.burnin);
    for j = 1:numel(ch{k}.param_names)
        fprintf('    %-7s %8.4f (%.4f, %.4f)\n', ch{k}.param_names{j}, mean(P(:,j)), q(:,j));
    end
    fprintf('    inefficiency factors (Bartlett, 100 lags): %s\n', ...
        mat2str(ssm.inefficiency_factor(P, 100), 3));
    a = ch{k}.acc; fa = fieldnames(a);
    for j = 1:numel(fa), fprintf('    acceptance of %s: %.2f\n', fa{j}, a.(fa{j})); end
end

%% Part 2: the recursive forecasts
fprintf('\n=== Part 2: recursive forecasts ===\n');
i0 = find(dates > first_target, 1);
orig = i0 - 1 : T - 1;                          % the forecast of y(t+1) is made at t
no = numel(orig);
if isinf(W), wl = 'all returns up to the origin'; else, wl = sprintf('the last %d returns', W); end
fprintf('%d origins, %s to %s; targets %s to %s; estimated on %s; %d draws after %d\n', ...
    no, char(dates(orig(1))), char(dates(orig(end))), char(dates(orig(1)+1)), ...
    char(dates(orig(end)+1)), wl, fc.nsim, fc.burnin);
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')) && isempty(gcp('nocreate'))
    c = parcluster('Processes'); nw = max(1, feature('numcores') - 2);
    c.NumWorkers = max(c.NumWorkers, nw); parpool(c, nw);
end
sc = nan(no, 7, nm);                            % [m v lpl q01 q05 q95 q99] by origin and model
nb = ceil(no/blk); t1 = tic;
for b = 1:nb
    idx = (b-1)*blk + 1 : min(b*blk, no);
    sc(idx,:,:) = cached(fullfile(rdir, sprintf('fc_block_%03d.mat', b)), ...
        [W fc.nsim fc.burnin seed cver orig(idx([1 end]))], ...
        @() run_block(y, orig(idx), W, names, fc, seed));
    fprintf('block %d of %d, %s to %s, %.1f minutes so far\n', b, nb, char(dates(orig(idx(1)))), ...
        char(dates(orig(idx(end)))), toc(t1)/60);
end

%% Part 3: table and figures
fprintf('\n=== Part 3: table and figures ===\n');
yy = y(orig + 1);
m = squeeze(sc(:,1,:)); v = squeeze(sc(:,2,:)); lpl = squeeze(sc(:,3,:));
f2 = v + m.^2;                                  % the forecast of the squared return
ql = log(f2) + yy.^2./f2;                       % QLIKE, up to terms free of the forecast
se = (yy.^2 - f2).^2;
in90 = yy >= squeeze(sc(:,5,:)) & yy <= squeeze(sc(:,6,:));
in98 = yy >= squeeze(sc(:,4,:)) & yy <= squeeze(sc(:,7,:));
L = 10;                                         % lags of the Newey-West standard errors
nwse = @(d) sqrt(ssm.specvar0(d, L));           % the standard error of the mean of d
fprintf('%d one-day-ahead forecasts, %s to %s; standard errors: Newey-West, Bartlett, %d lags\n', ...
    no, char(dates(orig(1)+1)), char(dates(orig(end)+1)), L);
fprintf(['| Model | Log predictive likelihood | Difference against EWMA (s.e.) | ' ...
    'QLIKE difference against EWMA (s.e.) | MSE relative to EWMA | 90%% interval coverage | ' ...
    '98%% interval coverage |\n|---|---|---|---|---|---|---|\n']);
for k = 1:nm
    dl = lpl(:,k) - lpl(:,1); dq = ql(:,k) - ql(:,1);
    c = {'0.0', '0.0000'};
    if k > 1
        c = {sprintf('%.1f (%.1f)', sum(dl), no*nwse(dl)), sprintf('%.4f (%.4f)', mean(dq), nwse(dq))};
    end
    fprintf('| %s | %.1f | %s | %s | %.3f | %.3f | %.3f |\n', names{k}, sum(lpl(:,k)), c{:}, ...
        mean(se(:,k))/mean(se(:,1)), mean(in90(:,k)), mean(in98(:,k)));
end
% pairs of models that differ in one feature: {feature, model, the model compared with}
pairs = {'Student-t errors', 'SV-t', 'SV'
    'Student-t errors', 'SV-RW-t', 'SV-RW'
    'Random-walk log-volatility', 'SV-RW', 'SV'
    'Random-walk log-volatility', 'SV-RW-t', 'SV-t'
    'MA(1) errors', 'SV-MA', 'SV'};
fprintf('\nPairs of models that differ in one feature:\n');
fprintf(['| Feature | Comparison | Log predictive likelihood difference (s.e.) | ' ...
    'QLIKE mean difference (s.e.) |\n|---|---|---|---|\n']);
for i = 1:size(pairs, 1)
    a = strcmp(names, pairs{i,2}); b = strcmp(names, pairs{i,3});
    dl = lpl(:,a) - lpl(:,b); dq = ql(:,a) - ql(:,b);
    fprintf('| %s | %s minus %s | %.1f (%.1f) | %.4f (%.4f) |\n', pairs{i,:}, sum(dl), ...
        no*nwse(dl), mean(dq), nwse(dq));
end
fprintf('\nCorrelation of the forecast volatility with |y(t+1)|:\n');
for k = 1:nm, fprintf('%-8s %.3f\n', names{k}, corr(sqrt(f2(:,k)), abs(yy))); end
yr = year(dates(orig + 1));
fprintf('\nLog predictive likelihood by year, difference against EWMA:\n');
for yv = unique(yr)'
    j = yr == yv;
    fprintf('%d (%3d days) %s\n', yv, sum(j), sprintf('%8.1f', sum(lpl(j,issv) - lpl(j,1))));
end

% the days on which SV-t gains most on the EWMA, with the volatility forecasts for each of
% them and for the day after
ix = cellfun(@(s) find(strcmp(names, s)), {'EWMA', 'SV', 'SV-t'});
g = lpl(:,ix(3)) - lpl(:,ix(1));
[~, j] = sort(g, 'descend'); j = j(1:10);
vf = sqrt(f2(:,ix));
fprintf(['\nSV-t against EWMA: a higher log predictive likelihood on %d of %d days; ' ...
    'the 10 largest gains add up to %.1f of %.1f\n'], sum(g > 0), no, sum(g(j)), sum(g));
fprintf(['| Day | Return | Return over the EWMA volatility | Gain | Volatility forecast: EWMA | ' ...
    'SV | SV-t | For the next day: EWMA | SV | SV-t |\n|---|---|---|---|---|---|---|---|---|---|\n']);
for i = j'
    nx = nan(1, 3); if i < no, nx = vf(i+1,:); end
    fprintf('| %s | %.2f | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n', ...
        char(dates(orig(i)+1)), yy(i), yy(i)/vf(i,1), g(i), vf(i,:), nx);
end

% every forecast, for reuse: forecasts.csv next to this file
ko = repmat(1:nm, no, 1); io = repmat((1:no)', 1, nm);
fcsv = table(names(ko(:))', dates(orig(io(:))), dates(orig(io(:)) + 1), yy(io(:)), ...
    'VariableNames', {'model', 'origin', 'target', 'realized'});
fcsv.mean = m(:); fcsv.variance = v(:); fcsv.log_predictive_density = lpl(:);
for j = 4:7
    nmj = {'lower_98', 'lower_90', 'upper_90', 'upper_98'};
    qj = squeeze(sc(:,j,:)); fcsv.(nmj{j-3}) = qj(:);
end
writetable(fcsv, fullfile(tdir, 'forecasts.csv'));
fprintf('forecasts.csv: %d rows written\n', height(fcsv));

% Figure 1: the cumulative log predictive likelihood against the EWMA
cols = lines(nm); td = dates(orig + 1);
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 10]); hold on
hc = gobjects(1, numel(issv));
for i = 1:numel(issv)
    k = issv(i);
    hc(i) = plot(td, cumsum(lpl(:,k) - lpl(:,1)), 'Color', cols(k,:), 'LineWidth', 1.3);
end
yline(0, 'Color', [.6 .6 .6]);
hold off; box off; xlim(td([1 end]));
legend(hc, names(issv), 'Location', 'southoutside', 'Orientation', 'horizontal'); legend boxoff
exportgraphics(fig, fullfile(tdir, 'fig_score.png'), 'Resolution', 150); close(fig);

% Figure 2: the day of the largest gain of SV-t over the EWMA. (a) The log predictive densities
% of the EWMA, SV and SV-t for that day, recomputed with the seeds of Part 2; (b) their
% volatility forecasts from 20 days before that day to 40 days after it
[~, i1] = max(g); t1 = orig(i1);
xg = (-3.5:.01:3.5)';
lg = cached(fullfile(rdir, sprintf('density_%d.mat', t1)), ...
    [W fc.nsim fc.burnin seed cver t1 xg(1) xg(end) numel(xg)], ...
    @() day_density(y, t1, W, names(ix), ix, fc, seed, xg));
d1 = max(abs(lg(end,:) - lpl(i1,ix)));
w = max(1, i1 - 20) : min(no, i1 + 40);
fprintf(['\n%s, the largest gain of SV-t: log predictive densities at the outcome %s' ...
    '(EWMA, SV, SV-t), recomputed; largest difference from Part 2 %.1e; gains over the EWMA: ' ...
    'SV %.1f, SV-t %.1f; panel (b) %s to %s\n'], char(td(i1)), sprintf('%.2f ', lg(end,:)), d1, ...
    lg(end,2:3) - lg(end,1), char(td(w(1))), char(td(w(end))));
assert(d1 < 1e-8, 'build: the recomputed densities differ from those of Part 2');
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 9]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile(tl); hold(ax1, 'on')
hd = gobjects(1, 3);
for i = 1:3, hd(i) = plot(ax1, xg, lg(1:end-1,i), 'Color', cols(ix(i),:), 'LineWidth', 1.3); end
hy = xline(ax1, yy(i1), '--', 'Color', [.3 .3 .3]);
hb = patch(ax1, NaN, NaN, [.8 .8 .8], 'EdgeColor', 'none');   % legend entry for the bars of (b)
hold(ax1, 'off'); box(ax1, 'off'); xlim(ax1, xg([1 end])); xlabel(ax1, 'return');
title(ax1, sprintf('(a) Log predictive density, %s', char(td(i1), 'd MMMM yyyy')), ...
    'FontWeight', 'normal');
ax2 = nexttile(tl); hold(ax2, 'on')
bar(ax2, td(w), abs(yy(w)), .8, 'FaceColor', [.8 .8 .8], 'EdgeColor', 'none');
for i = 1:3, plot(ax2, td(w), vf(w,i), 'Color', cols(ix(i),:), 'LineWidth', 1.3); end
hold(ax2, 'off'); box(ax2, 'off'); xlim(ax2, [td(w(1)) - days(1), td(w(end)) + days(1)]);
xticks(ax2, dateshift(td(w(1)), 'start', 'month', 'next') : calmonths(1) : td(w(end)));
xtickformat(ax2, 'd MMM yyyy');
title(ax2, '(b) Volatility forecast and absolute return', 'FontWeight', 'normal');
lgd = legend(ax1, [hd hb hy], [names(ix) {'absolute return', 'outcome'}], ...
    'Orientation', 'horizontal');
lgd.Layout.Tile = 'south'; lgd.Box = 'off';
exportgraphics(fig, fullfile(tdir, 'fig_largest_gain.png'), 'Resolution', 150); close(fig);

fprintf('\nbuild finished in %.1f minutes\n', toc(t0)/60);
diary off

%% local functions
function r = cached(file, key, fun)
% load the result saved under key, or compute it and save it
if exist(file, 'file')
    s = load(file);
    if isequal(s.key, key), r = s.r; return; end
end
r = fun();
save(file, 'key', 'r', '-v7.3');
end

function r = run_chain(y, name, whole, sd)
% one whole-sample chain of ssm.models.sv, in the fields the build reads
o = sv_options(name);
res = ssm.models.sv(y, o{:}, 'nsim', whole.nsim, 'burnin', whole.burnin, 'seed', sd);
r = struct('params', res.params, 'param_names', {res.param_names}, 'sdh_mean', res.sdh_mean, ...
    'acc', res.acc);
end

function out = run_block(y, t, W, names, fc, seed)
% the forecasts of every model from the origins t, one origin per parallel iteration
nm = numel(names); out = nan(numel(t), 7, nm);
parfor i = 1:numel(t)
    ri = nan(1, 7, nm);
    for k = 1:nm
        ri(1,:,k) = fc_origin(y, t(i), W, names{k}, fc, seed + 10000*k + t(i));
    end
    out(i,:,:) = ri;
end
end

function lg = day_density(y, t, W, nms, kk, fc, seed, x)
% the log predictive densities for y(t+1) of the models nms, at x and, in the last row, at
% y(t+1); model nms{k} is model kk(k) of the build, with its seed of Part 2
yw = y(max(1, t-W+1):t); lg = nan(numel(x) + 1, numel(nms));
for k = 1:numel(nms)
    if strcmp(nms{k}, 'EWMA')
        v = ewma(yw, .94); s = sqrt(v); lp = @(z) -.5*log(2*pi*v) - .5*(z/s).^2;
    else
        o = sv_options(nms{k});
        res = ssm.models.sv(yw, o{:}, 'nsim', fc.nsim, 'burnin', fc.burnin, ...
            'seed', seed + 10000*kk(k) + t);
        f = ssm.models.sv_forecast(res); lp = f.logpdf;
    end
    lg(:,k) = lp([x; y(t+1)]);
end
end

function y1 = sim_next(res)
% y_{T+1} simulated forward from each posterior draw of ssm.models.sv
P = res.params; n = size(P, 1); m = res.spec;
g = @(s) P(:, strcmp(res.param_names, s));
mu = zeros(n,1); rho = mu; psi = mu;
if m.mean, mu = g('mu'); end
if strcmp(m.errors, 'arma'), rho = g('rho'); end
if ~strcmp(m.errors, 'iid'), psi = g('psi'); end
hT = res.last(:,1);
if strcmp(m.vol, 'ar1'), a = g('mu_h') + g('phi').*(hT - g('mu_h')); else, a = hT; end
h = a + sqrt(g('sig2_h')).*randn(n,1);
if strcmp(m.dist, 't'), nu = g('nu'); lam = 1./gamrnd(nu/2, 2./nu); else, lam = 1; end
y1 = mu + rho.*res.last(:,2) + psi.*res.last(:,3) + sqrt(lam).*exp(h/2).*randn(n,1);
end
