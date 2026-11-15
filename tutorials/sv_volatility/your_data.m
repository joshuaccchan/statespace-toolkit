%% your_data - compare the volatility forecasts of the six models on your own series
%
% Estimates the models in `models` (the EWMA of RiskMetrics and the stochastic volatility
% models SV, SV-t, SV-RW, SV-RW-t and SV-MA) on all the returns up to each of the last nfc
% days, or on the last W returns before each, with the priors of build.m, and scores their
% forecasts of the next day's return
% by the sum of log predictive likelihoods, and the forecasts of its variance by QLIKE and
% the mean squared error against the squared return. With latest = true each model also
% forecasts the day after the last one. The defaults use the tutorial's dollar-euro rate
% and take about a minute; their chains and forecast window are far too short for the
% numbers on the page, which come from build.m. The report goes to outdir, tempdir by
% default.
%
% The series is a daily price or exchange rate, from which the script takes 100 times the
% log change (kind = 'price'), or a daily return in percent (kind = 'return'), the scale
% the priors are set for, in one column of a csv file with a date column. Rows without a
% value, such as holidays, are dropped; a series with fewer than 200 returns before the
% first origin (or W, if W is finite), a constant series, a price that is not positive and
% dates that are not increasing or not daily stop the script.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
run(fullfile(repo, 'setup.m'));
lastwarn('');

%% ---- settings ----
file = fullfile(repo, 'examples', 'data', 'USDEUR_daily.csv');
col = 'USD_EUR';                   % the series, by name
datecol = 'observation_date';
kind = 'price';                    % 'price': 100 times the log change; 'return': as it is
models = {'EWMA', 'SV', 'SV-t', 'SV-RW', 'SV-RW-t', 'SV-MA'};   % delete one to leave it out
W = Inf;                           % estimation window: Inf for every return up to the origin
nfc = 6;                           % origins: the last nfc days with a return after them
nsim = 400; burnin = 200;          % draws per model and origin
seed = 1;
latest = true;                     % also forecast the day after the last one
outdir = tempdir;                  % where the report goes; '' for none

%% ---- data ----
tbl = readtable(file);
x = tbl.(col); dates = tbl.(datecol);
keep = ~isnan(x); x = x(keep); dates = dates(keep);
assert(all(diff(dates) > 0), 'your_data: the dates are not increasing');
switch kind
    case 'price'
        assert(all(x > 0), 'your_data: %s has values that are not positive', col);
        y = 100*diff(log(x)); dates = dates(2:end);
    case 'return'
        y = x;
    otherwise
        error('your_data: kind must be ''price'' or ''return''');
end
T = numel(y);
assert(T - nfc >= min(W, 200) && std(y) > 0, 'your_data: %s is too short or constant', col);
assert(median(days(diff(dates))) <= 3, 'your_data: the dates are not daily');
if isinf(W), wl = 'all returns up to each origin'; else, wl = sprintf('the last %d returns', W); end
fprintf('\n%s, %s to %s, T = %d returns; forecasts from %d origins, estimated on %s\n', col, ...
    char(dates(1)), char(dates(end)), T, nfc, wl);

%% ---- recursive forecasts ----
orig = T - nfc : T - 1;
fc = struct('nsim', nsim, 'burnin', burnin);
nm = numel(models);
yy = y(orig + 1);
res = zeros(nm, 3);                % [LPL QLIKE MSE] by model
for k = 1:nm
    r = zeros(numel(orig), 7);
    for i = 1:numel(orig)
        r(i,:) = fc_origin(y, orig(i), W, models{k}, fc, seed + 10000*k + orig(i));
    end
    f2 = r(:,2) + r(:,1).^2;       % the forecast of the squared return
    res(k,:) = [sum(r(:,3)) mean(log(f2) + yy.^2./f2) mean((yy.^2 - f2).^2)];
end
fprintf('\n| Model | log predictive likelihood | QLIKE | MSE |\n|---|---|---|---|\n');
for k = 1:nm
    fprintf('| %s | %.1f | %.3f | %.3f |\n', models{k}, res(k,:));
end
fprintf('%d one-day-ahead forecasts; QLIKE is log(f) + y^2/f for the forecast f of y^2\n', numel(orig));

%% ---- the forecast from the last day ----
nxt = nan(nm, 4);                  % [mean volatility lower_90 upper_90] by model
if latest
    fprintf('\nThe day after %s, from %s:\n', char(dates(end)), strrep(wl, 'each origin', 'the last day'));
    fprintf('| Model | mean | volatility | 90%% interval |\n|---|---|---|---|\n');
    for k = 1:nm
        r = fc_origin(y, T, W, models{k}, fc, seed + 10000*k + T);
        nxt(k,:) = [r(1) sqrt(r(2)) r(5) r(6)];
        fprintf('| %s | %.3f | %.3f | (%.3f, %.3f) |\n', models{k}, nxt(k,:));
    end
end

%% ---- the report ----
if ~isempty(outdir)
    if ~isfolder(outdir), mkdir(outdir); end
    report = array2table([res nxt], 'VariableNames', {'log_predictive_likelihood', 'qlike', ...
        'mse', 'next_mean', 'next_volatility', 'next_lower_90', 'next_upper_90'});
    report = [table(models', 'VariableNames', {'model'}) report];
    [msg, id] = lastwarn;
    meta = struct('file', file, 'column', col, 'kind', kind, 'first', dates(1), ...
        'last', dates(end), 'T', T, 'window', W, 'origins', dates(orig), 'nsim', nsim, ...
        'burnin', burnin, 'seed', seed, 'latest', latest, 'run_time', datetime('now'), ...
        'matlab_version', version, 'last_warning', msg, 'last_warning_id', id);
    writetable(report, fullfile(outdir, 'sv_volatility_report.csv'));
    save(fullfile(outdir, 'sv_volatility_report.mat'), 'report', 'meta');
    fprintf('\nreport written to %s and .mat\n', fullfile(outdir, 'sv_volatility_report.csv'));
end
