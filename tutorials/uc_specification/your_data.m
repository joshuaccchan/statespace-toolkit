%% your_data - compare the seven inflation models on your own series
%
% Estimates the models in `models` (an AR(4), UC, UC-SVgap, UCSV, UC-MA, the bounded
% trend model of Chan, Koop and Potter (2013) and M1 of Chan, Clark and Koop (2018))
% on one quarterly series, with the priors of build.m, recursively over the last nfc
% origins at which the next four quarters are observed, and scores the forecasts of
% the average inflation over those four quarters by the root mean squared forecast
% error and the sum of log predictive likelihoods. With latest = true each model also
% forecasts the average over the four quarters after the last one, from all the data.
% M1 (CCK) also needs a long-run inflation expectation; with zfile = '' it is left
% out. The defaults use the tutorial's quarterly CPI inflation and the SPF 10-year CPI
% expectation and take about a minute; their chains and forecast window are far too
% short for the numbers on the page, which come from build.m. The report goes to
% outdir, tempdir by default.
%
% The series is an inflation rate in annualized percent, the scale the priors are set
% for, in one column of a csv file with a date column, one row per quarter. Its first
% value is the presample of the bounded model, whose trend lies between 0 and 5.
% Missing values at either end are dropped; a missing value inside the sample, a
% constant series or dates that are not quarterly stop the script, and so does a new
% file with zfile still the US SPF expectation. The expectation, in percent per year,
% is matched to the series by date; its first quarter is the presample of M1, and it
% may start later than the series but must have no gap after that.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
run(fullfile(repo, 'setup.m'));
lastwarn('');

%% ---- settings ----
file = fullfile(repo, 'examples', 'data', 'USCPI_quarterly.csv');
col = 'inflation';                 % the series, by name
datecol = 'observation_date';
zfile = fullfile(repo, 'examples', 'data', 'USCPI10_SPF.csv');   % '' to leave out CCK
zcol = 'CPI10';
models = {'AR4',      'AR(4)'      % delete a row to leave a model out
          'UC',       'UC'
          'UC-SVgap', 'UC-SVgap'
          'UCSV',     'UCSV'
          'UC-MA',    'UC-MA'
          'ARbound',  'AR-trend-bound'
          'CCK',      'CCK'};
nfc = 6;                           % origins: the last nfc with four quarters after them
nsim = 1000; burnin = 500;         % draws per model and origin
seed = 1;
latest = true;                     % also forecast the four quarters after the last one
outdir = tempdir;                  % where the report goes; '' for none

%% ---- data ----
cpi = fullfile(repo, 'examples', 'data', 'USCPI_quarterly.csv');
spf = fullfile(repo, 'examples', 'data', 'USCPI10_SPF.csv');
assert(~strcmp(zfile, spf) || strcmp(file, cpi), ['your_data: zfile is the US SPF ' ...
    'expectation, which goes with the US CPI series; set zfile to the expectation that ' ...
    'matches your series, or to '''' to leave out CCK']);
tbl = readtable(file);
y = tbl.(col); dates = tbl.(datecol);
keep = ~isnan(y);
assert(any(keep), 'your_data: %s has no values', col);
lo = find(keep, 1); hi = find(keep, 1, 'last');
assert(all(keep(lo:hi)), 'your_data: %s has missing values inside the sample', col);
y = y(lo:hi); dates = dates(lo:hi);
T = numel(y);
assert(T > nfc + 44 && std(y) > 0, 'your_data: %s is too short or constant', col);
gap = days(diff(dates));
assert(all(gap >= 89 & gap <= 93), 'your_data: the dates are not one quarter apart');
if isempty(zfile), models(strcmp(models(:,1), 'CCK'), :) = []; end
names = models(:,1)'; labels = models(:,2)';
z = nan(T, 1);
if ~isempty(zfile)
    zt = readtable(zfile);
    [~, ia, ib] = intersect(dates, zt.(datecol));
    z(ia) = zt.(zcol)(ib);
    i0 = find(~isnan(z), 1);
    assert(~isempty(i0) && all(~isnan(z(i0:end))), ...
        'your_data: %s must cover the end of the sample without a gap', zcol);
    assert(T - 3 - nfc - i0 > 20, 'your_data: %s starts too late for the first origin', zcol);
end
qlab = @(d) sprintf('%dQ%d', year(d), quarter(d));
fprintf('\n%s, %s to %s, T = %d; forecasts from %d origins\n', col, qlab(dates(1)), ...
    qlab(dates(end)), T, nfc);

%% ---- recursive forecasts ----
orig = T - 3 - nfc : T - 4;
fc = struct('nsim', nsim, 'burnin', burnin);
nm = numel(names);
y4 = arrayfun(@(t) mean(y(t+1:t+4)), orig)';
res = zeros(nm, 2);                % [RMSFE LPL] by model
for k = 1:nm
    r = zeros(numel(orig), 6);
    for i = 1:numel(orig)
        r(i,:) = fc_origin(repo, y, z, orig(i), names{k}, fc, seed + 1000*k + orig(i));
    end
    res(k,:) = [sqrt(mean((r(:,3) - y4).^2)) sum(r(:,4))];
end
fprintf('\n| Model | RMSFE | log predictive likelihood |\n|---|---|---|\n');
for k = 1:nm
    fprintf('| %s | %.2f | %.1f |\n', labels{k}, res(k,:));
end
fprintf('%d forecasts of the average over the next four quarters\n', numel(orig));

%% ---- the forecast from the last quarter ----
fut = nan(nm, 3);                  % [point lower_90 upper_90] by model
if latest
    nxt = dates(end) + calmonths(3*(1:4));
    fprintf('\nThe average over %s to %s, from all %d quarters:\n', qlab(nxt(1)), qlab(nxt(4)), T);
    fprintf('| Model | point forecast | 90%% interval |\n|---|---|---|\n');
    for k = 1:nm
        r = fc_origin(repo, y, z, T, names{k}, fc, seed + 1000*k + T);
        fut(k,:) = r([3 5 6]);
        fprintf('| %s | %.2f | (%.2f, %.2f) |\n', labels{k}, fut(k,:));
    end
end

%% ---- the report ----
if ~isempty(outdir)
    report = array2table([res fut], 'VariableNames', {'rmsfe', 'log_predictive_likelihood', ...
        'forecast_next4', 'forecast_lower_90', 'forecast_upper_90'});
    report = [table(labels', names', 'VariableNames', {'model', 'id'}) report];
    [msg, id] = lastwarn;
    meta = struct('file', file, 'column', col, 'zfile', zfile, 'zcolumn', zcol, ...
        'first', dates(1), 'last', dates(end), 'T', T, 'origins', dates(orig), ...
        'nsim', nsim, 'burnin', burnin, 'seed', seed, 'latest', latest, ...
        'run_time', datetime('now'), 'matlab_version', version, 'last_warning', msg, ...
        'last_warning_id', id);
    writetable(report, fullfile(outdir, 'uc_specification_report.csv'));
    save(fullfile(outdir, 'uc_specification_report.mat'), 'report', 'meta');
    fprintf('\nreport written to %s and .mat\n', fullfile(outdir, 'uc_specification_report.csv'));
end
