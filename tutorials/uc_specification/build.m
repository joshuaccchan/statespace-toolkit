%% build - regenerate the figures and numbers of tutorials/uc_specification/README.md
%
% Seven models of quarterly US CPI inflation, 1948Q1-2025Q3
% (examples/data/USCPI_quarterly.csv), with common priors for the parameters the
% unobserved components models share: an AR(4) benchmark; UC, UC-SVgap (UC with stochastic
% volatility in the gap), UCSV and UC-MA; the bounded trend model of Chan, Koop and Potter
% (2013); and M1 of Chan, Clark and Koop (2018), which adds the SPF 10-year CPI
% expectation from 1991Q4 (examples/data/USCPI10_SPF.csv). Each model is re-estimated on
% the data up to each origin, 1999Q4 to 2024Q3, and its forecast of the average inflation
% over the next four quarters is scored by the root mean squared forecast error and the
% log predictive likelihood, as in the forecasting code of Chan, Koop and Potter (2016)
% and Chan, Clark and Koop (2018). Part 0 checks the predictive densities; Part 1 runs one
% chain per model on the whole sample; Part 2 runs the recursive forecasts, saved in runs/
% so that an interrupted build resumes; Part 3 prints the table and draws the figures,
% written next to this file. Everything printed goes to a log in tempdir.
%
% Usage, from anywhere:  run tutorials/uc_specification/build.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
logf = fullfile(tempdir, 'ssm_uc_specification_build_log.txt');
if exist(logf, 'file'), delete(logf); end
diary(logf);
fprintf('build.m of tutorials/uc_specification, %s, MATLAB %s\n', char(datetime('now')), version);
t0 = tic;
run(fullfile(repo, 'setup.m'));

% settings
seed = 20260922;
names = {'UC', 'UC-MA', 'UCSV', 'ARbound', 'CCK', 'AR4', 'UC-SVgap'};
labels = {'UC', 'UC-MA', 'UCSV', 'AR-trend-bound', 'CCK', 'AR(4)', 'UC-SVgap'};
order = [6 1 7 3 2 4 5];                        % the order of the table
isuc = [1 2 3 7];                               % the models of uc_sampler
nsim = 50000; burnin = 5000; thin = 10;         % whole sample, the models of uc_sampler
ckp = struct('nloop', 35000, 'burnin', 5000);   % whole sample, the archived run length
fc = repmat({struct('nsim', 20000, 'burnin', 5000)}, 1, 7);   % each origin
fc{5} = struct('nsim', 10000, 'burnin', 1000);  % CCK
first_target = 2000;                            % 2000Q1
ver = [2 3 2 2 2 2 2];                          % cache version by model, in every key:
                                                % 2 the common priors, 3 UC-MA's u_T fix

data = readtable(fullfile(repo, 'examples', 'data', 'USCPI_quarterly.csv'));
y = data.inflation; T = numel(y);
assert(T == 311 && abs(y(1) - 8.3865) < 1e-4 && abs(y(end) - 3.0426) < 1e-4 && ...
    data.observation_date(1) == datetime(1948,1,1) && data.observation_date(end) == datetime(2025,7,1), ...
    'build: examples/data/USCPI_quarterly.csv is not the file this page was built from');
spf = readtable(fullfile(repo, 'examples', 'data', 'USCPI10_SPF.csv'));
z = nan(T, 1);
[~, ia, ib] = intersect(data.observation_date, spf.observation_date);
z(ia) = spf.CPI10(ib);
i0 = find(~isnan(z), 1);
assert(data.observation_date(i0) == datetime(1991,10,1) && all(~isnan(z(i0:end))) && ...
    abs(z(i0) - 4) < 1e-12 && abs(z(end) - 2.305) < 1e-12, ...
    'build: examples/data/USCPI10_SPF.csv is not the file this page was built from');
tid = 1948 + (0:T-1)'/4;
qlab = @(i) sprintf('%dQ%d', floor(tid(i)), round(4*(tid(i) - floor(tid(i)))) + 1);
lme = @(x) max(x) + log(mean(exp(x - max(x))));
lnpdf = @(x, m, v) -.5*log(2*pi*v) - .5*(x - m).^2./v;

%% Part 0: checks
fprintf('\n=== Part 0: checks ===\n');
% (a) UC at fixed variances: the predictive density of y_{T+1}, averaged over draws
% of tau_T from ssm.simulate_states, against the exact value from ssm.intlike
s2 = [1.5 .1]; n = T - 1; M = 20000;
H = ssm.diffmat(n);
P = H'*spdiags([1/(100 + s2(2)); ones(n-1,1)/s2(2)], 0, n, n)*H;
K = speye(n)/s2(1) + P;
bK = y(1:n)/s2(1) + P*(5*ones(n,1));
rng(seed, 'twister');
tauT = zeros(M, 1);
for i = 1:M, s = ssm.simulate_states(K, bK); tauT(i) = s(n); end
out = struct('phi', repmat(log(s2), M, 1), 'last', [tauT repmat(log(s2), M, 1) zeros(M,1)]);
f = uc_forecast(uc_model('UC'), out);
lw = lnpdf(y(T), f.m1, f.v1);
se = std(exp(lw - max(lw)))/mean(exp(lw - max(lw)))/sqrt(M);
Hn = ssm.diffmat(T);
Pn = Hn'*spdiags([1/(100 + s2(2)); ones(T-1,1)/s2(2)], 0, T, T)*Hn;
ex = ssm.intlike(y, speye(T), speye(T)/s2(1), Pn, 5*ones(T,1)) ...
    - ssm.intlike(y(1:n), speye(n), speye(n)/s2(1), P, 5*ones(n,1));
assert(abs(lme(lw) - ex) < 4*se + 1e-3, 'build: the UC predictive density differs from ssm.intlike');
fprintf('(a) UC, log p(y_T | y_1:T-1) at fixed variances: %.4f (%.4f) against ssm.intlike %.4f\n', ...
    lme(lw), se, ex);

% (b) every model: the predictive mixtures of both targets against paths of y
% simulated forward from the same posterior draws, compared by their distribution
% functions at the 5th, ..., 95th percentiles of the simulated values
R = 50; p = (.05:.05:.95)';
for k = 1:7
    rng(seed + k, 'twister');
    switch k
        case {1, 2, 3, 7}
            m = uc_model(names{k});
            out = uc_sampler(y, m, 2000, 1000, 10);
            fk = @() uc_forecast(m, out); sk = @() sim_uc(m, out);
        case 4
            out = ckp_run(repo, y, 3000, 1000, Inf, seed + k);
            fk = @() ckp_forecast(out, y(T)); sk = @() sim_ckp(out, y(T));
        case 5
            out = cck_sampler(y(i0+1:T), y(i0), z(i0+1:T), 2000, 1000);
            fk = @() cck_forecast(out, y(T)); sk = @() sim_cck(out, y(T));
        case 6
            f6 = ar_model(y, 4, 2000, 1000);
            fk = @() f6; sk = @() sim_ar(f6, y);
    end
    [F, S] = deal(cell(R, 1));
    for r = 1:R, F{r} = fk(); S{r} = sk(); end
    F = [F{:}]; S = [S{:}];
    d = zeros(1, 2);
    for h = 1:2
        mm = vertcat(F.(sprintf('m%d', 3*h - 2))); vv = vertcat(F.(sprintf('v%d', 3*h - 2)));
        q = quantile(vertcat(S.(sprintf('y%d', 3*h - 2))), p);
        Fq = arrayfun(@(x) mean(normcdf((x - mm)./sqrt(vv))), q);
        d(h) = max(abs(Fq - p));
    end
    assert(all(d < .01), 'build: the predictive density of %s differs from simulated paths', labels{k});
    fprintf('(b) %-14s largest difference in the distribution function: next quarter %.4f, four-quarter average %.4f\n', ...
        labels{k}, d);
end

%% Part 1: the whole sample
fprintf('\n=== Part 1: the whole sample ===\n');
rdir = fullfile(tdir, 'runs');
if ~exist(rdir, 'dir'), mkdir(rdir); end
ch = cell(1, 7);
for k = isuc
    m = uc_model(names{k});
    ch{k} = cached(fullfile(rdir, [strrep(names{k}, '-', '_') '_chain.mat']), [nsim burnin thin seed + k ver(k)], ...
        @() run_chain(y, m, nsim, burnin, thin, seed + k));
    fprintf('%-14s %d draws after %d burn-in\n', labels{k}, nsim, burnin);
end
% UC-MA forecasts from u_T: it must be the error of the stored tau under the stored psi
i = thin:thin:nsim; ps = tanh(ch{2}.phi(i, end)); e = zeros(numel(i), 1);
for j = 1:numel(i)
    u = filter(1, [1 ps(j)], y - ch{2}.tau(j,:)');
    e(j) = abs(u(T) - ch{2}.last(i(j), 4));
end
assert(max(e) < 1e-8, 'build: UC-MA''s u_T does not match its tau and psi');
fprintf('UC-MA acceptance: psi %.2f, AR coefficient of the log-volatility %.2f\n', ...
    ch{2}.acc.psi, ch{2}.acc.gap_phi);
ch{4} = cached(fullfile(rdir, 'ARbound_chain.mat'), [ckp.nloop ckp.burnin thin seed + 4 ver(4)], ...
    @() ckp_run(repo, y, ckp.nloop, ckp.burnin, thin, seed + 4));
ch{4}.phi = log(ch{4}.sig);
fprintf('%-14s %d draws after %d burn-in; acceptance of tau, rho, h %s\n', labels{4}, ...
    ckp.nloop - ckp.burnin, ckp.burnin, mat2str(ch{4}.acc, 2));
ch{5} = cached(fullfile(rdir, 'CCK_chain.mat'), [nsim burnin seed + 5 ver(5)], ...
    @() run_cck(y(i0+1:T), y(i0), z(i0+1:T), nsim, burnin, seed + 5));
fprintf('%-14s %d draws after %d burn-in, 1992Q1-2025Q3; acceptance of the blocks of b, psi, sigb2 %s\n', ...
    labels{5}, nsim, burnin, mat2str(ch{5}.acc, 2));
ps = tanh(ch{2}.phi(:, end)); q = quantile(ps, [.05 .95]);
fprintf('UC-MA, MA coefficient psi: posterior mean %.3f, 90%% interval (%.3f, %.3f)\n', mean(ps), q);
fprintf('Inefficiency factors (Bartlett, 100 lags) of the free parameters:\n');
for k = [isuc 4]
    fprintf('%-14s %s\n', labels{k}, mat2str(ineff(ch{k}.phi, 100), 3));
end
fprintf('%-14s %s\n', labels{5}, mat2str(ineff(ch{5}.theta, 100), 3));

%% Part 2: the recursive forecasts
fprintf('\n=== Part 2: recursive forecasts ===\n');
orig = find(tid == first_target) - 1 : T - 4;    % the origins, 1999Q4 to 2024Q3
no = numel(orig);
sc = cell(1, 7);
for k = 1:7
    t1 = tic;
    sc{k} = nan(no, 6);                          % [point1 lpl1 point4 lpl4 lo4 hi4]
    for i = 1:no
        t = orig(i); sd = seed + 1000*k + t;
        sc{k}(i,:) = cached(fullfile(rdir, sprintf('fc_%s_%d.mat', strrep(names{k}, '-', '_'), t)), ...
            [fc{k}.nsim fc{k}.burnin sd ver(k)], @() fc_origin(repo, y, z, t, names{k}, fc{k}, sd));
    end
    fprintf('%-14s %d origins, %s to %s, %d draws after %d, %.1f minutes\n', labels{k}, no, ...
        qlab(orig(1)), qlab(orig(end)), fc{k}.nsim, fc{k}.burnin, toc(t1)/60);
end

%% Part 3: table and figures
fprintf('\n=== Part 3: table and figures ===\n');
y4 = arrayfun(@(t) mean(y(t+1:t+4)), orig)';
rm = @(k) sqrt(mean((sc{k}(:,3) - y4).^2));
fprintf('%d forecasts of the average inflation over the next four quarters, %s-%s to %s-%s\n', ...
    no, qlab(orig(1)+1), qlab(orig(1)+4), qlab(orig(end)+1), qlab(orig(end)+4));
fprintf(['| Model | RMSFE | RMSFE relative to AR(4) | log predictive likelihood | ' ...
    'difference in log predictive score against AR(4) |\n|---|---|---|---|---|\n']);
for k = order
    fprintf('| %s | %.2f | %.2f | %.1f | %.1f |\n', labels{k}, rm(k), rm(k)/rm(6), ...
        sum(sc{k}(:,4)), sum(sc{k}(:,4) - sc{6}(:,4)));
end
fprintf('\nMean absolute change in the point forecast from one origin to the next:\n');
for k = order
    fprintf('%-14s %.3f\n', labels{k}, mean(abs(diff(sc{k}(:,3)))));
end
fprintf('\n90%% predictive interval: share of the outcomes inside it, average width\n');
for k = order
    fprintf('%-14s %.2f %.2f\n', labels{k}, mean(y4 >= sc{k}(:,5) & y4 <= sc{k}(:,6)), ...
        mean(sc{k}(:,6) - sc{k}(:,5)));
end

% every forecast, for reuse: forecasts.csv next to this file
nm = numel(order); ok = repmat(orig(:), nm, 1);
fcsv = table(reshape(repmat(labels(order), no, 1), [], 1), ...
    arrayfun(qlab, ok, 'UniformOutput', false), arrayfun(qlab, ok + 1, 'UniformOutput', false), ...
    arrayfun(qlab, ok + 4, 'UniformOutput', false), repmat(y4, nm, 1), ...
    'VariableNames', {'model', 'origin', 'target_first', 'target_last', 'realized'});
S = vertcat(sc{order});
fcsv.point_forecast = S(:,3); fcsv.lower_90 = S(:,5); fcsv.upper_90 = S(:,6);
fcsv.log_predictive_density = S(:,4);
writetable(fcsv, fullfile(tdir, 'forecasts.csv'));
fprintf('forecasts.csv: %d rows written\n', height(fcsv));

% the trend under the six trend models, whole sample
cols = lines(7);
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 14]);
panels = {[1 7 3], [2 4 5]};
for j = 1:2
    subplot(2, 1, j); hold on
    hy = plot(tid, y, 'Color', [.75 .75 .75]); hm = gobjects(1, 3);
    for i = 1:3
        k = panels{j}(i);
        if k == 5
            hm(i) = plot(tid(i0+1:T), ch{5}.pistar, 'Color', cols(k,:), 'LineWidth', 1.3);
        else
            tk = tid(end-size(ch{k}.tau, 2)+1:end);   % the bounded model starts in 1948Q2
            hm(i) = plot(tk, mean(ch{k}.tau)', 'Color', cols(k,:), 'LineWidth', 1.3);
        end
    end
    hold off; box off; xlim([tid(1) tid(end)]); ylim([-10 16]);
    legend([hy hm], [{'CPI inflation'} labels(panels{j})], 'Location', 'northeast', 'NumColumns', 2); legend boxoff
end
exportgraphics(fig, fullfile(tdir, 'fig_trend.png'), 'Resolution', 150); close(fig);

% cumulative log predictive likelihood against AR(4), four-quarter average
fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 9]); hold on
yline(0, 'Color', [.6 .6 .6]);
hc = gobjects(1, 6);
for i = 2:7
    k = order(i);
    hc(i-1) = plot(tid(orig) + .25, cumsum(sc{k}(:,4) - sc{6}(:,4)), 'Color', cols(k,:), 'LineWidth', 1.3);
end
hold off; box off; xlim(tid(orig([1 end]))' + .25);
legend(hc, labels(order(2:end)), 'Location', 'northwest'); legend boxoff
exportgraphics(fig, fullfile(tdir, 'fig_score.png'), 'Resolution', 150); close(fig);

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

function r = run_chain(y, m, nsim, burnin, thin, sd)
rng(sd, 'twister');
r = uc_sampler(y, m, nsim, burnin, thin);
end

function r = run_cck(pi, pi0, z, nsim, burnin, sd)
rng(sd, 'twister');
r = cck_sampler(pi, pi0, z, nsim, burnin);
end

function s = sim_uc(m, out)
% one path of y_{T+1}, ..., y_{T+4} per posterior draw of uc_sampler, by forward
% simulation of every state; returns y_{T+1} and the four-quarter average
n = size(out.phi, 1); ph = out.phi; j = 0;
x = cell(1, 2); xT = {out.last(:,2), out.last(:,3)}; c = {'gap', 'trend'};
for i = 1:2
    k = m.(c{i}); x{i} = zeros(n, 4); prev = xT{i};
    for q = 1:4
        switch k.type
            case 'const', prev = ph(:,j+1);
            case 'rw',    prev = prev + sqrt(exp(ph(:,j+2))).*randn(n,1);
            case 'ar1',   prev = ph(:,j+1) + tanh(ph(:,j+2)).*(prev - ph(:,j+1)) + sqrt(exp(ph(:,j+3))).*randn(n,1);
        end
        x{i}(:,q) = prev;
    end
    j = j + struct('const', 1, 'rw', 2, 'ar1', 3).(k.type);
end
if m.ma, psi = tanh(ph(:,end)); else, psi = zeros(n,1); end
tau = out.last(:,1); u = out.last(:,4); yy = zeros(n, 4);
for q = 1:4
    tau = tau + sqrt(exp(x{2}(:,q))).*randn(n,1);
    un = sqrt(exp(x{1}(:,q))).*randn(n,1);
    yy(:,q) = tau + un + psi.*u; u = un;
end
s = struct('y1', yy(:,1), 'y4', mean(yy, 2));
end

function s = sim_ckp(r, yT)
% the same for the bounded model
n = size(r.sig, 1);
tau = r.last(:,1); rho = r.last(:,2); h = r.last(:,3); c = yT - tau; yy = zeros(n, 4);
for q = 1:4
    tau = ssm.tnormrnd(tau, r.sig(:,1), 0, 5);
    rho = ssm.tnormrnd(rho, r.sig(:,2), 0, 1);
    h = h + sqrt(r.sig(:,3)).*randn(n,1);
    c = rho.*c + exp(h/2).*randn(n,1);
    yy(:,q) = tau + c;
end
s = struct('y1', yy(:,1), 'y4', mean(yy, 2));
end

function s = sim_cck(out, piT)
% the same for M1 of Chan, Clark and Koop (2018)
n = size(out.last, 1); th = out.theta;
ps = out.last(:,1); b = out.last(:,2); lv = out.last(:,3); ln = out.last(:,4);
c = piT - ps; yy = zeros(n, 4);
for q = 1:4
    lv = lv + sqrt(th(:,10)).*randn(n,1);
    ln = ln + sqrt(th(:,11)).*randn(n,1);
    b = ssm.tnormrnd(b, th(:,8), 0, 1);
    ps = ps + exp(ln/2).*randn(n,1);
    c = b.*c + exp(lv/2).*randn(n,1);
    yy(:,q) = ps + c;
end
s = struct('y1', yy(:,1), 'y4', mean(yy, 2));
end

function s = sim_ar(f, y)
% the same for the AR(4)
n = size(f.beta, 1); p = size(f.beta, 2) - 1;
lag = repmat(flipud(y(end-p+1:end))', n, 1); yy = zeros(n, 4);
for q = 1:4
    yy(:,q) = f.beta(:,1) + sum(f.beta(:,2:end).*lag, 2) + sqrt(f.sig2).*randn(n,1);
    lag = [yy(:,q) lag(:,1:end-1)];
end
s = struct('y1', yy(:,1), 'y4', mean(yy, 2));
end
