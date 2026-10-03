function r = ckp_run(repo, y, nloop, burnin, thin, seed)
% ckp_run - the archived sampler of Chan, Koop and Potter (2013),
% replications/chan_koop_potter2013_jbes_trendbound/legacy/ARtrendbound/ARtrend_bound.m,
% run whole on y, whose first value is the presample y0. The copy is patched only to
% read y, to set the run length, to drop the seed from the clock (rng(seed) is set
% before the script, whose initial values are drawn first) and to stop before the
% plots; each patch must match exactly once. Returns the draws of the variances
% [sigtau sigrho sigh], the draws of [tau_T rho_T h_T] that the forecasts start
% from, every thin-th draw of the paths tau, rho and h (none when thin = Inf), and
% the acceptance rates.
file = fullfile(repo, 'replications', 'chan_koop_potter2013_jbes_trendbound', 'legacy', ...
    'ARtrendbound', 'ARtrend_bound.m');
txt = fileread(file);
subs = {'clear; clc;', ''; ...
        'nloop = 35000;', 'nloop = ckp_nloop;'; ...
        'burnin = 5000;', 'burnin = ckp_burnin;'; ...
        'load ''USCPI_Q.csv'';', 'USCPI_Q = ckp_y;'; ...
        'randn(''seed'',sum(clock*100)); rand(''seed'',sum(clock*1000));', ''};
for k = 1:size(subs, 1)
    assert(numel(strfind(txt, subs{k,1})) == 1, 'ckp_run: ARtrend_bound.m has changed');
    txt = strrep(txt, subs{k,1}, subs{k,2});
end
k = strfind(txt, '%% plot graphs');
assert(isscalar(k), 'ckp_run: ARtrend_bound.m has changed');
txt = txt(1:k-1);
ckp_y = y; ckp_nloop = nloop; ckp_burnin = burnin; %#ok<NASGU>
rng(seed, 'twister');                         % the initial draws come before the seed line
evalc(txt);
keep = thin:thin:(nloop - burnin);
r = struct('sig', store_sig, 'last', [store_tau(:,end) store_rho(:,end) store_h(:,end)], ...
    'tau', store_tau(keep,:), 'rho', store_rho(keep,:), 'h', store_h(keep,:), ...
    'acc', [counttau countrho counth]/nloop);
end
