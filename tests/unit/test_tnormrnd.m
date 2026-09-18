function test_tnormrnd
% Bounds, and draw-for-draw equality under one seed with every legacy tnormrnd.m in
% this repository, for scalar and for vector arguments.
rng(4, 'twister');
t = ssm.tnormrnd(0.3, 2.0, -1, 1.5, 500);
assert(all(t >= -1 & t <= 1.5), 'tnormrnd: draws outside bounds');

N = 50;
mu = linspace(-1, 1, N)'; s2 = linspace(0.5, 2, N)'; a = mu - 1; b = mu + 0.5;
rng(5, 'twister');
tv = ssm.tnormrnd(mu, s2, a, b);
assert(all(tv >= a & tv <= b), 'tnormrnd: vector draws outside bounds');

leg = fullfile(getappdata(0, 'ssm_repo_root'), 'replications');
copies = { ...
    fullfile(leg, 'chan_clark_koop2018_jmcb_trendie', 'legacy'); ...
    fullfile(leg, 'chan_eisenstat2015_er_mlce', 'legacy', 'ML_CE'); ...
    fullfile(leg, 'chan_grant2016_eneco_garchsv', 'legacy'); ...
    fullfile(leg, 'chan_koop_potter2016_jae_boundedpc', 'legacy'); ...
    fullfile(leg, 'grant_chan2017_jedc_hpfilter', 'legacy'); ...
    fullfile(leg, 'grant_chan2017_jmcb_trendcycle', 'legacy')};
for ii = 1:numel(copies)
    addpath(copies{ii}); c = onCleanup(@() rmpath(copies{ii}));   % the copy shadows by name
    rng(4, 'twister'); t1 = feval('tnormrnd', 0.3, 2.0, -1, 1.5, 500);
    rng(5, 'twister'); tv1 = feval('tnormrnd', mu, s2, a, b);
    clear c                                    % off the path before the next copy
    assert(isequal(t1, t) && isequal(tv1, tv), ...
        'tnormrnd: differs from the tnormrnd.m in %s under the same seed', copies{ii});
end
end
