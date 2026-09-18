function test_surform
% Structure, and exact equality with every legacy SURform.m in this repository:
% four archived copies and the chan-jeliazkov-2009 fixture.
rng(1, 'twister');
X = randn(7, 3);
X(3, 2) = 0;                                   % a zero entry, which sparse() does not store
Z = ssm.surform(X);
assert(isequal(size(Z), [7, 21]), 'surform: wrong size');
assert(isequal(full(Z(2, 4:6)), X(2, :)), 'surform: wrong block placement');
assert(nnz(Z) == nnz(X), 'surform: wrong sparsity');

root = getappdata(0, 'ssm_repo_root');
copies = { ...
    fullfile(root, 'replications', 'chan2017_jbes_svm', 'legacy'); ...
    fullfile(root, 'replications', 'chan_clark_koop2018_jmcb_trendie', 'legacy'); ...
    fullfile(root, 'replications', 'chan_eisenstat2015_er_mlce', 'legacy', 'ML_CE'); ...
    fullfile(root, 'replications', 'chan_grant2016_csda_dic', 'legacy', 'DIC'); ...
    fullfile(root, 'tests', 'fixtures', 'chan-jeliazkov-2009')};
for ii = 1:numel(copies)
    addpath(copies{ii}); c = onCleanup(@() rmpath(copies{ii}));   % the copy shadows by name
    Zleg = feval('SURform', X);
    clear c                                    % off the path before the next copy
    assert(isequal(Zleg, Z), 'surform: differs from the SURform.m in %s', copies{ii});
end
end
