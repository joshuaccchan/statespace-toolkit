function test_twins
% Each twin in core/+ssm/ must equal its bvar-toolkit original, held verbatim in
% tests/fixtures/bvar-toolkit/, byte for byte from the function line on. The one
% permitted difference is diffmat's error identifiers.
root = getappdata(0, 'ssm_repo_root');
fx = fullfile(root, 'tests', 'fixtures', 'bvar-toolkit', 'core', '+bvar');
pairs = { ...
    'surform',     fullfile(fx, '+util', 'surform.m'),     {}; ...
    'tnormrnd',    fullfile(fx, '+util', 'tnormrnd.m'),    {}; ...
    'shaded_band', fullfile(fx, '+util', 'shaded_band.m'), {}; ...
    'ksc_rw_h0',   fullfile(fx, '+sv', 'ksc_rw_h0.m'),     {}; ...
    'diffmat',     fullfile(fx, '+util', 'diffmat.m'),     {'ssm:diffmat:', 'bvar:util:diffmat:'}};
for ii = 1:size(pairs, 1)
    mine = body(fullfile(root, 'core', '+ssm', [pairs{ii,1} '.m']));
    subs = pairs{ii,3};
    if ~isempty(subs)
        assert(contains(mine, subs{1}), 'twins: ssm.%s has no %s identifiers', pairs{ii,1}, subs{1});
        mine = strrep(mine, subs{1}, subs{2});
    end
    assert(strcmp(mine, body(pairs{ii,2})), ...
        'twins: ssm.%s differs from its bvar-toolkit twin', pairs{ii,1});
end
end

function b = body(file)
% the file's text from its first function line on
txt = fileread(file);
k = regexp(txt, '(?m)^function', 'once');
assert(~isempty(k), 'twins: no function line in %s', file);
b = txt(k:end);
end
