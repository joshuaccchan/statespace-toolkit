function test_twins
% Each twin in core/+ssm/ must equal its bvar-toolkit original, held verbatim in
% tests/fixtures/bvar-toolkit/, byte for byte from the function line on. The
% permitted differences are listed per function, each with the number of times
% it must occur; the only one is diffmat's error identifiers.
root = getappdata(0, 'ssm_repo_root');
fx = fullfile(root, 'tests', 'fixtures', 'bvar-toolkit', 'core', '+bvar');
pairs = { ...
    'surform',     fullfile(fx, '+util', 'surform.m'),     {}; ...
    'tnormrnd',    fullfile(fx, '+util', 'tnormrnd.m'),    {}; ...
    'shaded_band', fullfile(fx, '+util', 'shaded_band.m'), {}; ...
    'ksc_rw_h0',   fullfile(fx, '+sv', 'ksc_rw_h0.m'),     {}; ...
    'ksc_rw_diffuse', fullfile(fx, '+sv', 'ksc_rw_diffuse.m'), {}; ...
    'diffmat',     fullfile(fx, '+util', 'diffmat.m'), ...
        {'ssm:diffmat:', 'bvar:util:diffmat:', 2}};
for ii = 1:size(pairs, 1)
    mine = body(fullfile(root, 'core', '+ssm', [pairs{ii,1} '.m']));
    subs = pairs{ii,3};
    for s = 1:size(subs, 1)
        assert(count(mine, subs{s,1}) == subs{s,3}, ...
            'twins: ssm.%s must contain "%s" exactly %d times', pairs{ii,1}, subs{s,1}, subs{s,3});
        mine = strrep(mine, subs{s,1}, subs{s,2});
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
