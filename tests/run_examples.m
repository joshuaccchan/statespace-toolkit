% run_examples - run setup.m, then every script in examples/, each in its own
% workspace; error if any of them fails.
% Usage (from repo root or anywhere):  matlab -batch "run('tests/run_examples.m')"
%
% Checks that setup.m puts the +ssm package on the path, as the README quick start
% assumes, and that every example runs to the end without an error. Examples are
% found by name (examples/ex*.m), so a new one is covered as soon as it is added.

function run_examples

thisdir = fileparts(mfilename('fullpath'));
root = fileparts(thisdir);

% No example needs the Optimization Toolbox, which only four archived packages
% use, so setup.m's warning about it is turned off while this runs.
wopt = warning('off', 'ssm:setup:noOptim');
restore = onCleanup(@() warning(wopt));

% Take core/ off the path first, so the check below depends on setup.m alone
% even in a session where it was already added.
w = warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(fullfile(root, 'core'));
warning(w);
run(fullfile(root, 'setup.m'));
if isempty(meta.package.fromName('ssm'))
    error('setup.m ran, but the +ssm package is not on the path');
end
fprintf('PASS  setup.m\n');

exdir = fullfile(root, 'examples');
ex = dir(fullfile(exdir, 'ex*.m'));
names = sort(erase({ex.name}, '.m'));
if isempty(names)
    error('no examples found in %s', exdir);
end

failed = {};
for ii = 1:numel(names)
    fprintf('\n---- %s ----\n', names{ii});
    t0 = tic;
    try
        run_one(fullfile(exdir, [names{ii} '.m']));
        fprintf('PASS  %s (%.0f s)\n', names{ii}, toc(t0));
    catch err
        failed{end+1} = names{ii}; %#ok<AGROW>
        fprintf('FAIL  %s\n%s\n', names{ii}, getReport(err, 'extended', 'hyperlinks', 'off'));
    end
    close all force
end

if ~isempty(failed)
    error('%d of %d examples failed: %s', numel(failed), numel(names), strjoin(failed, ', '));
end
fprintf('\nAll %d examples ran.\n', numel(names));
end

function run_one(file)
% Each example runs in this function's workspace, so no example can see or clear
% the variables of another, or those of the loop above.
run(file);
end
