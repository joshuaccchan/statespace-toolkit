% run_examples - run setup.m, then every script in examples/, the own-data
% script of each tutorial (tutorials/*/your_data.m) and the script of each
% how-to guide (guides/*/guide.m), each in its own workspace; error if any of
% them fails.
% Usage (from repo root or anywhere):  matlab -batch "run('tests/run_examples.m')"
%
% Checks that setup.m puts the +ssm package on the path, as the README quick start
% assumes, and that every example runs to the end without an error. Scripts are
% found by name (examples/ex*.m, tutorials/*/your_data.m, guides/*/guide.m), so a
% new one is covered as soon as it is added.

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
files = fullfile(exdir, strcat(names, '.m'));
n_ex = numel(names);
tut = dir(fullfile(root, 'tutorials', '*', 'your_data.m'));
for ii = 1:numel(tut)
    [~, folder] = fileparts(tut(ii).folder);
    names{end+1} = ['tutorials/' folder '/your_data']; %#ok<AGROW>
    files{end+1} = fullfile(tut(ii).folder, tut(ii).name); %#ok<AGROW>
end
gd = dir(fullfile(root, 'guides', '*', 'guide.m'));
for ii = 1:numel(gd)
    [~, folder] = fileparts(gd(ii).folder);
    names{end+1} = ['guides/' folder '/guide']; %#ok<AGROW>
    files{end+1} = fullfile(gd(ii).folder, gd(ii).name); %#ok<AGROW>
end

failed = {};
for ii = 1:numel(names)
    fprintf('\n---- %s ----\n', names{ii});
    t0 = tic;
    try
        run_one(files{ii});
        fprintf('PASS  %s (%.0f s)\n', names{ii}, toc(t0));
    catch err
        failed{end+1} = names{ii}; %#ok<AGROW>
        fprintf('FAIL  %s\n%s\n', names{ii}, getReport(err, 'extended', 'hyperlinks', 'off'));
    end
    close all force
end

if ~isempty(failed)
    error('%d of %d scripts failed: %s', numel(failed), numel(names), strjoin(failed, ', '));
end
n_tut = numel(tut); n_gd = numel(gd);
fprintf('\nAll %d examples, %d tutorial script%s and %d guide%s ran.\n', n_ex, n_tut, ...
    repmat('s', 1, n_tut ~= 1), n_gd, repmat('s', 1, n_gd ~= 1));
end

function run_one(file)
% Each script runs in this function's workspace, so no script can see or clear
% the variables of another, or those of the loop above.
run(file);
end
