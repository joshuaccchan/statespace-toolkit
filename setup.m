% setup.m - put statespace-toolkit on the MATLAB path for this session.
%
%   run setup.m          (from anywhere; the script locates the repo itself)
%
% Adds the repo's core/ folder to the path. core/ contains the MATLAB package
% folder +ssm, so functions are called with the package prefix, ssm.<name>.
% The Econometrics Toolbox has a class of the same name: dotted calls resolve
% to this package, and ssm(...) still constructs the toolbox's state space
% object.
%
% The path change lasts for the session. Add this line to your own startup.m
% to have it every time:
%
%   run('<path-to-repo>/setup.m')
%
% Replication drivers are not added to the path: each lives beside the package
% it runs, under replications/<slug>/, and expects that folder as the working
% directory so that its legacy data files resolve.

ssm_root = fileparts(mfilename('fullpath'));
addpath(fullfile(ssm_root, 'core'));

if isempty(ver('stats'))
    warning('ssm:setup:noStats', ...
        ['The Statistics and Machine Learning Toolbox is not available. ' ...
         'Every archived package calls gamrnd.']);
end
if isempty(ver('optim'))
    warning('ssm:setup:noOptim', ...
        ['The Optimization Toolbox is not available. Four archived packages ' ...
         'call fminunc: those of Chan (2013), Chan and Eisenstat (2015), ' ...
         'Chan, Clark and Koop (2018), and Chan and Song (2018).']);
end

fprintf('statespace-toolkit: core/ added to the path (%s).\n', ssm_root);
fprintf('  run tests:  run(fullfile(''%s'',''tests'',''unit'',''run_unit_tests.m''))\n', ssm_root);

clear ssm_root
