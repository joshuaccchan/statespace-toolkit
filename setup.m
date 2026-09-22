% setup.m - put statespace-toolkit on the MATLAB path for this session.
%
%   run setup.m                    (from the repository root)
%   run('<path-to-repo>/setup.m')  (from any other folder)
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
% The archived packages under replications/<slug>/legacy/ are not added to the
% path: run each entry script from its own folder, so that its data files and
% functions resolve.

ssm_root = fileparts(mfilename('fullpath'));
addpath(fullfile(ssm_root, 'core'));

if isempty(ver('stats'))
    warning('ssm:setup:noStats', ...
        ['The Statistics and Machine Learning Toolbox is not available. ' ...
         'ssm.tnormrnd, ssm.ksc_rw_h0 and most of the examples need it, ' ...
         'and every archived package calls gamrnd.']);
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
