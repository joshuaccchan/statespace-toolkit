function ssmtest_out = run_anchor(ssmtest_a, ssmtest_patches, ssmtest_tag)
% Copy an anchor script's files to a fresh folder, patch the script with
% patch_lines, run it whole, and return the variables named in ssmtest_a.out,
% with the state of the random number stream afterwards in the field rngstate.
% ssmtest_a has fields dir, files, script, seed (empty: the script seeds itself)
% and out. Variable names here are prefixed so that the script, which runs in
% this workspace, cannot overwrite them.
ssmtest_dir = tempname; mkdir(ssmtest_dir);
ssmtest_clean = onCleanup(@() rmdir(ssmtest_dir, 's'));
for ssmtest_f = ssmtest_a.files
    copyfile(fullfile(ssmtest_a.dir, ssmtest_f{1}), ssmtest_dir);
end
ssmtest_txt = patch_lines(fileread(fullfile(ssmtest_a.dir, ssmtest_a.script)), ...
    ssmtest_patches, ssmtest_a.script);
% a new file name per run: MATLAB can run a stale copy of a file rewritten within
% the same second
ssmtest_file = fullfile(ssmtest_dir, ['anchor_' erase(ssmtest_a.script, '.m') '_' ssmtest_tag '.m']);
ssmtest_fid = fopen(ssmtest_file, 'w');
fwrite(ssmtest_fid, ssmtest_txt);
fclose(ssmtest_fid);
if ~isempty(ssmtest_a.seed)
    rng(ssmtest_a.seed, 'twister');
end
evalc('run(ssmtest_file)');
ssmtest_state = rng;
close all force
ssmtest_out = struct('rngstate', ssmtest_state);
for ssmtest_v = ssmtest_a.out
    ssmtest_out.(ssmtest_v{1}) = eval(ssmtest_v{1});
end
end
