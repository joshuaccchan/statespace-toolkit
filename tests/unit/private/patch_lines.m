function txt = patch_lines(txt, patches, script)
% Replace whole lines of a script, each found by the statement it starts with; the
% file keeps its own line endings. A row {start, new} replaces the one line that
% starts with start. A row {start, new, stop} replaces the lines from that line up
% to, not including, the one line that starts with stop. new is a line or a cell
% array of lines. Every start and stop must match exactly one line.
if contains(txt, sprintf('\r\n')), eol = sprintf('\r\n'); else, eol = newline; end
lines = strsplit(txt, eol, 'CollapseDelimiters', false);
for p = 1:size(patches, 1)
    trimmed = strtrim(lines);
    first = find(startsWith(trimmed, patches{p,1}));
    assert(isscalar(first), 'patch "%s" matches %d lines of %s', ...
        patches{p,1}, numel(first), script);
    last = first;
    if size(patches, 2) > 2 && ~isempty(patches{p,3})
        stop = find(startsWith(trimmed, patches{p,3}));
        assert(isscalar(stop) && stop > first, ...
            'patch end "%s" matches %d lines of %s, or none after its start', ...
            patches{p,3}, numel(stop), script);
        last = stop - 1;
    end
    new = patches{p,2};
    if ~iscell(new), new = {new}; end
    lines = [lines(1:first-1), new(:)', lines(last+1:end)];
end
txt = strjoin(lines, eol);
end
