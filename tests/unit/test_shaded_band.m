function test_shaded_band
% The band is one fill object tracing lo from left to right and hi from right to
% left, in the requested grey and with no edge; row and column inputs agree.
f = figure('Visible', 'off');
cleanup = onCleanup(@() close(f));

x = (1:5)'; lo = [1 2 3 2 1]'; hi = lo + [1 2 1 2 1]';
h = ssm.shaded_band(x, lo, hi);
assert(isequal(h.XData(:), [x; flipud(x)]), 'shaded_band: x vertices');
assert(isequal(h.YData(:), [lo; flipud(hi)]), 'shaded_band: y vertices');
assert(isequal(h.FaceColor, .85*[1 1 1]), 'shaded_band: default grey');
assert(strcmp(h.EdgeColor, 'none'), 'shaded_band: edge');

h2 = ssm.shaded_band(x', lo', hi', .6);
assert(isequal(h2.XData(:), [x; flipud(x)]) && isequal(h2.YData(:), [lo; flipud(hi)]), ...
    'shaded_band: row inputs');
assert(isequal(h2.FaceColor, .6*[1 1 1]), 'shaded_band: custom grey');
end
