% bvar.util.shaded_band - a pointwise credible band drawn as a shaded region.
%
%   h = bvar.util.shaded_band(x, lo, hi)
%   h = bvar.util.shaded_band(x, lo, hi, shade)
%
%   x      : points on the horizontal axis, numeric or datetime
%   lo, hi : lower and upper bounds at those points, the same length as x
%   shade  : grey level in [0, 1] (default .85)
%   h      : handle to the fill object
%
% Draws on the current axes; call hold on first to plot lines over the band.
% The function is shaded_band.m from the code of Bayesian Macroeconometrics:
% Methods and Applications (chapter 14).

function h = shaded_band(x, lo, hi, shade)
if nargin < 4
    shade = .85;
end
x = x(:); lo = lo(:); hi = hi(:);
h = fill([x; flipud(x)], [lo; flipud(hi)], shade*[1 1 1], 'EdgeColor', 'none');
end
