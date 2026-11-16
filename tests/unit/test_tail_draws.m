function test_tail_draws
% Draws far in the tails. (1) ssm.tnormrnd on (9, 10) and (-10, -9) for a standard normal, where
% both normal cdf values round to 1 in the upper tail: the draws are finite, lie in the bounds
% and have the mean of the truncated normal. (2) The KSC samplers when every mixture density of
% some periods underflows: the paths are finite, and those periods take the widest component,
% whose density is largest there by thousands of log points.
rng(5, 'twister');
N = 2000;
m = (normpdf(9) - normpdf(10))/(normcdf(-9) - normcdf(-10));   % the mean of N(0,1) on (9, 10)
t = ssm.tnormrnd(0, 1, 9, 10, N);
assert(all(isfinite(t) & t >= 9 & t <= 10), 'tnormrnd: upper-tail draws outside (9, 10)');
assert(abs(mean(t) - m) < 4*std(t)/sqrt(N), 'tnormrnd: upper-tail mean off');
t = ssm.tnormrnd(0, 1, -10, -9, N);
assert(all(isfinite(t) & t >= -10 & t <= -9), 'tnormrnd: lower-tail draws outside (-10, -9)');
assert(abs(mean(t) + m) < 4*std(t)/sqrt(N), 'tnormrnd: lower-tail mean off');

T = 60;
ystar = log(randn(T,1).^2 + 1e-4);
h = zeros(T,1); h(10:12) = 2000;         % every density underflows in periods 10 to 12
[h1, S1] = ssm.ksc_ar1_mean(ystar, h, 0, .9, .1);
[h2, S2] = ssm.ksc_rw_diffuse(ystar, h, .1, 10);
h3 = ssm.ksc_rw_h0(ystar, h, .1, 0);
htilde = h/.4;                           % h0 + omegah*htilde = h
[h4, h0, omegah] = ssm.ksc_rw_noncentered(ystar, htilde, 0, .4, 0, 10, 10, .2);
assert(all(isfinite([h1; h2; h3; h4; h0; omegah])), 'KSC samplers: non-finite draw after underflow');
assert(all(ismember([S1; S2], 1:7)), 'KSC samplers: indicators outside 1..7');
assert(all(S1(10:12) == 1) && all(S2(10:12) == 1), 'KSC samplers: underflow periods not in component 1');
end
