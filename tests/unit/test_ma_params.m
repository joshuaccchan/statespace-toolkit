function test_ma_params
% ssm.ma_params must draw from the conditional density of psi: chains of the step alone,
% with x and w held fixed, must match the density computed on a grid of 20,001 points, under
% the default and an informative prior and with psi near the boundary. It must use one randn
% and one rand per call, and return u under the returned psi. Under the prior N(0, 1),
% test_uc checks it draw for draw against the psi step of the UC tutorial's sampler, through
% ssm.models.uc.

cases = {.6, struct('psi0', 0, 'Vpsi', 1); .3, struct('psi0', -.4, 'Vpsi', .05); ...
    .95, struct('psi0', 0, 'Vpsi', 1)};
T = 150; N = 10000;
g = linspace(-.9999, .9999, 20001)';
for c = 1:size(cases, 1)
    rng(10 + c, 'twister');
    w = exp(-.5*randn(T,1));
    x = filter([1 cases{c,1}], 1, randn(T,1)./sqrt(w));
    Hy = cases{c,2};
    lf = zeros(size(g));
    for k = 1:numel(g)
        u = filter(1, [1 g(k)], x);
        lf(k) = -.5*(w'*u.^2) - .5*(g(k) - Hy.psi0)^2/Hy.Vpsi;
    end
    pg = exp(lf - max(lf)); pg = pg/trapz(g, pg);
    mg = trapz(g, g.*pg); sg = sqrt(trapz(g, (g - mg).^2.*pg)); Fg = cumtrapz(g, pg);
    [~, im] = max(lf);
    d = zeros(N,1); psi = g(im) + .05;                 % near the mode: see the trap in ma_params
    for i = 1:N, psi = ssm.ma_params(x, w, psi, Hy); d(i) = psi; end
    q = quantile(d, [.1 .5 .9]);
    assert(abs(mean(d) - mg) < 4*ssm.mcse(d, 50) + 1e-4, 'ma_params: the mean differs from the grid, case %d', c);
    assert(abs(std(d)/sg - 1) < .05, 'ma_params: the sd differs from the grid, case %d', c);
    assert(max(abs(interp1(g, Fg, q) - [.1 .5 .9])) < .025, 'ma_params: the quantiles differ from the grid, case %d', c);
end

% one randn, then one rand, per call; u under the returned psi
rng(3, 'twister'); [psi, u] = ssm.ma_params(x, w, .5, Hy); s1 = rng;
rng(3, 'twister'); randn; rand; s2 = rng;
assert(isequal(s1.State, s2.State), 'ma_params: the random numbers per call differ from the header');
assert(isequal(u, filter(1, [1 psi], x)), 'ma_params: u is not the errors under the returned psi');
end
