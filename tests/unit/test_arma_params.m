function test_arma_params
% ssm.arma_params must draw from the joint conditional density of rho and psi: chains of the
% step alone, with e and w held fixed, must match the means, standard deviations and
% correlation computed on a 300 x 300 grid, for coefficients of the same sign and of
% opposite signs. It must use one rand before the random numbers of ssm.ma_params, and
% return u under the returned rho and psi.

cases = [.6 .3; .9 -.5];
T = 200; N = 10000;
gr = linspace(-.999, .999, 300);
Hy = struct('rho0', 0, 'Vrho', 1, 'psi0', 0, 'Vpsi', 1);
for c = 1:size(cases, 1)
    rng(20 + c, 'twister');
    w = exp(-.5*randn(T,1));
    e = filter([1 cases(c,2)], [1 -cases(c,1)], randn(T,1)./sqrt(w));
    LF = zeros(300);
    for i = 1:300
        for j = 1:300
            u = filter([1 -gr(i)], [1 gr(j)], e);
            LF(i,j) = -.5*(w'*u.^2) - .5*gr(i)^2 - .5*gr(j)^2;
        end
    end
    P = exp(LF - max(LF(:))); P = P/sum(P(:));
    pr = sum(P, 2)'; pp = sum(P, 1);
    m = [pr*gr' pp*gr'];
    s = sqrt([pr*(gr' - m(1)).^2 pp*(gr' - m(2)).^2]);
    r = sum(sum(P.*((gr' - m(1))*(gr - m(2)))))/prod(s);
    d = zeros(N, 2); rho = 0; psi = 0;
    for i = 1:N, [rho, psi] = ssm.arma_params(e, w, rho, psi, Hy); d(i,:) = [rho psi]; end
    assert(all(abs(mean(d) - m) < 4*ssm.mcse(d, 100) + 2e-3), 'arma_params: the means differ from the grid, case %d', c);
    assert(all(abs(std(d)./s - 1) < .06), 'arma_params: the sds differ from the grid, case %d', c);
    assert(abs(corr(d(:,1), d(:,2)) - r) < .04, 'arma_params: the correlation differs from the grid, case %d', c);
end

% one rand, then the randn and rand of ssm.ma_params; u under the returned rho and psi
rng(4, 'twister'); [rho, psi, u] = ssm.arma_params(e, w, .5, -.2, Hy); s1 = rng;
rng(4, 'twister'); rand; randn; rand; s2 = rng;
assert(isequal(s1.State, s2.State), 'arma_params: the random numbers per call differ from the header');
assert(max(abs(u - filter([1 -rho], [1 psi], e))) < 1e-12*max(abs(u)), ...
    'arma_params: u is not the errors under the returned rho and psi');
end
