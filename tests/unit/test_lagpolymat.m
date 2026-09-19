function test_lagpolymat
% ssm.lagpolymat must reproduce exactly the lag polynomial matrices built inline: the
% second-difference and AR(2) matrices of the book's UC_output_gap.m and of
% chan2018_er_spectest's TV_NAIRU_AR2.m, and ssm.diffmat when p = 1.
T = 40;

% the book's chapter09/UC_output_gap.m, lines 41-45
S1 = sparse(2:T,1:T-1,1,T,T);                  % first-lag shift matrix
S2 = sparse(3:T,1:T-2,1,T,T);                  % second-lag shift matrix
H2 = speye(T) - 2*S1 + S2;
assert(isequal(ssm.lagpolymat(T, [2 -1]), H2), 'lagpolymat: second differences differ');
phi = [1.34 -.7]';
Hphi = speye(T) - phi(1)*S1 - phi(2)*S2;
assert(isequal(ssm.lagpolymat(T, phi), Hphi), 'lagpolymat: AR(2) of UC_output_gap.m differs');

% chan2018_er_spectest TV_NAIRU_AR2.m, lines 59-60
phi = [1.52 -.61];
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
     - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
assert(isequal(ssm.lagpolymat(T, phi), Hphi), 'lagpolymat: AR(2) of TV_NAIRU_AR2.m differs');

% p = 1: ssm.diffmat, for a random walk, an AR(1) and an MA(1) transform
for a = [1 0.937 -0.42]
    assert(isequal(ssm.lagpolymat(T, a), ssm.diffmat(T, a)), ...
        'lagpolymat: p = 1 differs from ssm.diffmat at a = %g', a);
end

% the defining property: H*x applies the lag polynomial, pre-sample values zero
phi = [0.5 -0.3 0.2];
x = randn(T,1);
assert(norm(ssm.lagpolymat(T, phi)*x - filter([1 -phi], 1, x)) < 1e-12, ...
    'lagpolymat: H*x must equal filter([1 -phi], 1, x)');

% structure, and lags beyond the path
H = ssm.lagpolymat(T, phi);
assert(issparse(H) && isequal(size(H), [T T]), 'lagpolymat: must be sparse and T x T');
assert(nnz(H) == T + (T-1) + (T-2) + (T-3), 'lagpolymat: wrong sparsity');
assert(isequal(ssm.lagpolymat(T, []), speye(T)), 'lagpolymat: empty phi must give the identity');
assert(isequal(ssm.lagpolymat(3, [.5 .2 .1 .05]), ...
    speye(3) - .5*sparse(2:3,1:2,1,3,3) - .2*sparse(3,1,1,3,3)), ...
    'lagpolymat: lags beyond T must drop out');
assert(isequal(ssm.lagpolymat(1, phi), speye(1)), 'lagpolymat: T = 1');

% bad input rejected, with the ssm identifiers
bad = {{0, 1}, 'ssm:lagpolymat:badT'; {2.5, 1}, 'ssm:lagpolymat:badT'; ...
    {5, ones(2)}, 'ssm:lagpolymat:badPhi'; {5, 'a'}, 'ssm:lagpolymat:badPhi'};
for ii = 1:size(bad, 1)
    try
        ssm.lagpolymat(bad{ii,1}{:});
        error('test:noThrow', 'lagpolymat should reject case %d', ii);
    catch err
        assert(strcmp(err.identifier, bad{ii,2}), 'lagpolymat: case %d gave %s', ii, err.identifier);
    end
end
end
