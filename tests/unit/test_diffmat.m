function test_diffmat
% ssm.diffmat must reproduce exactly the state equation matrices that the archived
% packages and ssm.ksc_rw_h0 build inline, in each of their spellings.
T = 40;

% random walk: ssm.ksc_rw_h0 (spdiags) and chan2013_joe_masv UC_MA.m line 30 (sparse)
H_rw = speye(T) - spdiags(ones(T-1,1), -1, T, T);
assert(isequal(ssm.diffmat(T), H_rw), 'diffmat: RW form differs from the spdiags spelling');
assert(isequal(ssm.diffmat(T, 1), H_rw), 'diffmat: a = 1 must equal the RW form');
H_uc = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
assert(isequal(ssm.diffmat(T), H_uc), 'diffmat: RW form differs from the sparse spelling');

% AR(1), coefficient inside the values: chan2013_joe_masv SV.m line 30
phi = 0.937;
H_ar = speye(T)-sparse(2:T,1:(T-1),phi*ones(1,T-1),T,T);
assert(isequal(ssm.diffmat(T, phi), H_ar), 'diffmat: AR(1) form differs');

% MA(1) transform I + psi*L: chan_grant2016_eneco_garchsv loglike_garch_ma.m line 16
psi = -0.42;
Hpsi = speye(T) + psi*sparse(2:T,1:T-1,ones(T-1,1),T,T);
assert(isequal(ssm.diffmat(T, -psi), Hpsi), 'diffmat: MA(1) form differs');

% structure: unit diagonal, one subdiagonal, sparse, right size
H = ssm.diffmat(T, phi);
assert(issparse(H), 'diffmat: must be sparse');
assert(isequal(size(H), [T T]), 'diffmat: wrong size');
assert(nnz(H) == T + (T-1), 'diffmat: wrong sparsity');
assert(all(full(diag(H)) == 1), 'diffmat: diagonal must be 1');
assert(isequal(full(diag(H,-1)), -phi*ones(T-1,1)), 'diffmat: wrong subdiagonal');

% the defining property: H*x differences a path
x = cumsum(randn(T,1));
assert(norm(H_rw*x - [x(1); diff(x)]) < 1e-12, 'diffmat: H*x must difference x');

% edge case
assert(isequal(ssm.diffmat(1, phi), speye(1)), 'diffmat: T = 1');

% bad input rejected, with the ssm identifiers
bad = {{0}, 'ssm:diffmat:badT'; {2.5}, 'ssm:diffmat:badT'; {5, [1 2]}, 'ssm:diffmat:badA'};
for ii = 1:size(bad, 1)
    try
        ssm.diffmat(bad{ii,1}{:});
        error('test:noThrow', 'diffmat should reject case %d', ii);
    catch err
        assert(strcmp(err.identifier, bad{ii,2}), 'diffmat: case %d gave %s', ii, err.identifier);
    end
end
end
