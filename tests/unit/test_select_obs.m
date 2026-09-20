function test_select_obs
% ssm.select_obs must reproduce the two illustrations of Chan, Poon and Zhu (2023),
% Section 2.1, and the conditional distribution of the missing data built from its
% output must equal the dense Gaussian conditioning formula.

% the paper's first illustration: T = 2, n = 3, with y_{3,1}, y_{1,2} and y_{3,2} missing
Y = [1 2 NaN; NaN 5 NaN];
[So, Sm, yo] = ssm.select_obs(Y);
assert(isequal(full(So), [1 0 0; 0 1 0; 0 0 0; 0 0 0; 0 0 1; 0 0 0]), ...
    'select_obs: So differs from the illustration in the paper');
assert(isequal(full(Sm), [0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 0; 0 0 1]), ...
    'select_obs: Sm differs from the illustration in the paper');
assert(isequal(yo, [1; 2; 5]), 'select_obs: yo must be the observed values in stacked order');

% the paper's second illustration: y_{1,t} observed every third period, the rest always.
% So is block diagonal with blocks I_n at t = 3, 6, 9 and [0'; I_{n-1}] elsewhere; the
% Sm blocks are [1; 0] at the same periods and empty at t = 3, 6, 9
T = 9; n = 4;
Y = randn(T, n);
Y(mod(1:T,3) ~= 0, 1) = NaN;
[So, Sm] = ssm.select_obs(Y);
bo = cell(T,1); bm = cell(T,1);
for t = 1:T
    if mod(t,3) == 0
        bo{t} = eye(n);            bm{t} = zeros(n,0);
    else
        bo{t} = [zeros(1,n-1); eye(n-1)];   bm{t} = [1; zeros(n-1,1)];
    end
end
assert(isequal(full(So), blkdiag(bo{:})), 'select_obs: So is not the block diagonal of the paper');
assert(isequal(full(Sm), blkdiag(bm{:})), 'select_obs: Sm is not the block diagonal of the paper');

% [So, Sm] is a permutation matrix, and it splits and rebuilds a complete path
P = [So Sm];
assert(isequal(size(P), [T*n T*n]), 'select_obs: [So Sm] must be square');
assert(all(sum(P,1) == 1) && all(sum(P,2) == 1) && nnz(P) == T*n, ...
    'select_obs: [So Sm] must be a permutation matrix');
assert(isequal(So'*So, speye(size(So,2))) && isequal(Sm'*Sm, speye(size(Sm,2))) ...
    && nnz(So'*Sm) == 0, 'select_obs: the two selections must be orthonormal');
y = randn(T*n, 1);
assert(isequal(So*(So'*y) + Sm*(Sm'*y), y), 'select_obs: y = So*yo + Sm*ym must hold');

% the conditional distribution of the missing data. For a VAR(1) with H*y = eps and
% eps ~ N(0, I_T kron Sigma), the paper gives (ym | yo) ~ N(mu, inv(K)) with
% K = Gm'*inv(Sigma)*Gm and Gm = H*Sm; both must match conditioning the joint normal.
T = 8; n = 2;
Phi = [.6 .1; -.2 .5];  Sig = [1 .3; .3 .8];
H = speye(T*n) - kron(sparse(2:T,1:T-1,1,T,T), Phi);
iSig = kron(speye(T), inv(Sig));
V = inv(full(H'*iSig*H));                           % the joint covariance of y
yfull = chol(V,'lower')*randn(T*n,1);
Y = reshape(yfull, n, T)';
Y(4,1) = NaN; Y(7,2) = NaN; Y(8,2) = NaN;           % a hole and a ragged edge
[So, Sm, yo] = ssm.select_obs(Y);
Gm = H*Sm;  Km = Gm'*iSig*Gm;
mum = Km\(Gm'*iSig*(-(H*So)*yo));
im = find(isnan(reshape(Y', T*n, 1)));  io = find(~isnan(reshape(Y', T*n, 1)));
mudense = V(im,io)*(V(io,io)\yo);
Kdense = inv(V(im,im) - V(im,io)*(V(io,io)\V(io,im)));
assert(norm(mum - mudense, inf) < 1e-9, 'select_obs: the conditional mean differs');
assert(norm(full(Km) - Kdense, inf) < 1e-8, 'select_obs: the conditional precision differs');

% the mixed-frequency pattern keeps that precision banded: two missing values are linked
% only when their periods are adjacent, because each column of a VAR(1) H is nonzero in
% the rows of its own period and the next
T = 12;
Y = randn(T, n);
Y(mod(1:T,3) ~= 0, 1) = NaN;
[~, Sm] = ssm.select_obs(Y);
Gm = (speye(T*n) - kron(sparse(2:T,1:T-1,1,T,T), Phi))*Sm;
Km = Gm'*kron(speye(T), inv(Sig))*Gm;
[lo, up] = bandwidth(full(Km));
assert(lo == 1 && up == 1, 'select_obs: the mixed-frequency precision should be tridiagonal');

% patterns with nothing missing, or nothing observed
[So, Sm, yo] = ssm.select_obs(ones(5,2));
assert(isequal(size(So), [10 10]) && isequal(size(Sm), [10 0]) && numel(yo) == 10, ...
    'select_obs: a complete panel must give an empty Sm');
[So, Sm, yo] = ssm.select_obs(nan(5,2));
assert(isequal(size(So), [10 0]) && isequal(size(Sm), [10 10]) && isempty(yo), ...
    'select_obs: an empty panel must give an empty So');

% bad input rejected, with the ssm identifier
bad = {[], 'a', ones(2,2,2)};
for ii = 1:numel(bad)
    try
        ssm.select_obs(bad{ii});
        error('test:noThrow', 'select_obs should reject case %d', ii);
    catch err
        assert(strcmp(err.identifier, 'ssm:select_obs:badY'), ...
            'select_obs: case %d gave %s', ii, err.identifier);
    end
end
end
