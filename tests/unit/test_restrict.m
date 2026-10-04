function test_restrict
% ssm.restrict must equal, bit for bit, the update that ex05 and ex10 wrote out before
% it existed, and its output must have the moments of the dense Gaussian conditional
% of x given M*x = z.

rng(1);
% a banded precision: the states of a VAR(1) of two series over 40 periods
T = 40; n = 2;
Phi = [.6 .1; -.2 .5]; Sig = [1 .3; .3 .8];
H = speye(T*n) - kron(sparse(2:T,1:T-1,1,T,T), Phi);
K = H'*kron(speye(T), inv(Sig))*H;
c = randn(T*n, 1);
[x, xhat, C] = ssm.simulate_states(K, c, 50);

% a sparse restriction, the quarterly aggregation of the second series as in ex05, and
% a full one, a sum of the two series in every period as in ex10
qend = (6:3:T)'; nq = numel(qend);
Ma = sparse(repmat((1:nq)',5,1), n*(reshape(qend - (0:4), [], 1)), ...
    kron([1 2 3 2 1]'/3, ones(nq,1)), nq, T*n);
Mb = full(kron(speye(T), [1 1]));
for k = 1:2
    if k == 1, M = Ma; else, M = Mb; end
    r = size(M, 1); z = randn(r, 1);

    % the inline spellings of ex05 (draws and mean) and ex10 (one draw)
    U = C'\(C\full(M'));  MU = M*U;
    xa = x + U*(MU\(z - M*x));
    ma = xhat + U*(MU\(z - M*xhat));
    Mt = full(M');  U2 = C'\(C\Mt);
    xb = x(:,1) + U2*((M*U2)\(z - M*x(:,1)));
    [xr, Ur, MUr] = ssm.restrict(x, C, M, z);
    assert(isequal(xr, xa) && isequal(Ur, U) && isequal(MUr, MU), ...
        'restrict: differs from the update of ex05');
    assert(isequal(ssm.restrict(xhat, C, M, z), ma), ...
        'restrict: differs from the update of ex05 applied to the mean');
    assert(isequal(ssm.restrict(x(:,1), C, M, z), xb), 'restrict: differs from the update of ex10');

    % the dense conditional of x given M*x = z
    V = inv(full(K));  mu = V*c;
    A = V*M';  G = M*A;
    mu_c = mu + A*(G\(z - M*mu));
    V_c = V - A*(G\A');
    assert(norm(ssm.restrict(xhat, C, M, z) - mu_c, inf) < 1e-9*norm(mu_c, inf), ...
        'restrict: the conditional mean differs from dense algebra');
    assert(max(max(abs(V - Ur*(MUr\Ur') - V_c))) < 1e-9*max(max(abs(V))), ...
        'restrict: the conditional covariance differs from dense algebra');
    assert(max(max(abs(M*xr - z))) < 1e-10*max(1, norm(z, inf)), ...
        'restrict: the draws do not satisfy the restriction');

    % a block of draws gives what each draw gives on its own
    for j = [1 17 50]
        assert(norm(xr(:,j) - ssm.restrict(x(:,j), C, M, z), inf) < 1e-12*norm(xr(:,j), inf), ...
            'restrict: a block of draws differs from the draws one at a time');
    end
end
end
