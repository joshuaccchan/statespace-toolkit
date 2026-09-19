function test_intlike
% ssm.intlike against the Kalman filter's prediction-error log likelihood on a
% simulated local level model, a simulated TVP regression, and the UC model with
% an AR(2) cycle on the GDP data of grant_chan2017_jmcb_trendcycle; and against
% the archived intlike_UC0.m and intlike_tvpvar.m on their packages' own data.
% Measured differences on 18 September 2026: 8e-13, 9e-13 and 4e-12 from the
% Kalman filters, 4e-12 from intlike_tvpvar, and 2.6e-8 from intlike_UC0, whose
% b'*inv(K)*b form cancels terms near 1e7 on data near 750 (the UC Kalman filter
% agrees with ssm.intlike to 4e-12).
root = getappdata(0, 'ssm_repo_root');
tol = 1e-9;

% --- local level: y_t = tau_t + e_t, tau_t = tau_{t-1} + u_t, tau_1 ~ N(tau0, om2)
rng(21, 'twister');
T = 200; sig2 = .8; om2 = .15; tau0 = 1.2;
y = tau0 + cumsum(sqrt(om2)*randn(T,1)) + sqrt(sig2)*randn(T,1);
H = ssm.diffmat(T);
[ll, alphahat, K] = ssm.intlike(y, speye(T), speye(T)/sig2, H'*H/om2, tau0*ones(T,1));
a = tau0; P = om2; lk = 0;
for t = 1:T
    F = P + sig2; v = y(t) - a;
    lk = lk - .5*(log(2*pi) + log(F) + v^2/F);
    a = a + P*v/F; P = P - P^2/F + om2;
end
assert(abs(ll - lk) < tol, 'intlike: local level differs from the Kalman filter by %.3g', ll - lk);
assert(isequal(K, H'*H/om2 + speye(T)'*(speye(T)/sig2)*speye(T)), 'intlike: K is not P + Z''*iR*Z');
assert(max(abs(alphahat - K\(H'*H/om2*(tau0*ones(T,1)) + y/sig2))) < 1e-10, ...
    'intlike: alphahat differs from K\\(P*b + Z''*iR*y)');

% --- TVP regression: y_t = x_t'*b_t + e_t, b_t = b_{t-1} + u_t, b_1 ~ N(b0, Om)
rng(22, 'twister');
T = 150; k = 3; sig2 = .5; Om = diag([.05 .02 .01]); b0 = [1; .5; -.3];
Xr = [ones(T,1) randn(T,2)];
B = b0' + cumsum(randn(T,k)*sqrt(Om));
y = sum(Xr.*B, 2) + sqrt(sig2)*randn(T,1);
Hk = speye(T*k) - sparse(k+1:T*k, 1:(T-1)*k, 1, T*k, T*k);
ll = ssm.intlike(y, ssm.surform(Xr), speye(T)/sig2, ...
    Hk'*kron(speye(T), sparse(inv(Om)))*Hk, repmat(b0, T, 1));
a = b0; P = Om; lk = 0;
for t = 1:T
    x = Xr(t,:)'; F = x'*P*x + sig2; v = y(t) - x'*a;
    lk = lk - .5*(log(2*pi) + log(F) + v^2/F);
    g = P*x/F; a = a + g*v; P = P - g*x'*P + Om;
end
assert(abs(ll - lk) < tol, 'intlike: TVP regression differs from the Kalman filter by %.3g', ll - lk);

% --- UC model with AR(2) cycle, 100*log US GDP: intlike_UC0 and a Kalman filter
leg = fullfile(root, 'replications', 'grant_chan2017_jmcb_trendcycle', 'legacy');
addpath(leg); c = onCleanup(@() rmpath(leg));
y = 100*log(load(fullfile(leg, 'USGDP.csv'))); T = length(y);
mu = .8; phi = [1.3; -.4]; sigc2 = .5; sigtau2 = .4; tau0 = y(1) - .5;
lleg = intlike_UC0(y, mu, phi, sigc2, sigtau2, tau0);
clear c
H = ssm.diffmat(T);
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
ll = ssm.intlike(Hphi*y, Hphi, speye(T)/sigc2, H'*H/sigtau2, H\(mu + [tau0; sparse(T-1,1)]));
Fm = [1 0 0; 0 phi(1) phi(2); 0 1 0]; Q = diag([sigtau2 sigc2 0]); Z = [1 1 0];
a = [tau0; 0; 0]; P = zeros(3); lk = 0;              % s_0 = [tau_0; c_0; c_{-1}], known
for t = 1:T
    a = Fm*a + [mu; 0; 0]; P = Fm*P*Fm' + Q;
    F = Z*P*Z'; v = y(t) - Z*a;
    lk = lk - .5*(log(2*pi) + log(F) + v^2/F);
    g = P*Z'/F; a = a + g*v; P = P - g*Z*P;
end
assert(abs(ll - lk) < tol, 'intlike: UC model differs from the Kalman filter by %.3g', ll - lk);
assert(abs(ll - lleg) < 1e-6, 'intlike: differs from intlike_UC0 by %.3g', ll - lleg);

% --- TVP-VAR, chan_grant2016_csda_dic's data and starting values: intlike_tvpvar
leg = fullfile(root, 'replications', 'chan_grant2016_csda_dic', 'legacy', 'DIC');
addpath(leg); c = onCleanup(@() rmpath(leg));
USdata = load(fullfile(leg, 'USdata.csv'));
Y0 = USdata(1:3,:); shortY = USdata(4:end,:); [T, n] = size(shortY);
Y = reshape(shortY', T*n, 1);
k = n^2 + n; Tk = T*k;                                % p = 1 lag
Xl = [Y0(end,:); shortY(1:T-1,:)];
bigX = ssm.surform([ones(n*T,1) kron(Xl, ones(n,1))]);
b0 = sparse(k,1); Q0 = 5*ones(k,1); Sig = cov(USdata); Omega = .01*ones(k,1);
lleg = intlike_tvpvar(Y, bigX, Sig, Omega, Q0, b0);
clear c
H = speye(Tk) - sparse(k+1:Tk, 1:(T-1)*k, ones(1,(T-1)*k), Tk, Tk);
K0 = H'*spdiags([1./Q0; repmat(1./Omega, T-1, 1)], 0, Tk, Tk)*H;
ll = ssm.intlike(Y, bigX, kron(speye(T), inv(Sig)), K0, H\[b0; sparse((T-1)*k,1)]);
assert(abs(ll - lleg) < tol, 'intlike: differs from intlike_tvpvar by %.3g', ll - lleg);

% --- a scalar iR would drop Tn-1 log-determinant terms: rejected
try
    ssm.intlike(ones(3,1), speye(3), 2, speye(3), zeros(3,1));
    error('test:noThrow', 'intlike should reject a scalar iR');
catch err
    assert(strcmp(err.identifier, 'ssm:intlike:badR'), 'intlike: wrong error %s', err.identifier);
end
end
