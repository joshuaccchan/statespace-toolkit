function test_mode_newton
% On a Gaussian log density Newton-Raphson reaches the mean in one step and stops
% after the second; on the stochastic volatility target of sample_CSV.m it stops
% at a point where the gradient is zero; the step cap and a NaN step raise.
T = 60;

% --- Gaussian: log f = -.5*(x - m)'*Q*(x - m), with banded Q
rng(3, 'twister');
H = ssm.diffmat(T, .9);
Q = H'*H/.2 + speye(T);
m = randn(T,1);
gK = @(x) deal(-Q*(x - m), Q);
[xhat, K, iter] = ssm.mode_newton(zeros(T,1), gK);
assert(max(abs(xhat - m)) < 1e-10, 'mode_newton: misses the mean of a Gaussian');
assert(isequal(K, Q) && iter == 2, 'mode_newton: wrong K or step count on a Gaussian');

% --- common stochastic volatility, as in sample_CSV.m of Chan (2023)
n = 15; rho = .95; sigh2 = .05;
h = zeros(T,1); h(1) = sqrt(sigh2/(1-rho^2))*randn;
for t = 2:T, h(t) = rho*h(t-1) + sqrt(sigh2)*randn; end
s2 = exp(h).*chi2rnd(n,T,1);
Hrho = ssm.diffmat(T, rho);
HiSH = Hrho'*sparse(1:T,1:T,[(1-rho^2)/sigh2; 1/sigh2*ones(T-1,1)])*Hrho;
gK = @(x) deal(-n/2 + .5*(s2./exp(x)) - HiSH*x, HiSH + sparse(1:T,1:T,.5*(s2./exp(x))));
[xhat, ~, iter] = ssm.mode_newton(log(s2/n), gK, 'Tol', 1e-10);
[g, ~] = gK(xhat);
assert(max(abs(g)) < 1e-8 && iter > 2, 'mode_newton: gradient not zero at the SV mode');

% --- the cap, and a NaN step, raise instead of returning
cases = {@() ssm.mode_newton(log(s2/n), gK, 'MaxIter', 1), ...
         @() ssm.mode_newton(zeros(T,1), @(x) deal(NaN(T,1), speye(T)))};
for ii = 1:numel(cases)
    try
        cases{ii}();
        error('test:noThrow', 'mode_newton: case %d did not raise', ii);
    catch err
        assert(strcmp(err.identifier, 'ssm:mode_newton:notConverged'), ...
            'mode_newton: case %d gave %s', ii, err.identifier);
    end
end
end
