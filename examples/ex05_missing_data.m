%% ex05 - Missing data and mixed frequencies, on generated data
%
% In a conditionally Gaussian state space model, the missing values are jointly Gaussian
% given the observed values, the states alpha and the parameters theta, and their
% precision matrix is banded, so they are drawn in one block from a single Cholesky
% factor; see Chan, Poon and Zhu (2023). Write the stacked data as y = So*yo + Sm*ym, with
% the selection matrices ssm.select_obs returns. Stacking the measurement equation gives
% Go*yo + Gm*ym = W*alpha + X*beta + e with e ~ N(0, Sigma), where Go and Gm include any
% difference matrices the model applies to y, and then
%
%   (ym | yo, alpha, theta) ~ N(ymhat, K^{-1}),   K = Gm'*Sigma^{-1}*Gm,
%   K*ymhat = Gm'*Sigma^{-1}*(W*alpha + X*beta - Go*yo),
%
% which ssm.simulate_states draws without forming K^{-1}.
%
% Section 1 prints what ssm.select_obs returns for the first illustration in Section 2.1
% of the paper: two periods of three variables, with y_{3,1}, y_{1,2} and y_{3,2} missing.
%
% Section 2 checks that the missing values are drawn from the right distribution. It
% removes values from a generated VAR(1) in three patterns at once, a series that starts
% late, a hole in the middle and a ragged edge at the end, and compares the draws with the
% values removed and with the mean and covariance of the conditional normal, from dense
% algebra on the joint normal of y.
%
% Section 3 checks the draw under a mixed-frequency constraint. In a generated monthly
% VAR(1) with known parameters, two series are observed every month and the third only
% through the log-linear quarterly aggregation of Mariano and Murasawa (2003),
%
%   z = (y_t + 2*y_{t-1} + 3*y_{t-2} + 2*y_{t-3} + y_{t-4})/3,
%
% at the last month of each quarter after the first, which stacks into the hard
% constraint M*ym = z. One update of an unconstrained draw u imposes it exactly,
% ym = u + K^{-1}*M'*(M*K^{-1}*M')^{-1}*(z - M*u), reusing the Cholesky factor
% ssm.simulate_states returns. The draws are compared with the values removed, before and
% after the update, and with the conditional normal of ym given yo and z, from dense
% algebra on the joint normal of (y, z).
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Section 2 and Algorithm 2.
% Mariano, R.S. and Murasawa, Y. (2003). A New Coincident Index of Business Cycles
% Based on Monthly and Quarterly Series, Journal of Applied Econometrics, 18(4):
% 427-443.
% Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
% Chapman & Hall/CRC, Algorithm 2.6.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex05: missing data and mixed frequencies ===\n');

%% Section 1: what ssm.select_obs returns, on the first illustration of the paper
Ytrue = [1 2 3; 4 5 6];                          % each value is its place in the stacked y
Y = Ytrue;
Y(1,3) = NaN; Y(2,1) = NaN; Y(2,3) = NaN;        % y_{3,1}, y_{1,2} and y_{3,2}
[So, Sm, yo] = ssm.select_obs(Y);

fprintf('\n-- Section 1: ssm.select_obs, T = 2, n = 3, y_{3,1}, y_{1,2} and y_{3,2} missing\n');
fprintf('Y, one period per row:\n');
fprintf('  %5g %5g %5g\n', Y');
fprintf('\n%-9s %5s    %-5s      %s\n', 'stacked', 'y', 'So', 'Sm');
ystack = reshape(Y', [], 1);
Sfull = full([So Sm]);
lab = {'y_{1,1}', 'y_{2,1}', 'y_{3,1}', 'y_{1,2}', 'y_{2,2}', 'y_{3,2}'};
for i = 1:numel(ystack)
    fprintf('%-9s %5g    %d %d %d      %d %d %d\n', lab{i}, ystack(i), Sfull(i,:));
end
fprintf('yo = (%s)''\n', strjoin(compose('%g', yo'), ', '));

ym = Sm'*reshape(Ytrue', [], 1);                 % the missing values of the complete data
yrec = So*yo + Sm*ym;
fprintf('\nwith ym = (%s)'', So*yo + Sm*ym = (%s)'', the complete stacked data:\n', ...
    strjoin(compose('%g', ym'), ', '), strjoin(compose('%g', yrec'), ', '));
fprintf('  max abs difference %g\n', norm(yrec - reshape(Ytrue', [], 1), inf));
is_perm = all(Sfull(:) == 0 | Sfull(:) == 1) && all(sum(Sfull,1) == 1) && all(sum(Sfull,2) == 1);
fprintf('[So Sm] is a permutation matrix (entries 0 or 1, one 1 in each row and column): %s\n', ...
    mat2str(is_perm));

%% Section 2: a late start, a hole and a ragged edge, in a generated VAR(1)
rng(42);
n = 4; T = 200;
Phi = [.5 .1 0 0; .2 .4 .1 0; 0 .2 .5 .1; .1 0 .2 .3];
CSig = [1 0 0 0; .5 1 0 0; .3 .4 1 0; .2 .3 .5 1]/sqrt(2);
Sig = CSig*CSig';

Ytrue = zeros(T,n);
yt = zeros(n,1);
for t = 1:T
    yt = Phi*yt + CSig*randn(n,1);
    Ytrue(t,:) = yt';
end

Y = Ytrue;
Y(1:24,4) = NaN;                                 % a series that starts two years late
Y(91:93,2) = NaN;                                % a hole
Y(T,1) = NaN; Y(T-1:T,3) = NaN; Y(T-2:T,4) = NaN;   % the ragged edge at the end

[So, Sm, yo] = ssm.select_obs(Y);
L = sparse(2:T,1:T-1,1,T,T);                     % the lag operator on the stacked path
H = speye(T*n) - kron(L, Phi);                   % H*y = e, with y_0 = 0
iSig = kron(speye(T), inv(Sig));
Gm = H*Sm; Go = H*So;
K = Gm'*iSig*Gm;
c = -Gm'*iSig*(Go*yo);                           % no states or regressors in this model
ndraws = 20000;
[ym, ymhat] = ssm.simulate_states(K, c, ndraws);

ytrue_m = Sm'*reshape(Ytrue', T*n, 1);
sd = std(ym, 0, 2);
band = quantile(ym, [.05 .95], 2);
inside = ytrue_m >= band(:,1) & ytrue_m <= band(:,2);
[blo, bup] = bandwidth(full(K));

fprintf('\n-- Section 2: %d of %d values blanked out, VAR(1) with n = %d, T = %d\n', ...
    numel(ytrue_m), T*n, n, T);
fprintf('precision of the missing data: %d x %d, %d nonzeros, bandwidth %d\n', ...
    size(K,1), size(K,2), nnz(K), max(blo,bup));

im = find(isnan(reshape(Y', T*n, 1)));
tm = ceil(im/n); vm = im - (tm-1)*n;
grp = {'series starting late', vm == 4 & tm <= 24; ...
       'hole in the middle',   vm == 2 & tm >= 91 & tm <= 93; ...
       'ragged edge',          tm > T-3};
fprintf('\n%-22s %5s %9s %9s %9s\n', 'pattern', 'count', 'RMSE', 'post sd', 'coverage');
for g = 1:size(grp,1)
    j = grp{g,2};
    fprintf('%-22s %5d %9.3f %9.3f %8.1f%%\n', grp{g,1}, sum(j), ...
        sqrt(mean((ymhat(j) - ytrue_m(j)).^2)), mean(sd(j)), 100*mean(inside(j)));
end
fprintf('%-22s %5d %9.3f %9.3f %8.1f%%\n', 'all', numel(ytrue_m), ...
    sqrt(mean((ymhat - ytrue_m).^2)), mean(sd), 100*mean(inside));

% the known answer: condition the joint normal of y on yo with dense algebra
V = inv(full(H'*iSig*H));
io = find(~isnan(reshape(Y', T*n, 1)));
mudense = V(im,io)*(V(io,io)\yo);
Vdense = V(im,im) - V(im,io)*(V(io,io)\V(io,im));
fprintf('\nmean against dense:                  max abs difference %.2e\n', ...
    norm(ymhat - mudense, inf));
fprintf('K^{-1} against the dense covariance: max abs difference %.2e\n', ...
    max(max(abs(inv(full(K)) - Vdense))));
fprintf('draws against the dense mean:        max abs difference %.3f posterior sd\n', ...
    max(abs(mean(ym,2) - mudense)./sqrt(diag(Vdense))));
fprintf(['draws'' covariance against dense:     max abs difference %.1f%% of its ' ...
    'largest entry\n'], 100*max(max(abs(cov(ym') - Vdense)))/max(max(abs(Vdense))));

%% Section 3: a series observed only through its quarterly aggregates
rng(42);
n = 3; T = 120;                                  % ten years of months
Phi = [.5 .1 .1; .1 .4 .1; .2 .2 .4];
CSig = [1 0 0; .5 1 0; .5 .3 .6];
Sig = CSig*CSig';

Ytrue = zeros(T,n);
yt = zeros(n,1);
for t = 1:T
    yt = Phi*yt + CSig*randn(n,1);
    Ytrue(t,:) = yt';
end

% the aggregation of series 3 at the last month of each quarter, as M*ym = z; the first
% quarter is left out because its aggregate needs months 0 and -1, before the sample
qend = (6:3:T)';
nq = numel(qend);
w = [1 2 3 2 1]'/3;
M = sparse(repmat((1:nq)',5,1), reshape(qend - (0:4), [], 1), ...
    kron(w, ones(nq,1)), nq, T);
z = M*Ytrue(:,3);

Y = [Ytrue(:,1:2), nan(T,1)];
[So, Sm, yo] = ssm.select_obs(Y);                % ym = (y_{3,1}, ..., y_{3,T})'
L = sparse(2:T,1:T-1,1,T,T);
H = speye(T*n) - kron(L, Phi);
iSig = kron(speye(T), inv(Sig));
Gm = H*Sm; Go = H*So;
K = Gm'*iSig*Gm;
ndraws = 20000;
[u, uhat, C] = ssm.simulate_states(K, -Gm'*iSig*(Go*yo), ndraws);
U = C'\(C\full(M'));                             % K^{-1}*M', by the factor already formed
MU = M*U;
ym = u + U*(MU\(z - M*u));
ymhat = uhat + U*(MU\(z - M*uhat));              % the same update, applied to the mean

% the known answer: condition the joint normal of (y, z), z = M*Sm'*y, with dense algebra
V = inv(full(H'*iSig*H));
Mz = M*Sm';
Vyz = [V, V*Mz'; Mz*V, Mz*V*Mz'];
im = find(isnan(reshape(Y', T*n, 1)));
ic = [find(~isnan(reshape(Y', T*n, 1))); T*n + (1:nq)'];   % yo and z
mudense = Vyz(im,ic)*(Vyz(ic,ic)\[yo; z]);
Vdense = Vyz(im,im) - Vyz(im,ic)*(Vyz(ic,ic)\Vyz(ic,im));
Vupd = C'\(C\eye(T)) - U*(MU\U');                % K^{-1} - U*(M*U)^{-1}*U', formed to check
[blo, bup] = bandwidth(full(K));

fprintf(['\n-- Section 3: VAR(1) with n = %d, T = %d months, series 3 only in %d quarterly ' ...
    'aggregates\n'], n, T, nq);
fprintf('precision of the missing data: %d x %d, %d nonzeros, bandwidth %d; %d draws\n', ...
    size(K,1), size(K,2), nnz(K), max(blo,bup), ndraws);
fprintf('M*ym - z after the update, every draw:     max abs value %.2e\n', ...
    max(max(abs(M*ym - z))));
fprintf('mean of the update against dense:          max abs difference %.2e\n', ...
    norm(ymhat - mudense, inf));
fprintf('covariance of the update against dense:    max abs difference %.2e\n', ...
    max(max(abs(Vupd - Vdense))));
fprintf('draws against the dense mean:              max abs difference %.3f posterior sd\n', ...
    max(abs(mean(ym,2) - mudense)./sqrt(diag(Vdense))));
fprintf(['draws'' covariance against dense:           max abs difference %.1f%% of its ' ...
    'largest entry\n'], 100*max(max(abs(cov(ym') - Vdense)))/max(max(abs(Vdense))));

ytrue_m = Ytrue(:,3);
fprintf('\n%-30s %9s %9s %9s\n', 'series 3, drawn given', 'RMSE', 'post sd', 'coverage');
D = {'the monthly series', u, uhat; 'and the quarterly aggregates', ym, ymhat};
for g = 1:size(D,1)
    band = quantile(D{g,2}, [.05 .95], 2);
    fprintf('%-30s %9.3f %9.3f %8.1f%%\n', D{g,1}, sqrt(mean((D{g,3} - ytrue_m).^2)), ...
        mean(std(D{g,2}, 0, 2)), 100*mean(ytrue_m >= band(:,1) & ytrue_m <= band(:,2)));
end
