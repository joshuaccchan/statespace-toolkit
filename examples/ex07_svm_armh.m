%% ex07 - Stochastic volatility in mean with time-varying parameters, by accept-reject MH
%
% y_t = tau_t + alpha_t*exp(h_t) + e_t, e_t ~ N(0, exp(h_t)), where y_t is quarterly US
% CPI inflation, 1948Q1-2025Q3, 400 times the log change in the quarterly average CPI.
% The log-volatility follows h_t = mu + phi*(h_{t-1} - mu) + v_t, v_t ~ N(0, sig2), with
% h_1 ~ N(mu, sig2/(1-phi^2)), and gam_t = (alpha_t, tau_t)' is a random walk,
% gam_t = gam_{t-1} + w_t, w_t ~ N(0, Omega), with Omega a full 2 x 2 matrix. ex07 is a
% simpler version of the model of Chan (2017), whose state equation for h also has the
% term beta*y_{t-1}. The sampler is that of the paper's UC_SVM.m, in
% replications/chan2017_jbes_svm, without the draw of beta, with the path of gam drawn by
% ssm.simulate_states and h by ssm.armh: the accept-reject Metropolis-Hastings step of
% Chan (2017), whose Gaussian proposal is centered at the mode of the conditional
% density of h.
%
% Section 1 estimates the model on US CPI inflation, from 20,000 draws after 5,000 of
% burn-in, and reports the acceptance rates of h and of (mu, phi). The chain starts where
% UC_SVM.m starts, and the first candidate for h is accepted outright, as in UC_SVM.m:
% from a start far from the target, the chain can reject every candidate.
%
% Section 2 shows how the acceptance rate depends on the envelope constant c_reject:
% 1,000 draws of h at each value, from the target density of the last draw of h in the
% chain, with the MH acceptance rate and the number of candidates per draw.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Sections 6.2.3 and 10.3.3.
% Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with Time-Varying
% Parameters: An Application to Inflation Modeling, Journal of Business and
% Economic Statistics, 35(1): 17-28.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex07: stochastic volatility in mean, h drawn by accept-reject MH ===\n');

%% 1. The posterior and the acceptance rates
y = readmatrix(fullfile(fileparts(mfilename('fullpath')), 'data', 'USCPI_quarterly.csv'), ...
    'Range', 'C2:C312');
T = length(y);
tid = 1948 + (0:T-1)'/4;                                   % quarterly, 1948Q1-2025Q3

rng(1);
nloop = 25000; burnin = 5000;

% prior, as in UC_SVM.m
phi0 = .97; Vphi = .1^2;
mu0 = 0; Vmu = 10;
Vgam = 10*eye(2); invVgam = Vgam\speye(2);
nuOmega = 10; SOmega = (nuOmega+3)*diag([.1^2 .25^2]);
nuh = 10; Sh = .2^2*(nuh-1);

% initialize the Markov chain, as in UC_SVM.m
mu = log(var(y)); phi = .98; sig2 = .2^2;
Omega = diag([.1^2 .25^2]);
h = mu + sqrt(sig2)*randn(T,1);
exph = exp(h);

store_theta = zeros(nloop - burnin,6);                     % [mu phi sig2 Omega([1 2 4])]
store_alp = zeros(nloop - burnin,T);
store_h = zeros(nloop - burnin,T);

H = kron(ssm.diffmat(T), speye(2));                        % first differences of gam_t
Hphi = ssm.diffmat(T, phi);
newnuh = T/2 + nuh;
counth = 0; countlam = 0;
for loop = 1:nloop
    % sample gam = (alpha_1, tau_1, ..., alpha_T, tau_T)'
    invOmegagam = [invVgam sparse(2,2*(T-1)); ...
        sparse(2*(T-1),2) kron(speye(T-1),Omega\speye(2))];
    Xgam = ssm.surform([exph ones(T,1)]);
    temp1 = Xgam' * sparse(1:T,1:T,1./exph);
    Kgam = H'*invOmegagam*H + temp1*Xgam;
    gam = ssm.simulate_states(Kgam, temp1*y);
    alp = gam(1:2:end);
    tau = gam(2:2:end);

    % sample h by accept-reject MH
    HinvSH = Hphi'*sparse(1:T,1:T,[(1-phi^2)/sig2; 1/sig2*ones(T-1,1)])*Hphi;
    deltah = Hphi\[mu; mu*(1-phi)*ones(T-1,1)];
    s2 = (y-tau).^2;
    logf_h = @(x) -.5*(x-deltah)'*HinvSH*(x-deltah) -.5*sum(x) ...
        -.5*exp(-x)'*(y-tau-alp.*exp(x)).^2;
    gradK_h = @(x) deal(-.5 + .5*(s2./exp(x)) - .5*(alp.^2.*exp(x)) - HinvSH*(x-deltah), ...
        HinvSH + spdiags(.5*(s2./exp(x)) + .5*(alp.^2.*exp(x)),0,T,T));
    [h, acc_h] = ssm.armh(h, logf_h, gradK_h, 'Tol', 1e-3, 'ForceAccept', loop == 1);
    if acc_h
        exph = exp(h);
        counth = counth + 1;
    end

    % sample Omega
    err = reshape(gam(3:end,:) - gam(1:end-2,:),2,T-1)';
    Omega = iwishrnd(SOmega + err'*err,nuOmega + T-1);

    % sample sig2
    errh = [(h(1)-mu)*sqrt(1-phi^2); h(2:end)-phi*h(1:end-1)-mu*(1-phi)];
    newSh = Sh + sum(errh.^2)/2;
    sig2 = 1/gamrnd(newnuh, 1./newSh);

    % sample mu and phi jointly by MH, with the proposal of proplam
    flam = @(x) .5*log(1-x(2)^2) - (1-x(2)^2)/(2*sig2)*(h(1)-x(1))^2 ...
        - 1/(2*sig2)*sum((h(2:end) - x(2)*h(1:end-1) - x(1)*(1-x(2))).^2)...
        - 1/(2*Vphi)*(x(2)-phi0)^2 - 1/(2*Vmu)*(x(1)-mu0)^2;
    [lamc,g] = proplam(h,sig2);
    MHprob = flam(lamc) - flam([mu; phi]) + g([mu; phi]) - g(lamc);
    if exp(MHprob) > rand
        mu = lamc(1);
        phi = lamc(2);
        countlam = countlam+1;
    end
    Hphi = ssm.diffmat(T, phi);

    if loop > burnin
        i = loop - burnin;
        store_h(i,:) = h';
        store_alp(i,:) = alp';
        store_theta(i,:) = [mu phi sig2 Omega([1 2 4])];
    end
end
thetahat = mean(store_theta)';
thetaCI = quantile(store_theta,[.05 .95])';
names = {'mu', 'phi', 'sigma2', 'omega2_alpha', 'omega_{alpha,tau}', 'omega2_tau'};
fprintf(['\n1. Quarterly US CPI inflation, 1948Q1-2025Q3, T = %d, %d draws after ' ...
    '%d burn-in\n'], T, nloop - burnin, burnin);
fprintf('   %-18s %15s   %s\n', 'parameter', 'posterior mean', '90% credible interval');
for j = 1:numel(names)
    fprintf('   %-18s %15.3f   (%.3f, %.3f)\n', names{j}, thetahat(j), thetaCI(j,:));
end
fprintf('   acceptance rates: h %.3f, (mu, phi) %.3f\n', counth/nloop, countlam/nloop);

%% 2. The acceptance rate against the envelope constant
ndraw = 1000;
fprintf(['\n2. Acceptance against the envelope constant, %d draws of h at each value,\n' ...
    '   from the target density of the last draw of h in the chain\n'], ndraw);
fprintf('   %9s %15s %21s\n', 'c_reject', 'MH acceptance', 'candidates per draw');
for c = [0.5 1 3 10]
    hc = h; nacc = 0; ntries = 0;
    for r = 1:ndraw
        [hc, a, nt] = ssm.armh(hc, logf_h, gradK_h, 'c_reject', c, 'Tol', 1e-3);
        nacc = nacc + a; ntries = ntries + nt;
    end
    fprintf('   %9g %15.3f %21.2f\n', c, nacc/ndraw, ntries/ndraw);
end

%% Figure: log-volatility and the volatility-in-mean coefficient
hq = quantile(store_h, [.05 .95])';
alpq = quantile(store_alp, [.05 .95])';
figure('Name', 'ex07 SVM');
subplot(1,2,1); hold on
ssm.shaded_band(tid, hq(:,1), hq(:,2));
plot(tid, mean(store_h)', 'k', 'LineWidth', 1.5);
hold off; box off; xlim([tid(1) tid(end)])
title('h_t: posterior mean and 90% band')
subplot(1,2,2); hold on
ssm.shaded_band(tid, alpq(:,1), alpq(:,2));
plot(tid, mean(store_alp)', 'k', 'LineWidth', 1.5);
plot(tid, zeros(T,1), '--k');
hold off; box off; xlim([tid(1) tid(end)])
title('\alpha_t: posterior mean and 90% band')
drawnow

fprintf('\nex07 done.\n');

function [lam, g] = proplam(h,sig2)
% proplam.m of chan2017_jbes_svm: a t proposal for (mu, phi), from a Newton-Raphson
% (BHHH) step on the conditional density of (mu, atanh(phi)), with the log density g of
% the proposal. The published loop stops at the first step larger than 1e-4 and
% never at its cap, so a first step below 1e-4 loops forever; here the cap stops it,
% and the fallback below then applies. The published code is otherwise unchanged.
maxcount = 100;
T = size(h,1);
s = zeros(T,2);
mut = mean(h);
phit = (h(1:end-1)-mut)'*(h(2:end)-mut)/sum((h(1:end-1)-mut).^2);
delt = [mut atanh(phit)]';
count = 0;
flag = 0;
while flag == 0
    s(1,1) = sech(delt(2))^2/sig2*(h(1)-delt(1));
    s(2:T,1) = (1-tanh(delt(2)))/sig2 * ...
        (h(2:end)-tanh(delt(2))*h(1:end-1) - delt(1)*(1-tanh(delt(2))));
    s(1,2) = sech(delt(2))^2*tanh(delt(2))/sig2 * (h(1)-delt(1));
    s(2:T,2) = sech(delt(2))^2/sig2 * (h(1:end-1)-delt(1)) .* ...
        (h(2:T)-delt(1)-tanh(delt(2))*(h(1:end-1)-delt(1)));
    S = sum(s)';
    B = s'*s;
    delt = delt + B\S;
    count = count + 1;
    if (sum(abs(B\S)) > 10^(-4) && count < maxcount) || count == maxcount
        flag = 1;
    end
end
[C,p] = chol(B,'lower');
if count == maxcount || p~=0
    delt = [mut atanh(phit)]';
    s(1,1) = sech(delt(2))^2/sig2*(h(1)-delt(1));
    s(2:T,1) = (1-tanh(delt(2)))/sig2 * ...
        (h(2:end)-tanh(delt(2))*h(1:end-1) - delt(1)*(1-tanh(delt(2))));
    s(1,2) = sech(delt(2))^2*tanh(delt(2))/sig2 * (h(1)-delt(1));
    s(2:T,2) = sech(delt(2))^2/sig2 * (h(1:end-1)-delt(1)) .* ...
        (h(2:T)-delt(1)-tanh(delt(2))*(h(1:end-1)-delt(1)));
    B = s'*s;
    C = chol(B,'lower');
end
% t proposal with df nu = 5
nu = 5;
del = delt + C'\randn(2,1)/sqrt(gamrnd(nu/2,2/nu));
lam = [del(1); tanh(del(2))];
c = gammaln((nu+2)/2) - gammaln(nu/2) - log(nu) - log(pi) + .5*log(det(B));
g = @(x) c + 2*log(cosh(atanh(x(2)))) ... % Jacobian of transformation
    - (nu+2)/2 * log( 1 + 1/nu * ([x(1) atanh(x(2))]-delt')*B*([x(1); atanh(x(2))]-delt));
end
