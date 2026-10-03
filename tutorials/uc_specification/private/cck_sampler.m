function out = cck_sampler(pi, pi0, z, nsim, burnin)
% cck_sampler - Gibbs sampler for M1 of Chan, Clark and Koop (2018), inflation pi_t
% with the long-run expectation z_t:
%   pi_t - pistar_t = b_t (pi_{t-1} - pistar_{t-1}) + v_t,  v_t ~ N(0, exp(lv_t)),
%   z_t = d0_t + d1_t pistar_t + w_t + psi w_{t-1},  w_t ~ N(0, sigw2),  w_0 = 0,
%   pistar_t = pistar_{t-1} + n_t,  n_t ~ N(0, exp(ln_t)) for t > 1,  pistar_1 ~ N(0, 5),
%   b_t | b_{t-1} ~ N(b_{t-1}, sigb2) truncated to (0,1),  b_1 ~ N(0, Vb) on (0,1),
%   d_t = mud + rhod.*(d_{t-1} - mud) + N(0, diag(sigd2)),  d_1 from its stationary law,
%   lv_t (t >= 1) and ln_t (t >= 2) random walks with variances phiv, phin,
%   lv_1, ln_2 ~ N(0, 5),
% with pi_0 = pi0 and pistar_0 = 0 in the first gap. The priors of pistar_1, b, the
% log-volatilities and their variances are the tutorial's common priors, those of the
% bounded trend model; the rest are those of the paper's main_forecasting.m. pi, z are
% T x 1, from the quarter after pi0.
%
% Blocks, in the order of forecast_M1.m: pistar by ssm.simulate_states, with the MA
% error of z projected out through Hpsi\ as in the published code (entries below
% 1e-6 dropped); b in blocks of five (draw_b); d by ssm.simulate_states; psi by
% ma1_psi; mud; rhod (independence Metropolis-Hastings); lv and ln by
% ssm.ksc_rw_diffuse; sigb2 (Metropolis-Hastings for the truncation constants of
% b_1, ..., b_{T-1}); sigd2; sigw2; phiv; phin. d_1 has its stationary precision
% (1-rhod^2)/sigd2 in every d step.
%   out.theta : nsim x 11 [psi mud' rhod' sigd2' sigb2 sigw2 phiv phin]
%   out.last  : nsim x 4 [pistar_T b_T lv_T ln_T], the states the forecasts start from
%   out.pistar: posterior mean of the path pistar
%   out.acc   : acceptance rates of the blocks of b, of psi and of sigb2
T = numel(pi);
% priors: Vpi1, Vb, nub0/Sb0, nuv0/Sv0, nun0/Sn0 and Vlam are the common ones, the
% rest those of main_forecasting.m
Vpi1 = 5; Vb = 1; Vpsi = .25^2;
mud0 = [0 1]'; Vmud = [.1^2 .1^2]';
rhod0 = [.95 .95]'; Vrhod = [.1^2 .1^2]';
nud0 = 5*ones(2,1); Sd0 = [.01 .001]'.*(nud0-1);
nub0 = 10; Sb0 = .001*(nub0-1);
nuw0 = 5; Sw0 = .01*(nuw0-1);
nuv0 = 10; Sv0 = .05*(nuv0-1);
nun0 = 10; Sn0 = .05*(nun0-1);
Vlam = 5;
% initial values (forecast_M1.m)
sigd2 = .001*ones(2,1); sigb2 = .01; sigw2 = .2; phiv = .1; phin = .1;
lv = log(var(pi)/2)*ones(T,1); ln = lv;
mud = [0 1]'; rhod = [.98 .98]';
d = repmat(mud, T, 1);
b = .3 + .05*rand(T,1);
psi = 0;
H = ssm.diffmat(T);
Hpsi = speye(T);

out.theta = zeros(nsim, 11); out.last = zeros(nsim, 4); out.pistar = zeros(T, 1);
acc = [0 0 0]; L = 5;
for loop = 1:nsim + burnin
    % pistar
    Hb = speye(T) - sparse(2:T, 1:T-1, b(2:T), T, T);
    cb = [b(1)*pi0; zeros(T-1,1)];
    iLv = spdiags(exp(-lv), 0, T, T);
    D = reshape(d, 2, T)';
    Xp = Hpsi\spdiags(D(:,2), 0, T, T); Xp = Xp.*(abs(Xp) > 1e-6);
    zt = Hpsi\(z - D(:,1));
    Kp = Hb'*iLv*Hb + Xp'*Xp/sigw2 + H'*spdiags([1/Vpi1; exp(-ln(2:end))], 0, T, T)*H;
    pistar = ssm.simulate_states(Kp, Hb'*iLv*(Hb*pi - cb) + Xp'*zt/sigw2);

    % b, in blocks of L at a random offset
    gap = pi - pistar; xb = [pi0; gap(1:T-1)];
    Kb = spdiags(xb.^2.*exp(-lv), 0, T, T) + H'*spdiags([1/Vb; ones(T-1,1)/sigb2], 0, T, T)*H;
    [b, a] = draw_b(b, Kb, xb.*gap.*exp(-lv), sqrt(sigb2), L);
    acc(1) = acc(1) + a;

    % d
    Xd = ssm.surform([ones(T,1) pistar]);
    Xt = Hpsi\Xd; Xt = Xt.*(abs(Xt) > 1e-6);
    Hr = speye(2*T) - sparse(3:2*T, 1:2*T-2, repmat(rhod, T-1, 1), 2*T, 2*T);
    Pd = Hr'*spdiags([(1 - rhod.^2)./sigd2; repmat(1./sigd2, T-1, 1)], 0, 2*T, 2*T)*Hr;
    deld = Hr\[mud; repmat((1 - rhod).*mud, T-1, 1)];
    d = ssm.simulate_states(Pd + Xt'*Xt/sigw2, Pd*deld + Xt'*(Hpsi\z)/sigw2);
    D = reshape(d, 2, T)';

    % psi
    [psi, a] = ma1_psi(psi, z - Xd*d, ones(T,1)/sigw2, Vpsi, .99);
    acc(2) = acc(2) + a;
    Hpsi = ssm.diffmat(T, -psi);

    % mud and rhod
    for i = 1:2
        Dm = 1/(1/Vmud(i) + ((T-1)*(1 - rhod(i))^2 + (1 - rhod(i)^2))/sigd2(i));
        mh = Dm*(mud0(i)/Vmud(i) + (1 - rhod(i)^2)/sigd2(i)*D(1,i) ...
            + (1 - rhod(i))/sigd2(i)*sum(D(2:end,i) - rhod(i)*D(1:end-1,i)));
        mud(i) = mh + sqrt(Dm)*randn;
    end
    for i = 1:2
        X = D(1:end-1,i) - mud(i); Y = D(2:end,i) - mud(i);
        Dr = 1/(1/Vrhod(i) + X'*X/sigd2(i));
        rc = Dr*(rhod0(i)/Vrhod(i) + X'*Y/sigd2(i)) + sqrt(Dr)*randn;
        g = @(x) -.5*log(sigd2(i)/(1 - x^2)) - .5*(1 - x^2)/sigd2(i)*(D(1,i) - mud(i))^2;
        if abs(rc) < .98 && exp(g(rc) - g(rhod(i))) > rand
            rhod(i) = rc;
        end
    end

    % the log-volatilities
    lv = ssm.ksc_rw_diffuse(log((gap - b.*xb).^2 + .0001), lv, phiv, Vlam);
    lnn = ssm.ksc_rw_diffuse(log(diff(pistar).^2 + .0001), ln(2:end), phin, Vlam);
    ln = [lnn(1); lnn];                          % ln(1) is not used

    % sigb2, with the truncation constants of b_1, ..., b_{T-1}
    e2 = diff(b).^2;
    sc = 1/gamrnd(nub0 + (T-1)/2, 1/(Sb0 + sum(e2)/2));
    lz = @(s) sum(log(normcdf((1 - b(1:T-1))/sqrt(s)) - normcdf(-b(1:T-1)/sqrt(s))));
    if lz(sigb2) - lz(sc) > log(rand)
        sigb2 = sc; acc(3) = acc(3) + 1;
    end

    % sigd2, sigw2, phiv, phin
    for i = 1:2
        e = [(D(1,i) - mud(i))*sqrt(1 - rhod(i)^2); D(2:end,i) - rhod(i)*D(1:end-1,i) - mud(i)*(1 - rhod(i))];
        sigd2(i) = 1/gamrnd(nud0(i) + T/2, 1/(Sd0(i) + e'*e/2));
    end
    e = Hpsi\(z - Xd*d);
    sigw2 = 1/gamrnd(nuw0 + T/2, 1/(Sw0 + e'*e/2));
    phiv = 1/gamrnd(nuv0 + (T-1)/2, 1/(Sv0 + sum(diff(lv).^2)/2));
    phin = 1/gamrnd(nun0 + (T-2)/2, 1/(Sn0 + sum(diff(ln(2:end)).^2)/2));

    if loop > burnin
        i = loop - burnin;
        out.theta(i,:) = [psi mud' rhod' sigd2' sigb2 sigw2 phiv phin];
        out.last(i,:) = [pistar(T) b(T) lv(T) ln(T)];
        out.pistar = out.pistar + pistar/nsim;
    end
end
out.acc = acc/(nsim + burnin);
end

function [b, rate] = draw_b(b, K, r, sb, L)
% one sweep over the path b in blocks of L: each block is proposed from its Gaussian
% conditional given the rest, N(K_CC^{-1}(r_C - K_{C,-C} b_{-C}), K_CC^{-1}), and kept
% if it lies in (0,1) and passes the Metropolis-Hastings step for the truncation
% constants of b_t | b_{t-1}, which depend on b_t through 1/A(b_t),
% A(x) = Phi((1-x)/sb) - Phi(-x/sb), for t < T. Returns the share of blocks kept.
T = numel(b);
edges = unique([1, (1 + randi(L) - 1):L:T, T + 1]);
nb = numel(edges) - 1; kept = 0;
for j = 1:nb
    C = edges(j):edges(j+1)-1;
    KC = K(C,C);
    rhs = r(C) - K(C,:)*b + KC*b(C);
    Ch = chol(KC, 'lower');
    bc = Ch'\(Ch\rhs) + Ch'\randn(numel(C),1);
    u = rand;
    if all(bc > 0 & bc < 1)
        t = C(C < T);
        la = sum(logA(b(t), sb)) - sum(logA(bc(C < T), sb));
        if log(u) < la
            b(C) = bc; kept = kept + 1;
        end
    end
end
rate = kept/nb;
end

function v = logA(x, sb)
v = log(normcdf((1 - x)/sb) - normcdf(-x/sb));
end
