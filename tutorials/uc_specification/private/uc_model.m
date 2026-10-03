function m = uc_model(name)
% uc_model - the unobserved components models of the tutorial. All but 'ARbound'
% share one form,
%   y_t = tau_t + u_t + psi*u_{t-1},  u_t ~ N(0, exp(h_t)),  u_0 = 0,
%   tau_t = tau_{t-1} + e_t,  e_t ~ N(0, exp(g_t)),  tau_1 ~ N(tau0, V1a + V1b*exp(g_1)),
% where each log variance, h (the gap) or g (the trend), is
%   'const' : a constant log(s2), with s2 ~ IG(nu, S);
%   'rw'    : a random walk x_t = x_{t-1} + N(0, s2), x_1 ~ N(x0, s2), x0 ~ N(m0, V0),
%             s2 ~ IG(nu, S);
%   'ar1'   : x_t = mu + phi*(x_{t-1} - mu) + N(0, s2), x_1 ~ N(mu, s2/(1-phi^2)),
%             mu ~ N(m0, V0), phi ~ N(phi0, Vphi) on (-1,1), s2 ~ IG(nu, S);
% tau0 ~ N(a0, b0), or fixed at a0 when b0 = 0; and psi ~ N(0,1) on (-1,1) when the
% gap has the MA term. IG(nu, S) has density proportional to x^-(nu+1)*exp(-S/x).
% 'ARbound' is the model of Chan, Koop and Potter (2013) (see ckp_run).
% m.free lists the free parameters on the real line: log variances, atanh of phi and
% psi, and the means as they are.
% The models share their priors: a constant trend variance is IG(10, 0.18), every
% log-volatility innovation variance IG(10, 0.45), a log-volatility starts from a
% level with variance 5, and tau_1 ~ N(0, 5) (in UCSV, tau_1 ~ N(tau0, exp(g_1)) with
% tau0 ~ N(0, 5)), the priors of Chan (2013) and of the bounded trend model.
sv = struct('type', 'rw', 'm0', 0, 'V0', 5, 'nu', 10, 'S', .45);
switch name
    case 'UC'
        m.gap = struct('type', 'const', 'nu', 3, 'S', 2);
        m.trend = struct('type', 'const', 'nu', 10, 'S', .18);
        m.tau1 = struct('a0', 0, 'b0', 0, 'V1a', 5, 'V1b', 0);
        m.ma = 0;
    case 'UC-MA'        % Chan (2013)
        m.gap = struct('type', 'ar1', 'm0', 0, 'V0', 5, 'phi0', .9, 'Vphi', 1, 'nu', 10, 'S', .45);
        m.trend = struct('type', 'const', 'nu', 10, 'S', .18);
        m.tau1 = struct('a0', 0, 'b0', 0, 'V1a', 5, 'V1b', 0);
        m.ma = 1;
    case 'UC-SVgap'     % UC with a random-walk log-volatility in the gap
        m.gap = sv;
        m.trend = struct('type', 'const', 'nu', 10, 'S', .18);
        m.tau1 = struct('a0', 0, 'b0', 0, 'V1a', 5, 'V1b', 0);
        m.ma = 0;
    case 'UCSV'         % Stock and Watson (2007)
        m.gap = sv;
        m.trend = sv;
        m.tau1 = struct('a0', 0, 'b0', 5, 'V1a', 0, 'V1b', 1);
        m.ma = 0;
    case 'ARbound'      % Chan, Koop and Potter (2013)
        m.free = {'log s2_tau', 'log s2_rho', 'log s2_h'};
        m.name = name;
        return
    otherwise
        error('uc_model: no model %s', name);
end
m.name = name;
m.free = {};
for c = {'gap', 'trend'}
    switch m.(c{1}).type
        case 'const', m.free = [m.free {[c{1} ' log s2']}];
        case 'rw',    m.free = [m.free {[c{1} ' x0'], [c{1} ' log s2']}];
        case 'ar1',   m.free = [m.free {[c{1} ' mu'], [c{1} ' atanh phi'], [c{1} ' log s2']}];
    end
end
if m.ma, m.free = [m.free {'atanh psi'}]; end
end
