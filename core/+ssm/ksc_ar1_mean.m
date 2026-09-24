% ssm.ksc_ar1_mean - KSC auxiliary-mixture sampler for the log-volatility path,
% stationary AR(1) state equation with mean mu:
%   ystar_t = h_t + eps_t,  eps_t = log of a chi^2_1 draw, approximated by the
%             Kim-Shephard-Chib (1998) 7-component normal mixture,
%   h_t = mu + rho*(h_{t-1} - mu) + v_t,  v_t ~ N(0,sig2),
%   h_1 ~ N(mu, sig2/(1-rho^2)),  with sig2 a VARIANCE.
%
%   [h, S] = ssm.ksc_ar1_mean(ystar, h, mu, rho, sig2)
%
%   ystar is T x 1; h on input is the current path and on output the new draw; S is
%   the T x 1 vector of mixture indicators, in 1:7, drawn alongside h. Random numbers
%   per call: rand(T,1), then randn(T,1).
%
% The path is drawn in one block from its banded precision (Chan and Jeliazkov
% 2009). Code-identical to bvar.sv.ksc_ar1_mean in bvar-toolkit, and the two must
% stay so; it draws what SV.m of chan2013_joe_masv draws, with the arguments in
% another order. Not interchangeable with ssm.ksc_rw_h0: another state equation.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 10.1.1.
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with
% Application to Inflation Forecast, Journal of Econometrics, 176(2): 162-172.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
% Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
% 361-393.

function [h,S] = ksc_ar1_mean(ystar,h,mu,rho,sig2)
T = length(h);
    % 7-component normal mixture
p_N = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
m_N = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  % means already adjusted!!
sig2_N = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];

    % sample S from a 7-point distrete distribution
tmprand = rand(T,1);
q = repmat(p_N,T,1).*normpdf(repmat(ystar,1,7),repmat(h,1,7)+repmat(m_N,T,1),...
    repmat(sqrt(sig2_N),T,1));
q = q./repmat(sum(q,2),1,7);
S = 7 - sum(repmat(tmprand,1,7)<cumsum(q,2),2)+1;

    % sample h using the precision sampler
% y^* = h + d + \epsilon, \epsilon \sim N(0,\Omega)
% h  ~ N(mu 1_T, sig2(Hrho' S^{-1} Hrho)^{-1}),
% where S^{-1} = diag(1-rho^2,1,1,...,1)

Hrho = speye(T) - sparse(2:T,1:(T-1),rho*ones(1,T-1),T,T);
HiSH = Hrho'*sparse(1:T,1:T,[1-rho^2, ones(1,T-1)])*Hrho;
d = m_N(S)'; iOmega = sparse(1:T,1:T,1./sig2_N(S));
Kh = HiSH/sig2 + iOmega;
CKh = chol(Kh,'lower');
h_hat = (CKh')\(CKh\(mu/sig2*HiSH*ones(T,1) + iOmega*(ystar-d)));
h = h_hat + CKh'\randn(T,1);
end
