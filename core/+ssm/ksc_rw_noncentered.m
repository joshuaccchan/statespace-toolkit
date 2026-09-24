% ssm.ksc_rw_noncentered - KSC auxiliary-mixture sampler for a random-walk
% log-volatility in the noncentered parameterization h_t = h0 + omegah*htilde_t:
%   Ystar_t = h0 + omegah*htilde_t + eps_t,  eps_t = log of a chi^2_1 draw,
%             approximated by the Kim-Shephard-Chib (1998) 7-component mixture,
%   htilde_t = htilde_{t-1} + v_t,  v_t ~ N(0,1),  htilde_1 ~ N(0,Vh),
% with priors h0 ~ N(b0,Vh0) and omegah ~ N(0,Vomegah) (all VARIANCES).
%
%   [htilde, h0, omegah, omegahhat, Domegah] = ...
%       ssm.ksc_rw_noncentered(Ystar, htilde, h0, omegah, b0, Vh0, Vh, Vomegah)
%
%   Ystar and htilde are T x 1. h0 is the level of h, not a value at time 0 as in
%   ssm.ksc_rw_h0. One call draws the mixture indicators, then htilde, then
%   (h0, omegah) jointly, then flips the signs of htilde and omegah together with
%   probability 1/2; the flip leaves h unchanged, and the posterior of omegah is
%   symmetric about 0. omegahhat and Domegah are the mean and variance of the
%   Gaussian conditional posterior of omegah given Ystar, the indicators and htilde,
%   with h0 integrated out, computed before the flip. Their density at 0 does not
%   depend on the flip; averaged over draws, it estimates the posterior density of
%   omegah at 0 in the Savage-Dickey ratio for omegah = 0 (Chan 2018).
%   Random numbers per call: rand(T,1), randn(T,1), randn(2,1), rand.
%
% Extracted from SVRW_gam.m of chan2018_er_spectest, with three edits: the
% function name; the posterior means of htilde and of (h0, omegah) are solved with
% the Cholesky factor that the draw uses, where SVRW_gam.m factors each matrix
% again with backslash; and the comments.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 10.1.1.
% Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter Models with
% Stochastic Volatility, Econometric Reviews, 37(8): 807-823.
% Fruhwirth-Schnatter, S. and Wagner, H. (2010). Stochastic Model Specification
% Search for Gaussian and Partial Non-Gaussian State Space Models, Journal of
% Econometrics, 154(1): 85-100.
% Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
% Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
% 361-393.

function [htilde,h0,omegah,omegahhat,Domegah] = ksc_rw_noncentered(Ystar,htilde,h0,omegah,b0,Vh0,Vh,Vomegah)

    T = length(htilde);
    %% normal mixture
    pj = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
    mj = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
    sigj = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
    sqrtsigj = sqrt(sigj);

    %% sample S from a 7-point discrete distribution
    temprand = rand(T,1);
    q = repmat(pj,T,1).*normpdf(repmat(Ystar,1,7),repmat(h0+omegah*htilde,1,7)+repmat(mj,T,1), repmat(sqrtsigj,T,1));
    q = q./repmat(sum(q,2),1,7);
    S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;

    %% sample htilde
    % Given the indicators S: Ystar = h0 + omegah*htilde + d + e with
    % e ~ N(0,Omega), where d_t and Omega_tt are the mean and variance of mixture
    % component S_t; and Hh*htilde = v with v ~ N(0,diag(Vh,1,...,1)).
    Hh = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
    invSh = sparse(1:T,1:T,[1/Vh; ones(T-1,1)]);
    dconst = mj(S)'; invOmega = sparse(1:T,1:T,1./sigj(S));
    Kh = Hh'*invSh*Hh + invOmega*omegah^2;
    CKh = chol(Kh,'lower');
    htildehat = (CKh')\(CKh\(invOmega*omegah*(Ystar-dconst-h0)));
    htilde = htildehat + CKh'\randn(T,1);

    %% sample h0 and omegah
    Xbeta = [ones(T,1) htilde];
    invVbeta = diag([1/Vh0 1/Vomegah]);
    XbetainvOmega = Xbeta'*invOmega;
    invDbeta = invVbeta + XbetainvOmega*Xbeta;
    CDbeta = chol(invDbeta,'lower');
    betahat = (CDbeta')\(CDbeta\(invVbeta*[b0;0] + XbetainvOmega*(Ystar-dconst)));
    beta = betahat + CDbeta'\randn(2,1);
    h0 = beta(1); omegah = beta(2);

    U = -1 + 2*(rand>0.5);
    htilde = U*htilde;
    omegah = U*omegah;

    %% compute the mean and variance of the conditional posterior of omegah
    Xbeta = [ones(T,1) htilde];
    Dbeta = (invVbeta + Xbeta'*invOmega*Xbeta)\speye(2);
    omegahhat = betahat(2);
    Domegah = Dbeta(2,2);
end
