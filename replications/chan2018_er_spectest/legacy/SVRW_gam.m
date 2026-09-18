% This function draws the log volatilities in an unobserved components 
% model with a gamma prior on the error variance.
%
% See:

% Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter 
% Models with Stochastic Volatility, Econometric Reviews, 37(8), 807-823

% Inputs: b0 and Vh0 are the prior mean and variance of h0;
%         Vh is the initial variance of h_1
%         Vomegah is the prior variance of omegah
function [htilde,h0,omegah,omegahhat,Domegah] = SVRW_gam(Ystar,htilde,h0,omegah,b0,Vh0,Vh,Vomegah)

    T = length(htilde);
    %% normal mixture
    pj = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
    mj = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
    sigj = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
    sqrtsigj = sqrt(sigj);

    %% sample S from a 7-point distrete distribution
    temprand = rand(T,1);
    q = repmat(pj,T,1).*normpdf(repmat(Ystar,1,7),repmat(h0+omegah*htilde,1,7)+repmat(mj,T,1), repmat(sqrtsigj,T,1));
    q = q./repmat(sum(q,2),1,7);
    S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;

    %% sample htilde
    % y^* = h0 + omegah htilde + d + \epsilon, \epsilon \sim N(0,\Omega),
    % H htilde = \nu, \nu \ sim N(0,S),
    % where d_t = Ez_t, \Omega = diag(\omega_1,\ldots,\omega_n), 
    % \omega_t = var z_t, S = diag(Vh, 1, \ldots, 1)
    Hh = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
    invSh = sparse(1:T,1:T,[1/Vh; ones(T-1,1)]);
    dconst = mj(S)'; invOmega = sparse(1:T,1:T,1./sigj(S));
    Kh = Hh'*invSh*Hh + invOmega*omegah^2;
    htildehat = Kh\(invOmega*omegah*(Ystar-dconst-h0));
    htilde = htildehat + chol(Kh,'lower')'\randn(T,1);

    %% sample h0 and omegah
    Xbeta = [ones(T,1) htilde];
    invVbeta = diag([1/Vh0 1/Vomegah]);
    XbetainvOmega = Xbeta'*invOmega;
    invDbeta = invVbeta + XbetainvOmega*Xbeta;
    betahat = invDbeta\(invVbeta*[b0;0] + XbetainvOmega*(Ystar-dconst));
    beta = betahat + chol(invDbeta,'lower')'\randn(2,1);
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