% negative log conditional kernel of psi
% the truncation is managed in the MH step
function fpsi = sy_CS5_fpsi(logz, a, g, psi, sigma2z, mpsi, Kpsi, T)
    x0 = logz - a(1) - a(2) *g;
    Hpsi = speye(T) + spdiags(zeros(T-1,1)+psi,-1,T,T);
    x1 = Hpsi \ x0;
    fpsi = x1'*x1 / (2*sigma2z) + ... %likelihood part
           (psi-mpsi)^2 * Kpsi / 2; % prior part
end