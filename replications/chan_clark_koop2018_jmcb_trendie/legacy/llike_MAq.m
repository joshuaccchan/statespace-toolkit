% Support function to estimate the trend inflation models in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

function lden = llike_MAq(psi,y,sig2)
T = length(y);
Hpsi = buildHpsi(psi,T);
L = sig2*(Hpsi*Hpsi');
lden = -T/2*log(2*pi*sig2) - .5*y'*(L\y);
end