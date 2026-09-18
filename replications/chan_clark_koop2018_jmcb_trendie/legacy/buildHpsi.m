% Support function to estimate the trend inflation models in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

function Hpsi = buildHpsi(psi,T)
q = length(psi);
Hpsi = speye(T);
for j=1:q
    Hpsi = Hpsi + psi(j)*sparse(j+1:T,1:T-j,ones(1,T-j),T,T);
end
end