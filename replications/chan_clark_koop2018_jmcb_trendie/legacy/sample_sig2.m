% Support function to estimate the trend inflation models in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

function sig2 = sample_sig2(e2,nu0,S0)
n = size(e2,1);
sig2 = 1./gamrnd(nu0+n/2, 1./(S0+sum(e2)'/2)); 
end