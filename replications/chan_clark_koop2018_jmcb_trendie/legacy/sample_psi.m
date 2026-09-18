% Support function to estimate the trend inflation models in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

function [psi,flag,psihat,invDpsic] = sample_psi(psi,fpsi,loop,invDpsic,options)
q = length(psi);
psihat = fminsearch(fpsi,psi);
Cpsi = chol(invDpsic,'lower');
if (mod(loop,100)==0) || loop == 1 %% get the Hessian every 100 iterations
    [psihat,fval,exitflag,output,grad,hess] = fminunc(fpsi,psihat,options); 
    [tmpCpsi,p] = chol(hess,'lower');
    if p == 0
        invDpsic = hess;
        Cpsi = tmpCpsi;
    end        
end
psic = psihat + Cpsi'\randn(q,1); 
if sum(abs(1./roots([flipud(psic);1]))<.99) == q
    alpMH = -fpsi(psic) + fpsi(psi) ...
        - .5*(psi-psihat)'*invDpsic*(psi-psihat) ...
        + .5*(psic-psihat)'*invDpsic*(psic-psihat);
else
    alpMH = -inf;
end
flag = alpMH>log(rand);
if flag
    psi = psic;
end   