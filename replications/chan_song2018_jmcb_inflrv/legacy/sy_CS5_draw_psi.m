% This is a simplified version of Chan, Clark and Koop (2017)
% Only MA(1)
function [psi,flag,psihat,invDpsic] = sy_CS5_draw_psi(psi,fpsi,loop,invDpsic,options)
    
    psihat = fminsearch(fpsi,psi);
    Cpsi = sqrt(invDpsic);
    if (mod(loop,100)==0) || loop == 1 %% get the Hessian every 100 iterations
        [psihat,fval,exitflag,output,grad,hess] = fminunc(fpsi,psihat,options); 
        if hess > 0
            invDpsic = hess;
            Cpsi = sqrt(hess);
        end        
    end
    psic = psihat + randn / Cpsi; 
    if abs(psic)<.99
        alpMH = -fpsi(psic) + fpsi(psi) ...
            - .5 * (psi-psihat)^2 * invDpsic...
            + .5 * (psic-psihat)^2 * invDpsic;
    else
        alpMH = -inf;
    end
    flag = alpMH>log(rand);
    if flag
        psi = psic;
    end
end