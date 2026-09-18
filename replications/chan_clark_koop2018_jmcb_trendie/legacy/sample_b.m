% Support function to estimate the trend inflation models in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

function [b,accept] = sample_b(b,pi,pistar,lamv,pi0,Vb,sigb2)
accept = 0;
T = size(pi,1);
H =  speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
pitilde = pi - pistar;
Xb = sparse(1:T,1:T,[pi0; pitilde(1:T-1)]);
iLamv = sparse(1:T,1:T,1./lamv);  
HiSbH = H'*sparse(1:T,1:T,[1/Vb repmat(1/sigb2,1,T-1)])*H;
Kb = Xb'*iLamv*Xb + HiSbH;
bhat = Kb\(Xb'*iLamv*pitilde);
cholKb = chol(Kb,'lower'); 
    % AR step        
sigb = sqrt(sigb2);
gb = @(x) -sum(log(normcdf((1-x(1:end-1))/sigb) - normcdf(-x(1:end-1)/sigb)));    
bstar = bhat;
u = pitilde-Xb*bstar; v = bhat-bstar;
logc = -.5*(u./lamv)'*u -.5*bstar(1)^2/Vb ...
    -.5*sum((bstar(2:end)-bstar(1:end-1)).^2)/sigb2 + gb(bstar) ...
    + .5*v'*Kb*v + log(3); 
flag = 0; count = 0;
while flag == 0 && count < 100 % give up when count >= 100
    bc = bhat + cholKb'\randn(T,1);
    if  max(bc)<= .995 && min(bc)>= .005
        uc =  pitilde-Xb*bc; vc = bhat-bc;
        alpARc =  -.5*(uc./lamv)'*uc -.5*bc(1)^2/Vb +...
            -.5*sum((bc(2:end)- bc(1:end-1)).^2)/sigb2 + gb(bc) ...
            + .5*vc'*Kb*vc - logc;
        if alpARc > log(rand)
            flag = 1;
        end
    end
    count = count + 1;
end      
if flag == 1
    u = pitilde-Xb*b; v = bhat-b;
    alpAR = -.5*(u./lamv)'*u -.5*b(1)^2/Vb +...
            -.5*sum((b(2:end)- b(1:end-1)).^2)/sigb2 + gb(b) ... 
            + .5*v'*Kb*v - logc; 
    if alpAR < 0 
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end        
    if alpMH > log(rand)
        b = bc;
        accept = 1;
    end
end