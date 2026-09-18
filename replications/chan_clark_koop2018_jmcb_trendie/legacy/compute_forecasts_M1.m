% This subroutine computes the forecasts from the M1 model in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

b_new = zeros(40,1);
pistar_new = zeros(40,1);
lamv_new = zeros(40,1);
lamn_new = zeros(40,1);

for tt=1:40  % generate future states
    if tt==1
        lamv_new(tt) = exp(log(lamv(end))+sqrt(phiv)*randn);
        lamn_new(tt) = exp(log(lamn(end))+sqrt(phin)*randn);
        b_new(tt) = tnormrnd(b(end),sigb2,0,1,1);                   
        pistar_new(tt) = pistar(end) + sqrt(lamn_new(tt))*randn;
    else
        lamv_new(tt) = exp(log(lamv_new(tt-1))+sqrt(phiv)*randn);
        lamn_new(tt) = exp(log(lamn_new(tt-1))+sqrt(phin)*randn);
        b_new(tt) = tnormrnd(b_new(tt-1),sigb2,0,1,1); 
        pistar_new(tt) = pistar_new(tt-1) + sqrt(lamn_new(tt))*randn;
    end               
end            
    % compute point and density forecasts
Epi = zeros(12,1); % point forecasts for h=1,2,4,8,12,16,20,24,28,32,36,40
Vpi = zeros(12,1);            
pi_t = pi(t);
pistar_t = pistar(end);            
Epi(1) = pistar_new(1)+b_new(1)*(pi_t-pistar_t);
Vpi(1) = lamv_new(1);
Epi(2) = .5*(sum(pistar_new(1:2))+b_new(1)*(1+b_new(2))*(pi_t-pistar_t));
Vpi(2) = .25*((1+b_new(2))^2*lamv_new(1)+lamv_new(2));
Epi(3) = .25*(sum(pistar_new(1:4))+b_new(1)*(1+b_new(2)...
    +b_new(2)*b_new(3)+b_new(2)*b_new(3)*b_new(4))*(pi_t-pistar_t));
Vpi(3) = 1/16*((1+b_new(2)+b_new(2)*b_new(3)+b_new(2)*b_new(3)*b_new(4))^2*lamv_new(1) ...
    +(1+b_new(3)+b_new(3)*b_new(4))^2*lamv_new(2)...
    +(1+b_new(4))^2*lamv_new(3)+lamv_new(4));           
for tt=1:36
    Epi_t = pistar_new(tt) + b_new(tt)*(pi_t-pistar_t);
    pi_t = Epi_t + sqrt(lamv_new(tt))*randn;
    pistar_t = pistar_new(tt);                
    if mod(tt,4)==0
        Epi(3+tt/4) = .25*(sum(pistar_new(tt+1:tt+4))+b_new(tt+1)*(1+b_new(tt+2)...
            +b_new(tt+2)*b_new(tt+3)+b_new(tt+2)*b_new(tt+3)*b_new(tt+4))*(pi_t-pistar_t));
        Vpi(3+tt/4) = 1/16*((1+b_new(tt+2)+b_new(tt+2)*b_new(tt+3)+b_new(tt+2)*b_new(tt+3)*b_new(tt+4))^2*lamv_new(tt+1) ...
            +(1+b_new(tt+3)+b_new(tt+3)*b_new(tt+4))^2*lamv_new(tt+2)...
            +(1+b_new(tt+4))^2*lamv_new(tt+3)+lamv_new(tt+4));
    end               
end