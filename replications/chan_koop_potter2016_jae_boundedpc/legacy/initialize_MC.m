% initialize the Markov chain 
if sv == 1
    sigh2 = .2;
elseif sv == 0
    sigh2 = 10^(-6);
end
sigu2 = .2;
sigtaupi2 = .02;
sigtauu2 = .01;
sigrhopi2 = .005;
if tvp_lam == 1
    siglam2 = .005;
elseif tvp_lam == 0    
    siglam2 = 10^(-6);
end
h = ones(T,1);
rhopi = .5*ones(T,1);
rhou = [1.6 -.7]';
lam = -.4*ones(T,1);

Kpi = speye(T) - sparse(2:T,1:(T-1),rhopi(2:T),T,T); 
mupi = Kpi\([rhopi(1)*(y0-taupi0); sparse(T-1,1)]);
expinvh = exp(-h);    
alppi = Kh\[taupi0; sparse(T-1,1)];
Staupi = Kh'*sparse(id1,id1,[invVtaupi repmat(1/sigtaupi2,1,T-1)],T,T)*Kh;
Sy = Kpi'*sparse(id1,id1,expinvh,T,T)*Kpi;
Hpi =  Sy + Staupi;
taut = Hpi\(Staupi*alppi + Sy*(y-mupi)); 
taut = min(taut,bpi-.1);    taut = max(taut,api+.1); 
count = 0; flag =0;
while flag == 0 && count<1000
    taupi = taut + chol(Hpi,'lower')'\randn(T,1);
    if  max(taupi) <= bpi - .01 && min(taupi) >= api + .01
        flag = 1;        
    end
    count = count + 1;
end
if count == 1000
    taupi = taut;
end

Ku = spdiags([ones(T,1) [-rhou(1)*ones(T-1,1); 0] [-rhou(2)*ones(T-2,1);0;0]], [0 -1 -2], T, T);   
muu = Ku\[rhou(1)*(u0(2)-tauu0(2))+rhou(2)*(u0(1)-tauu0(1)); rhou(2)*(u0(2)-tauu0(2)); sparse(T-2,1)];    
alpu = Kh\[tauu0(2); sparse(T-1,1)];
Su  = Ku'*sparse(id1,id1,[1/5 1/5 repmat(1/sigu2,1,T-2)])*Ku;     %% u1 and u2 have variances 5
Stauu = Kh'*sparse(id1,id1,[invVtauu repmat(1/sigtauu2,1,T-1)])*Kh;
Hu =  Su + Stauu;
taut = Hu\(Stauu*alpu + Su*(u-muu));
taut = min(taut,bu-.1);    tauu = max(taut,au+.1);
count = 0; flag =0;
while flag == 0 && count<1000
    tauu = taut + chol(Hu,'lower')'\randn(T,1);
    if  max(tauu) <= bu - .01 && min(tauu) >= au + .01
        flag = 1;        
    end
    count = count + 1;
end
if count == 1000
    tauu = taut;
end
pistar = y - taupi;
ustar = u - tauu;

Sh = spdiags([invVh; 1/sigh2*ones(T-1,1)],0,T,T);
KSKh = Kh'*Sh*Kh;    
s2 = (pistar - rhopi.*[(y0-taupi0);pistar(id2)]).^2;
h = mean(log(s2)) + std(log(s2))*randn(T,1);
errh = 1; ht = h;
while errh> 10^(-3);
    expht = exp(ht);
    sexpht = s2./expht;
    fh = -.5 + .5*sexpht;
    Gh = .5*sexpht;
    Hh = KSKh + spdiags(Gh,0,T,T);
    newht = Hh\(fh+Gh.*ht);
    errh = max(abs(newht-ht));
    ht = newht;          
end 
h = ht;

sigtaupi = sqrt(sigtaupi2);
sigtauu = sqrt(sigtauu2);
sigrhopi = sqrt(sigrhopi2);
siglam = sqrt(siglam2);
