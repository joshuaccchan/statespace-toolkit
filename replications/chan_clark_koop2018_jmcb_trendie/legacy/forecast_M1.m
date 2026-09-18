% This script estimates the M1 model in Chan, Clark and Koop (2018) and 
% computes the corresponding forecasts
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

tmpyhat1 = zeros(nsim,2);  %% [point forecast prelike]
tmpyhat2 = zeros(nsim,2); 
tmpyhat4 = zeros(nsim,2); 
tmpyhat8 = zeros(nsim,2); 
tmpyhat12 = zeros(nsim,2); 
tmpyhat16 = zeros(nsim,2); 
tmpyhat20 = zeros(nsim,2); 
tmpyhat40 = zeros(nsim,2); 

% initialize the Markov chain       
sigd2 = .001*ones(2,1);
sigb2 = .01;
sigb = sqrt(sigb2);
sigw2 = .2;
phiv = .1;
phin = .1;
lamv = var(pit)/2*ones(Tt,1);
lamw = var(zt)*ones(Tt,1);
invLamw = sparse(1:Tt,1:Tt,1./lamw);
lamn = var(pit)/2*ones(Tt,1);
mud = [0 1]';
rhod = [.98 .98]';
shortd = repmat(mud',Tt,1);
d = reshape(shortd',Tt*2,1);
b = .3 + .05*rand(Tt,1);
psi = zeros(q,1);
psihat = psi;
invDpsic = .0001*eye(q);
H =  speye(Tt) - sparse(2:Tt,1:(Tt-1),ones(1,Tt-1),Tt,Tt);
Hrhod =  speye(2*Tt) - sparse(2+1:Tt*2,1:(Tt-1)*2,repmat(rhod,Tt-1,1),2*Tt,2*Tt);
Hpsi = buildHpsi(psi,Tt);

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
for isim = 1:nsim

        % sample pistar     
    Hb = speye(Tt) - sparse(2:Tt,1:(Tt-1),b(2:Tt),Tt,Tt); 
    alppi = Hb\[b(1)*pi0; sparse(Tt-1,1)];
    HbiLamvHb = Hb'*sparse(1:Tt,1:Tt,1./lamv)*Hb;       
    HiLamnH = H'*sparse(1:Tt,1:Tt,[1/(Vpistar*lamn(1)); 1./lamn(2:end)])*H;
    ztilde = Hpsi\(zt-shortd(:,1));
    tmpXpi = Hpsi\sparse(1:Tt,1:Tt,shortd(:,2));
    Xtilde_pi = tmpXpi.*(abs(tmpXpi)>1e-6);
    Kpistar = HbiLamvHb + Xtilde_pi'*Xtilde_pi/sigw2 + HiLamnH;
    pistarhat = Kpistar\(HbiLamvHb*(pit-alppi) + Xtilde_pi'*ztilde/sigw2); % pit*_0 = 0
    pistar = pistarhat + chol(Kpistar,'lower')'\randn(Tt,1);
    
        % sample b
    b = sample_b(b,pit,pistar,lamv,pi0,Vb,sigb2);    
    
        % sample d    
    Xd = SURform([ones(Tt,1) pistar]);     
    tmpXd = Hpsi\Xd;    
    Xtilde_d = tmpXd.*(abs(tmpXd)>1e-6);       
    HdiSigdHd = Hrhod'*sparse(1:2*Tt,1:2*Tt,[(1-rhod)./sigd2; repmat(1./sigd2,Tt-1,1)])*Hrhod;
    Kd = HdiSigdHd + Xtilde_d'*Xtilde_d/sigw2;
    deld = Hrhod\[mud; repmat((1-rhod).*mud,Tt-1,1)];
    CKd = chol(Kd,'lower');    
    dhat = CKd'\(CKd\(HdiSigdHd*deld + Xtilde_d'*(Hpsi\zt)/sigw2));
    d = dhat + CKd'\randn(2*Tt,1); 
    shortd = reshape(d,2,Tt)';
    
        % sample psi    
    fpsi = @(x) -llike_MAq(x,zt-Xd*d,sigw2) - lpsipri(x);
    [psi,flag,psihat,invDpsic] = sample_psi(psi,fpsi,isim,invDpsic,options);
    Hpsi = buildHpsi(psi,Tt);    
    
        % sample mud
    for i=1:2
        Dmud = 1/(1/Vmud(i) + ((Tt-1)*(1-rhod(i))^2 + (1-rhod(i)^2))/sigd2(i));
        mudhat = Dmud*(mud0(i)/Vmud(i) + (1-rhod(i)^2)/sigd2(i)*shortd(1,i) ...
            + (1-rhod(i))/sigd2(i)*sum(shortd(2:end,i)-rhod(i)*shortd(1:end-1,i)));
        mud(i) = mudhat + sqrt(Dmud)*randn;
    end
    
        % sample rhod
    for i=1:2
        Xrhod = shortd(1:end-1,i)-mud(i);
        yrhod = shortd(2:end,i) - mud(i);
        Drhod = 1/(1/Vrhod(i) + Xrhod'*Xrhod/sigd2(i));
        rhodhat = Drhod*(rhod0(i)/Vrhod(i) + Xrhod'*yrhod/sigd2(i));  
        rhodc = rhodhat + sqrt(Drhod)*randn;
        g = @(x) -.5*log(sigd2(i)./(1-x.^2)) ...
            -.5*(1-x.^2)/sigd2(i)*(shortd(1,i)-mud(i))^2;
        if abs(rhodc) < .98
            alpMH = exp(g(rhodc)-g(rhod(i)));
            if alpMH>rand
                rhod(i) = rhodc;                
             end
        end    
    end
    Hrhod =  speye(2*Tt) - sparse(2+1:Tt*2,1:(Tt-1)*2,repmat(rhod,Tt-1,1),2*Tt,2*Tt);
    
        % sample lamv
    Ystar = log((pit-pistar-b.*[pi0; pit(1:Tt-1)-pistar(1:Tt-1)]).^2 + .0001);
    loglamv = SVRW(Ystar,log(lamv),phiv,lamv0,Vlamv);    
    lamv = exp(loglamv);    
    
        % sample lamn
    Ystar = log(([pistar(1)/sqrt(Vpistar); pistar(2:end)-pistar(1:end-1)]).^2 + .0001);
    loglamn = SVRW(Ystar,log(lamn),phin,lamn0,Vlamn);
    lamn = exp(loglamn); 

        % sample sigb2
    e2 = (b(2:end) - b(1:end-1)).^2;
    sigb2c = sample_sig2(e2,nub0,Sb0);    
    sigbc = sqrt(sigb2c);
    sigb = sqrt(sigb2);
    alpMH = -sum(log(normcdf((1-b(1:Tt))/sigbc)-normcdf(-b(1:Tt)/sigbc))) + ...
        sum(log(normcdf((1-b(1:Tt))/sigb)-normcdf(-b(1:Tt)/sigb)));    
    if alpMH > log(rand)
        sigb2 = sigb2c;         
        sigb = sigbc;
    end
    
        % sample sigd2
    for i=1:2       
        e2 = [(shortd(1,i)-mud(i))*sqrt(1-rhod(i)^2); ...
            shortd(2:end,i)-rhod(i)*shortd(1:end-1,i)-mud(i)*(1-rhod(i))].^2;        
        sigd2(i) = sample_sig2(e2,nud0(i),Sd0(i));
    end   
    
        % sample sigw2
    e2 = (Hpsi\(zt - Xd*d)).^2;
    sigw2 = sample_sig2(e2,nuw0,Sw0);    
    
        % sample phiv
    e2 = (loglamv(2:end) - loglamv(1:end-1)).^2;
    phiv = sample_sig2(e2,nuv0,Sv0);   
    
        % sample phin
    e2 = (loglamn(2:end) - loglamn(1:end-1)).^2;
    phin = sample_sig2(e2,nun0,Sn0);  

    if isim>burnin 
            % compute various forecasts
        i = isim-burnin;
        
        compute_forecasts_M1;

            % store forecasts
        tmpyhat1(i,:) = [Epi(1) normpdf(pi(t+1),Epi(1),sqrt(Vpi(1)))]; 
        if t<=T-40
            tmp1 = mean(Epi(8:12));
            tmp2 = 1/25*sum(Vpi(8:12)); 
            tmpyhat40(i,:) = [tmp1 normpdf(mean(pi(t+21:t+40)),tmp1,sqrt(tmp2))];
        end        
        if t<=T-20
            tmpyhat20(i,:) = [Epi(7) normpdf(mean(pi(t+17:t+20)),Epi(7),sqrt(Vpi(7)))];
        end
        if t<=T-16
            tmpyhat16(i,:) = [Epi(6) normpdf(mean(pi(t+13:t+16)),Epi(6),sqrt(Vpi(6)))];
        end
        if t<=T-12
            tmpyhat12(i,:) = [Epi(5) normpdf(mean(pi(t+9:t+12)),Epi(5),sqrt(Vpi(5)))];
        end
        if t<=T-8
            tmpyhat8(i,:) = [Epi(4) normpdf(mean(pi(t+5:t+8)),Epi(4),sqrt(Vpi(4)))]; 
        end
        if t<=T-4
            tmpyhat4(i,:) = [Epi(3) normpdf(mean(pi(t+1:t+4)),Epi(3),sqrt(Vpi(3)))]; 
        end
        if t<=T-2
            tmpyhat2(i,:) = [Epi(2) normpdf(mean(pi(t+1:t+2)),Epi(2),sqrt(Vpi(2)))];             
        end         
    end    
end    