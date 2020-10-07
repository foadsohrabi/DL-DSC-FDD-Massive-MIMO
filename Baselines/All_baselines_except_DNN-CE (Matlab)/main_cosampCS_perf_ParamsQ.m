clc;
close all;
clear all;
L = 8;
if L == 8
    load('Data_test_K2M64Lp2L8_withParams.mat');
elseif L == 64
    load('Data_test_K2M64Lp2L64_withParams.mat');
end

h_total = h_act_test_Final;
snr_dB = 10; 
snr = 10^(snr_dB/10); %% ==  P/(2*sigma^2)
P = 1;
sigma = sqrt(P/(2*snr));
K = 2;
ch_num = length(h_total);
M = 64;

if L == M
    X = dftmtx(M);
else
    X = randn(M,L)+1j*randn(M,L); %pilots
end
for ll = 1:L
    pow_X = abs(X(:,ll)'*X(:,ll));
    X(:,ll) = sqrt(P/pow_X)*X(:,ll);
end
supp_size = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine where your m-file's folder is.
folder = pwd;
folder = strcat(folder,'\Quantizer');
addpath(genpath(folder));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimal quantizers for theta dimesnsions (Uniform Dist.)
% Optimal quantizers for alpha dimesnsions (Gaussian Dist.)
FPDF_unif = PDFFn ('uniform', 0, 1);
FPDF = PDFFn ('gauss', 0, 1);
Aq = cell(10,1);
Tq = cell(10,1);
Q_vec = 1:10;
B_vec = 6*Q_vec;
B_num = length(B_vec);
min_theta = -30;
max_theta = +30;
for i = 1:B_num
    Tq{i} = (max_theta/sqrt(3))*QuantUnif (2^i, FPDF_unif, 0);
    Aq{i} = sqrt(1/2)*QuantLloyd (2^i, FPDF, 0);
end
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dict_size = 1000;
theta_vec = linspace(min_theta,max_theta,Dict_size);
theta_vec_rad = (pi/180)*theta_vec;
theta_vec_sin = sin(theta_vec_rad);
S = zeros(Dict_size,M);
for s = 1:Dict_size
   S(s,:) = exp(1j*pi*(0:M-1)*theta_vec_sin(s)); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rate_ZF = zeros(ch_num,1);
rate_MRT = zeros(ch_num,1);
rate_ZF_CS = zeros(ch_num,1);
rate_MRT_CS = zeros(ch_num,1);
rate_ZF_Q = zeros(ch_num,B_num);
rate_MRT_Q = zeros(ch_num,B_num);
rate_ZF_CSQ = zeros(ch_num,B_num);
rate_MRT_CSQ = zeros(ch_num,B_num);
for i = 1:ch_num
    disp(i);
    H = transpose(squeeze(h_total(i,:,:)));
   %%%%%%%%%%%%%%%%%%%%%% CoSamp
   noise = sigma*(randn(K,L)+1j*randn(K,L));
   Y = H*X + noise;
   Hhat = zeros(K,M);
   HhatQvec = zeros(K,M,B_num);
   for k=1:K
       u = transpose(Y(k,:));
       Phi = transpose(X)*transpose(S);
       idx = zeros(supp_size,1);
       v = u;
       for tt = 1:supp_size
            [~,idx(tt)] = max(abs(Phi'*v));
            phi_s = Phi(:,idx(1:tt));
            b = pinv(phi_s)*u;
            v = v - phi_s*b;
       end
       hhat = transpose(b)*S(idx,:);
       Hhat(k,:) = hhat;
       
       b_temp = [real(b);imag(b)];
       theta_temp = theta_vec(idx);
       for q=1:B_num
            bQ = func_quant(b_temp,Aq{q});
            thetaQ = func_quant(theta_temp,Tq{q});
            thetaQ_rad = (pi/180)*thetaQ;
            thetaQ_sin = sin(thetaQ_rad);
            SQ = zeros(supp_size,M);
            for s = 1:supp_size
               SQ(s,:) = exp(1j*pi*(0:M-1)*thetaQ_sin(s)); 
            end
            hhatQ = transpose(bQ(1:supp_size)+1j*bQ(supp_size+1:end))*SQ;
            HhatQvec(k,:,q) = hhatQ;            
       end
   end
   %%%%%%%%%%%%%%%%%%%%%% Zero_forcing Full CSI
   V = H'/(H*H'); 
   pow_V = trace(V'*V);
   V = sqrt(P/pow_V)*V;
   rate_ZF(i) = sum(func_rate(H,V,sigma,K));
   %%%%%%%%%%%%%%%%%%%%% MRT Full CSI
   V = H';
   pow_V = trace(V'*V);  
   V = sqrt(P/pow_V)*V;
   rate_MRT(i) = sum(func_rate(H,V,sigma,K));
   %%%%%%%%%%%%%%%%%%%%%% Full CSIR, limited feedback
   HhatQvec_FullCSI = zeros(K,M,B_num);
   for k = 1:K
       theta_k = transpose(theta_act_test_Final(i,:,k));
       alpha_k = transpose(alpha_act_test_Final(i,:,k));
       b_temp = [real(alpha_k);imag(alpha_k)];
       theta_temp = theta_k;
       for q=1:B_num
            bQ = func_quant(b_temp,Aq{q});
            thetaQ = func_quant(theta_temp,Tq{q});
            thetaQ_rad = (pi/180)*thetaQ;
            thetaQ_sin = sin(thetaQ_rad);
            SQ = zeros(supp_size,M);
            for s = 1:supp_size
               SQ(s,:) = exp(1j*pi*(0:M-1)*thetaQ_sin(s)); 
            end
            hhatQ = transpose(bQ(1:supp_size)+1j*bQ(supp_size+1:end))*SQ;
            HhatQvec_FullCSI(k,:,q) = hhatQ;            
       end
   end
   for q = 1:B_num
       HhatQ_CSIR = squeeze(HhatQvec_FullCSI(:,:,q));
       %%%%%%%%%%%%%%%%%%%%%% Zero_forcing Full CSIR, limited feedback
       V = HhatQ_CSIR'*pinv(HhatQ_CSIR*HhatQ_CSIR'); 
       pow_V = trace(V'*V);
       V = sqrt(P/pow_V)*V;
       rate_ZF_Q(i,q) = sum(func_rate(H,V,sigma,K));
       %%%%%%%%%%%%%%%%%%%%% MRT Full CSI Full CSIR, limited feedback
       V = HhatQ_CSIR';
       pow_V = trace(V'*V);  
       V = sqrt(P/pow_V)*V;
       rate_MRT_Q(i,q) = sum(func_rate(H,V,sigma,K));
   end
   
   %%%%%%%%%%%%%%%%%%%%%% Zero_forcing CS
   V = Hhat'/(Hhat*Hhat'); 
   pow_V = trace(V'*V);
   V = sqrt(P/pow_V)*V;
   rate_ZF_CS(i) = sum(func_rate(H,V,sigma,K));   
   %%%%%%%%%%%%%%%%%%%%% MRT CS
   V = Hhat';
   pow_V = trace(V'*V);  
   V = sqrt(P/pow_V)*V;
   rate_MRT_CS(i) = sum(func_rate(H,V,sigma,K));  
   for q = 1:B_num
       HhatQ = squeeze(HhatQvec(:,:,q));
       %%%%%%%%%%%%%%%%%%%%%% Zero_forcing
       V = HhatQ'*pinv(HhatQ*HhatQ'); 
       pow_V = trace(V'*V);
       V = sqrt(P/pow_V)*V;
       rate_ZF_CSQ(i,q) = sum(func_rate(H,V,sigma,K));

       %%%%%%%%%%%%%%%%%%%%% MRT
       V = HhatQ';
       pow_V = trace(V'*V);  
       V = sqrt(P/pow_V)*V;
       rate_MRT_CSQ(i,q) = sum(func_rate(H,V,sigma,K));
   end
end
if L == 8
    save('Data_baselines_CS_L8_Quant',...
        'rate_ZF','rate_MRT','rate_ZF_Q','rate_MRT_Q',...
        'rate_ZF_CSQ','rate_ZF_CS','rate_MRT_CSQ','rate_MRT_CS','snr_dB','P','K','X','B_vec');
elseif L == 64
    save('Data_baselines_CS_L64_Quant',...
        'rate_ZF','rate_MRT','rate_ZF_Q','rate_MRT_Q',...
        'rate_ZF_CSQ','rate_ZF_CS','rate_MRT_CSQ','rate_MRT_CS','snr_dB','P','K','X','B_vec');
end
%%% Uncomment if you want to visualize the results!
% plot(B_vec,ones(size(B_vec))*mean(rate_ZF),'-*k','linewidth',2,'markersize',8);
% hold on;
% plot(B_vec,ones(size(B_vec))*mean(rate_MRT),'-sk','linewidth',2,'markersize',8);
% hold on;
% plot(B_vec,mean(rate_ZF_Q,1),'--*k','linewidth',2,'markersize',8);
% hold on;
% plot(B_vec,mean(rate_MRT_Q,1),'--sk','linewidth',2,'markersize',8);
% hold on;
% plot(B_vec,ones(size(B_vec))*mean(rate_ZF_CS),'-*r','linewidth',2,'markersize',8);
% hold on;
% plot(B_vec,ones(size(B_vec))*mean(rate_MRT_CS),'-sg','linewidth',2,'markersize',8);
% hold on,
% plot(B_vec,mean(rate_ZF_CSQ,1),'--*r','linewidth',2,'markersize',8);
% hold on;
% plot(B_vec,mean(rate_MRT_CSQ,1),'--sg','linewidth',2,'markersize',8);