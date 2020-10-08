clc;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DNN-based w/ pilot design
B2 = [1,3,5,8,10,20,30,40,50,60];
L = 64;
f_proposed = zeros(length(B2),1);
for cnt = 1:length(B2)
    str = sprintf('Data_K2M64Lp2L%d_resultB%d.mat',L,B2(cnt));
    load(str);
    f_proposed(cnt) = eval(sprintf('-loss_test_Final_B%d', B2(cnt)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Pefect-CSI ZF/MRT
load('Data_baselines_CS_L64_Quant');
f_ZF = mean(rate_ZF);
f_MRT = mean(rate_MRT); %%%% up and MRT are the same thing!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Pefect-CSI ZF/MRT
B3 = B_vec;
f_ZF_Q = mean(rate_ZF_Q,1);
f_MRT_Q = mean(rate_MRT_Q,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  OMP-CS ZF/MRT
f_ZF_CS = mean(rate_ZF_CS);
f_MRT_CS = mean(rate_MRT_CS);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  OMP-CS ZF/MRT  Quantized
f_ZF_CSQ = mean(rate_ZF_CSQ,1);
f_MRT_CSQ = mean(rate_MRT_CSQ,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B5 = [1,10,20,30,40,50,60];
for cnt = 1:length(B5)
    str = sprintf('hhat_data_B%d_L%d',B5(cnt),L);
    load(str);
    hhat0 = hhat0(:,1:64)+1j*hhat0(:,65:end);
    hhat1 = hhat1(:,1:64)+1j*hhat1(:,65:end);
    hT_0 = squeeze(h_act_test_Final0);
    hT_1 = squeeze(h_act_test_Final1);
    rate_ZF_hat = zeros(10000,1);
    rate_MRT_hat = zeros(10000,1);
    snr_dB = 10; 
    snr = 10^(snr_dB/10); %% ==  P/(2*sigma^2)
    P = 1;
    sigma = sqrt(P/(2*snr));
    for i=1:10000
       h0 = (hhat0(i,:));
       h1 = (hhat1(i,:));
       H=[h0;h1];
       h0T = (hT_0(i,:));
       h1T = (hT_1(i,:));
       HT = [h0T;h1T];
       %%%%%%%%%%%%%%%%%%%%%% Zero_forcing Full CSI
       V = H'*pinv(H*H'); 
       pow_V = trace(V'*V);
       V = sqrt(P/pow_V)*V;
       rate_ZF_hat(i) = sum(func_rate(HT,V,sigma,K));
       %%%%%%%%%%%%%%%%%%%%% MRT Full CSI
       V = H';
       pow_V = trace(V'*V);  
       V = sqrt(P/pow_V)*V;
       rate_MRT_hat(i) = sum(func_rate(HT,V,sigma,K));
    end
    rate_ZF_hat_vec(cnt) = mean(rate_ZF_hat);
    rate_MRT_hat_vec(cnt) = mean(rate_MRT_hat);
end

color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];

figure('Renderer', 'painters', 'Position', [360 150 620 485])
set(0,'defaulttextInterpreter','latex');
B4 = union(B2,B3,'sorted');

plot(B2,f_proposed,'-sk','linewidth',2,'markersize',8);
hold on;
plot(B5,rate_MRT_hat_vec,'--hk','MarkerSize',12,'linewidth',2);
hold on;
plot(B5,rate_ZF_hat_vec,'--*k','MarkerSize',12,'linewidth',2);
hold on
plot(B4,ones(size(B4))*f_MRT,'-','color',color4,'linewidth',3,'markersize',8);
hold on;
plot(B3,f_MRT_Q,'-o','color',color4,'linewidth',2,'markersize',8);
hold on;
plot(B4,ones(size(B4))*f_MRT_CS,'-.','color',color3,'linewidth',3,'markersize',8);
hold on;
plot(B3,f_MRT_CSQ,'+','color',color3,'linewidth',2,'markersize',10);
hold on;
plot(B4,ones(size(B4))*f_ZF,'-','color',color1,'linewidth',3,'markersize',8);
hold on;
plot(B3,f_ZF_Q,'-o','color',color1,'linewidth',2,'markersize',8);
hold on;
plot(B4,ones(size(B4))*f_ZF_CS,'--','color',color5,'linewidth',3,'markersize',8);
hold on;
plot(B3,f_ZF_CSQ,'<','color',color5,'linewidth',2,'markersize',8);
hold on;

grid;
ylim([1,16]);
xlabel('Feedback Capacity $B$ (bits/coherence block)');
ylabel('Sum Rate (bits/s/Hz)');
fs2 = 8;
lg = legend({'Proposed DNN','MRT w/ DNN-CE','ZF w/ DNN-CE',...
    'MRT w/ Full CSIT', 'MRT w/ Full CSIR & Limited Feedback',...
    'MRT w/ OMP-CE & \infty Feedback', 'MRT w/ OMP-CE & Limited Feedback',...
    'ZF w/ Full CSIT', 'ZF w/ Full CSIR & Limited Feedback',...
    'ZF w/ OMP-CE & \infty Feedback', 'ZF w/ OMP-CE & Limited Feedback'},'Location','southwest');
set(lg,'Fontsize',fs2);
set(lg,'Location','southeast');