clc;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DNN-based w/ pilot design
Lpvec = 2:8;
f = zeros(size(Lpvec));
f_robust = zeros(size(Lpvec));
f_MRT = zeros(size(Lpvec));
f_ZF = zeros(size(Lpvec));
f_ZF_Q = zeros(size(Lpvec));
f_MRT_Q = zeros(size(Lpvec));
f_ZF_CS = zeros(size(Lpvec));
f_MRT_CS = zeros(size(Lpvec));
f_ZF_CSQ = zeros(size(Lpvec));
f_MRT_CSQ = zeros(size(Lpvec));
for cnt = 1:length(Lpvec)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DNN-based w/ joint design
    str = sprintf('Data_K2M64B30L8_resultLp%d',Lpvec(cnt));    
    load(str);
    f(cnt) = -loss_test_Final_B30;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DNN-based w/ joint design
    str = sprintf('Data_K2M64B30L8_resultLp%d_rangeTrained',Lpvec(cnt));    
    load(str);
    f_robust(cnt) = -loss_test_Final;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Pefect-CSI ZF/MRT
    str = sprintf('Data_baselines_CS_Lp%d_Quant',Lpvec(cnt));
    load(str);
    f_MRT(cnt) = mean(rate_MRT);
    f_ZF(cnt) = mean(rate_ZF);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Pefect-CSI ZF/MRT
    f_ZF_Q(cnt) = mean(rate_ZF_Q,1);
    f_MRT_Q(cnt) = mean(rate_MRT_Q,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  OMP-CS ZF/MRT
    f_ZF_CS(cnt) = mean(rate_ZF_CS);
    f_MRT_CS(cnt) = mean(rate_MRT_CS);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  OMP-CS ZF/MRT  Quantized
    f_ZF_CSQ(cnt) = mean(rate_ZF_CSQ,1);
    f_MRT_CSQ(cnt) = mean(rate_MRT_CSQ,1);   
end

color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];

figure('Renderer', 'painters', 'Position', [360 150 620 485])
set(0,'defaulttextInterpreter','latex');

plot(Lpvec,f_robust,'--sk','linewidth',2,'markersize',8);
hold on;
plot(Lpvec,f,'-sk','linewidth',2,'markersize',8);
hold on;
plot(Lpvec,f_MRT,'-','color',color4,'linewidth',3,'markersize',8);
hold on;
plot(Lpvec,f_MRT_Q,'-*','color',color4,'linewidth',2,'markersize',8);
hold on;
plot(Lpvec,f_MRT_CS,'-.','color',color3,'linewidth',3,'markersize',8);
hold on;
plot(Lpvec,f_MRT_CSQ,'--+','color',color3,'linewidth',2,'markersize',10);
hold on;
plot(Lpvec,f_ZF,'-','color',color1,'linewidth',3,'markersize',8);
hold on
plot(Lpvec,f_ZF_Q,'-o','color',color1,'linewidth',2,'markersize',8);
hold on;
plot(Lpvec,f_ZF_CS,'--','color',color5,'linewidth',3,'markersize',8);
hold on;
plot(Lpvec,f_ZF_CSQ,'--<','color',color5,'linewidth',2,'markersize',8);


grid;
xlim([2,8]);
ylim([-4,17]);
xlabel('Number of Paths ($L_p$)');
ylabel('Sum Rate (bits/s/Hz)');
fs2 = 8;
lg = legend({'Proposed DNN (trained for Lp\in[2,8])','Proposed DNN (trained for Lp=2)',...
    'MRT w/ Full CSIT', 'MRT w/ Full CSIR & Limited Feedback',...
    'MRT w/ OMP-CE & \infty Feedback', 'MRT w/ OMP-CE & Limited Feedback',...
    'ZF w/ Full CSIT', 'ZF w/ Full CSIR & Limited Feedback',...
    'ZF w/ OMP-CE & \infty Feedback', 'ZF w/ OMP-CE & Limited Feedback'},'Location','southwest');
set(lg,'Fontsize',fs2);
set(lg,'Location','southwest');
yticks(0:2:16)