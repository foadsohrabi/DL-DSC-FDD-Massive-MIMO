clc;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DNN-based w/ pilot design
Kvec = [1,2,3,4,5,6];
f = zeros(size(Kvec));
f_MRT = zeros(size(Kvec));
f_ZF = zeros(size(Kvec));
f_K1 = zeros(size(Kvec));
for cnt = 1:6
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DNN-based w/ joint design
    str = sprintf('Data_K%dM64Lp2_resultB30L8',Kvec(cnt));    
    load(str);
    f(cnt) = -loss_test_Final;
    f_MRT(cnt) = rate_MRT;
    f_ZF(cnt) = rate_ZF;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Pefect-CSI ZF/MRT
    str = sprintf('Data_baselines_CS_K%d_Quant',Kvec(cnt));
    load(str);
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
    if cnt == 1
        f_K1(cnt) = f(cnt);
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DNN-based w/ common Enc
        str = sprintf('Data_K%dM64Lp2_resultB30L8_K1',Kvec(cnt));
        load(str);
        f_K1(cnt) = -loss_test_Final;     
    end
end

color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];

figure('Renderer', 'painters', 'Position', [360 150 620 485])
set(0,'defaulttextInterpreter','latex');


plot(Kvec,f,'-sk','linewidth',2,'markersize',8);
hold on;
a = plot(Kvec,f_K1,'--mo',...
    'LineWidth',1,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[.49 1 .63],...
    'MarkerSize',8);
hold on;
plot(Kvec,f_MRT,'-','color',color4,'linewidth',2,'markersize',8);
hold on;
plot(Kvec,f_MRT_Q,'-o','color',color4,'linewidth',2,'markersize',8);
hold on;
plot(Kvec,f_MRT_CS,'-.','color',color3,'linewidth',2,'markersize',8);
hold on;
plot(Kvec,f_MRT_CSQ,'--+','color',color3,'linewidth',2,'markersize',10);
hold on;
plot(Kvec,f_ZF,'-','color',color1,'linewidth',2,'markersize',8);
hold on
plot(Kvec,f_ZF_Q,'-o','color',color1,'linewidth',2,'markersize',8);
hold on;
plot(Kvec,f_ZF_CS,'--','color',color5,'linewidth',2,'markersize',8);
hold on;
plot(Kvec,f_ZF_CSQ,'--<','color',color5,'linewidth',2,'markersize',8);


grid;
xlim([1,6]);
ylim([4,34]);
xlabel('Number of Users (K)');
ylabel('Sum Rate (bits/s/Hz)');
fs2 = 8;
lg = legend({'Proposed DNN','Porposed DNN w/ K-modified training',...
    'MRT w/ Full CSIT', 'MRT w/ Full CSIR & Limited Feedback',...
    'MRT w/ OMP-CE & \infty Feedback', 'MRT w/ OMP-CE & Limited Feedback',...
    'ZF w/ Full CSIT', 'ZF w/ Full CSIR & Limited Feedback',...
    'ZF w/ OMP-CE & \infty Feedback', 'ZF w/ OMP-CE & Limited Feedback'},'Location','northwest');
set(lg,'Fontsize',fs2);
set(lg,'Location','northwest');
xticks([1,2,3,4,5,6])

