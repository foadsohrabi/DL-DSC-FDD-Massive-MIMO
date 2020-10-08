clc;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BB = 8*(1:7);
f_robustB = zeros(length(BB),1);
for Q = 1:7
    str = sprintf('Data_K2M64Lp2_resultBL8_K1_B1to7_Q%d',Q);
    load(str);
    f_robustB(Q) = - loss_test_Final;    
end    

%  DNN-based w/ pilot design
B2 = [1,3,6,10,15,20,30,40,50,60];
cnt = 0;
load('Data_K2M64Lp2L8_resultB1.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B1;
load('Data_K2M64Lp2L8_resultB3.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B3;
load('Data_K2M64Lp2L8_resultB6.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B6;
load('Data_K2M64Lp2L8_resultB10.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B10;
load('Data_K2M64Lp2L8_resultB15.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B15;
load('Data_K2M64Lp2L8_resultB20.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B20;
load('Data_K2M64Lp2L8_resultB30.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B30;
load('Data_K2M64Lp2L8_resultB40.mat');
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B40;
load('Data_K2M64Lp2L8_resultB50.mat')
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B50;
load('Data_K2M64Lp2L8_resultB60.mat')
cnt = cnt +1;
f_proposed(cnt) = -loss_test_Final_B60;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Pefect-CSI ZF/MRT
load('Data_baselines_CS_L8_Quant');
f_ZF = mean(rate_ZF);
f_MRT = mean(rate_MRT); %%%% up and MRT are the same thing!


color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];

figure('Renderer', 'painters', 'Position', [360 150 620 485])
set(0,'defaulttextInterpreter','latex');
B3 = B_vec;
B4 = union(B2,B3,'sorted');

plot(BB,f_robustB,'--sb','linewidth',2,'markersize',8);
hold on;
plot(B2(3:end),f_proposed(3:end),'-sk','linewidth',2,'markersize',8);
hold on;
plot(B4(3:end),ones(size(B4(3:end)))*f_MRT,'-','color',color4,'linewidth',3,'markersize',8);
hold on;
plot(B4(3:end),ones(size(B4(3:end)))*f_ZF,'-','color',color1,'linewidth',3,'markersize',8);


grid;
xlim([6,60]);
ylim([11.5,15.5]);
xlabel('Feedback Capacity $B$ (bits/coherence block)');
ylabel('Sum Rate (bits/s/Hz)');
fs2 = 10;
lg = legend({'Proposed DNN w/ B-modifed training','Proposed DNN',...
    'MRT w/ Full CSIT',...
    'ZF w/ Full CSIT'},'Location','southeast');
set(lg,'Fontsize',fs2);
set(lg,'Location','southeast');
yticks(11.5:.5:15.5);

txt = {'$(S,Q)=(8,1)$'};
text(9,11.7,txt,'Color','blue');
txt = {'$(S,Q)=(8,2)$'};
text(16.8,13.4,txt,'Color','blue');
txt = {'$(S,Q)=(8,3)$'};
text(18,14.20,txt,'Color','blue');
txt = {'$(S,Q)=(8,4)$'};
text(27,14.37,txt,'Color','blue');
txt = {'$(S,Q)=(8,5)$'};
text(33,13.75,txt,'Color','blue');
txt = {'$(S,Q)=(8,6)$'};
text(40,14.05,txt,'Color','blue');
txt = {'$(S,Q)=(8,7)$'};
text(50,14.05,txt,'Color','blue');