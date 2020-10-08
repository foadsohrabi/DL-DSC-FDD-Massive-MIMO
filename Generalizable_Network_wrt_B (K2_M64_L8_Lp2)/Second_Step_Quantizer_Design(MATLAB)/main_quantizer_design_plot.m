clc;
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine where your m-file's folder is.
folder = pwd;
folder = strcat(folder,'\Quantizer');
addpath(genpath(folder));

S = 8;
str = sprintf('softBits_S%d_K1',S);
load(str);

Q = 3;
ini_codebook = linspace(-1,1,2^Q);
[PARTITION, CODEBOOK] = lloyds(softBits,ini_codebook);

c = jet(8);
[f,xi] = ksdensity(softBits,'Bandwidth',0.03);

figure('Renderer', 'painters', 'Position', [360 150 620 485])
set(0,'defaulttextInterpreter','latex');

H=area(xi,f);
hold on;
idx=xi<=PARTITION(1);
H1=area(xi(idx),f(idx));
set(H1(1),'FaceColor',c(1,:));
hold on;
idx=xi>=PARTITION(1)&xi<PARTITION(2);
H2=area(xi(idx),f(idx));
set(H2(1),'FaceColor',c(2,:));
hold on;
idx=xi>=PARTITION(2)&xi<PARTITION(3);
H3=area(xi(idx),f(idx));
set(H3(1),'FaceColor',c(3,:));
hold on;
idx=xi>=PARTITION(3)&xi<PARTITION(4);
H4=area(xi(idx),f(idx));
set(H4(1),'FaceColor',c(4,:));
hold on;
idx=xi>=PARTITION(4)&xi<PARTITION(5);
H5=area(xi(idx),f(idx));
set(H5(1),'FaceColor',c(5,:));
hold on;
idx=xi>=PARTITION(5)&xi<PARTITION(6);
H6=area(xi(idx),f(idx));
set(H6(1),'FaceColor',c(6,:));
hold on;
idx=xi>=PARTITION(6)&xi<PARTITION(7);
H7=area(xi(idx),f(idx));
set(H7(1),'FaceColor',c(7,:));
hold on;
idx=xi>=PARTITION(7);
H8=area(xi(idx),f(idx));
set(H8(1),'FaceColor',c(8,:));
grid;
xlim([-1,1]);
hold on;


color1=[0,0,0];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];

stem(CODEBOOK,0.15*ones(size(CODEBOOK)),'filled','LineWidth',2,'Color',color1,'MarkerFaceColor',color1,...
     'MarkerEdgeColor',color1);

x = linspace(-1,1,500);
 
mu = mean(softBits);
s = sqrt(sum((softBits-mu).^2)/length(softBits));
p1 = -.5 * ((x - mu)/s) .^ 2;
p2 = (s * sqrt(2*pi));
f = exp(p1) ./ p2; 

hold on;
plot(x,f,'-r','LineWidth',2);
xlabel('Output of tanh layers');
ylabel('PDF');
fs2 = 8;
lg = legend({'Decision Thresholds','Quantization Region 1',...
            'Quantization Region 2','Quantization Region 3',...
            'Quantization Region 4','Quantization Region 5',...
            'Quantization Region 6','Quantization Region 7',... 
            'Quantization Region 8','Representation Points','Gaussian Approximation'},'Location','northeast');
set(lg,'Fontsize',fs2);
set(lg,'Location','northeast');



