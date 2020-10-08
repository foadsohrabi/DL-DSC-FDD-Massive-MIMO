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

Q = ;
PARTITION = linspace(-1,1,2^Q+1);
CODEBOOK = zeros(1,2^Q);
for i = 1:2^Q
    CODEBOOK(i) = 0.5*(PARTITION(i)+PARTITION(i+1));
end
PARTITION(end) = [];
PARTITION(1) = [];

[f,xi] = ksdensity(softBits,'Bandwidth',0.02);
plot(xi,f);
hold on;
stem(CODEBOOK,0.25*ones(size(CODEBOOK)));
hold on;
stem(PARTITION,0.3*ones(size(PARTITION)));
str2 = sprintf('Quantize_Q%d_Uniform',Q);
save(str2,'PARTITION','CODEBOOK','Q','S');



