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

[f,xi] = ksdensity(softBits,'Bandwidth',0.03);
plot(xi,f);
hold on;
stem(CODEBOOK,0.25*ones(size(CODEBOOK)));
hold on;
stem(PARTITION,0.3*ones(size(PARTITION)));
str2 = sprintf('Quantize_Q%d',Q);
save(str2,'PARTITION','CODEBOOK','Q','S');



