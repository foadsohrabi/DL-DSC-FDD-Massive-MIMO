function xQ = func_quant(x,codebook)

N = length(x); %x is Nby1 vector
xQ = zeros(N,1);
for n=1:N
    x_temp = x(n);
    [~,idx] = min(abs(codebook-x_temp));
    xQ(n) = codebook(idx);
end