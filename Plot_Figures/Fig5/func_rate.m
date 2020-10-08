function rate = func_rate(H,V,sigma,K)
    rate = zeros(K,1);
    for k=1:K
        nom = abs(H(k,:)*V(:,k))^2;
        denom = 2*sigma^2 + sum(abs(H(k,:)*V).^2) - nom;
        rate(k) = log2(1+nom/denom);
    end 
end