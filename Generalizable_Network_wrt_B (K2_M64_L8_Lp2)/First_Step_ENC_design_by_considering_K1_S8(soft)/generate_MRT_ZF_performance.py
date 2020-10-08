import numpy as np

def generate_MRT_ZF_performance(h,noise_std_dl,P):
    noise_power = 2*(noise_std_dl**2)
    ch_size,M,K = h.shape
    rate_MRT = []
    rate_ZF = []
    for ch in range(ch_size):        
        H = np.transpose(np.squeeze(h[ch,:,:]))
        if K==1:
            H = np.reshape(H,[K,M])
        HH = H.conj().transpose()
        ###################### MRT
        V_MRT = HH
        norm_V_MRT = np.abs(np.trace(V_MRT@V_MRT.conj().transpose()))
        V_MRT = np.sqrt(P/norm_V_MRT)*V_MRT
        rate_MRT += [func_help_rate(V_MRT,H,K,noise_power)]        
        ###################### ZF
        V_ZF = HH@np.linalg.inv(H@HH)
        norm_V_ZF = np.abs(np.trace(V_ZF@V_ZF.conj().transpose()))
        V_ZF = np.sqrt(P/norm_V_ZF)*V_ZF
        rate_ZF += [func_help_rate(V_ZF,H,K,noise_power)]
        
    return np.mean(rate_MRT), np.mean(rate_ZF)    
        
        
def func_help_rate(V,H,K,noise_power):
    hv = H@V
    norm2_hv = np.abs(hv)**2
    rate = []
    for k_index in range(K):
        for kk in range(K):
            if kk == k_index:
                nom = norm2_hv[k_index,kk]
            if kk == 0:
                nom_denom = norm2_hv[k_index,kk] + noise_power
            else:
                nom_denom = nom_denom + norm2_hv[k_index,kk]
        denom = nom_denom - nom 
        rate += [np.log2(1 + nom/denom)] 
    return sum(rate)    
        
        
        
        
        
    