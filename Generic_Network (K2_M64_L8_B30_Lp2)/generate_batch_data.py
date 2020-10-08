import numpy as np
import random

def generate_batch_data(batch_size,M,K,L,LSF_UE,Mainlobe_UE,HalfBW_UE):
    alphaR_input = np.zeros((batch_size,L,K))
    alphaI_input = np.zeros((batch_size,L,K))
    theta_input = np.zeros((batch_size,L,K))
    for kk in range(K):
        alphaR_input[:,:,kk] = np.random.normal(loc=LSF_UE[kk], scale=1.0/np.sqrt(2), size=[batch_size,L])
        alphaI_input[:,:,kk] = np.random.normal(loc=LSF_UE[kk], scale=1.0/np.sqrt(2), size=[batch_size,L])
        theta_input[:,:,kk] = np.random.uniform(low=Mainlobe_UE[kk]-HalfBW_UE[kk], high=Mainlobe_UE[kk]+HalfBW_UE[kk], size=[batch_size,L])
 
    #### Actual Channel
    from0toM = np.float32(np.arange(0, M, 1))
    alpha_act = alphaR_input + 1j*alphaI_input
    theta_act = (np.pi/180)*theta_input
    h_act = np.complex128(np.zeros((batch_size,M,K)))
    hR_act = np.float32(np.zeros((batch_size,M,K)))
    hI_act = np.float32(np.zeros((batch_size,M,K)))
    for kk in range(K):
        for ll in range(L):
            theta_act_expanded_temp = np.tile(np.reshape(theta_act[:,ll,kk],[-1,1]),(1,M))
            response_temp = np.exp(1j*np.pi*np.multiply(np.sin(theta_act_expanded_temp),from0toM))
            alpha_temp = np.reshape(alpha_act[:,ll,kk],[-1,1])
            h_act[:,:,kk] += (1/np.sqrt(L))*alpha_temp*response_temp
        hR_act[:,:,kk] = np.real(h_act[:,:,kk])
        hI_act[:,:,kk] = np.imag(h_act[:,:,kk])
        


    return(h_act, hR_act, hI_act)

