import numpy as np
import random
import scipy.io as sio

def generate_batch_data_specializedB(h_act,Xp_r,Xp_i,noise_std_dl,K,\
                                     A1,A2,A3,A4,b1,b2,b3,b4,Qmin,Qmax):
    Xp = Xp_r + 1j*Xp_i
    h_actT = np.transpose(h_act,axes=[0,2,1])
    XpT = np.transpose(Xp,axes=[1,0])
    y_nless = np.transpose(h_actT@XpT,[0,2,1])
    noiseR = np.random.normal(loc=0.0, scale=noise_std_dl, size=y_nless.shape)
    noiseI = np.random.normal(loc=0.0, scale=noise_std_dl, size=y_nless.shape)
    y_noisy = y_nless + (noiseR + 1j*noiseI)
    InfoBits_soft = {0:0}
    for kk in range(K):
        y_noisy_kk = np.float32(np.concatenate([np.real(y_noisy[:,:,kk]),np.imag(y_noisy[:,:,kk])],axis=1))
        dense_layer1 = ReLU((y_noisy_kk@A1)+b1)
        dense_layer2 = ReLU((dense_layer1@A2)+b2)
        dense_layer3 = ReLU((dense_layer2@A3)+b3)
        InfoBits_soft[kk] = np.tanh((dense_layer3@A4)+b4)
        
        if kk == 0:
            DNN_input_BS = InfoBits_soft[kk]
        else:
            DNN_input_BS = np.concatenate([DNN_input_BS,InfoBits_soft[kk]],axis=1)
            
    size_per_Q = int(len(DNN_input_BS)/(Qmax-Qmin+1))
    for q in range(Qmin,Qmax+1):
        AA = sio.loadmat('Quantize_Q{}.mat'.format(q))
        CODEBOOK = AA['CODEBOOK']
        CODEBOOK = CODEBOOK[0]       
        PARTITION = AA['PARTITION']
        PARTITION = PARTITION[0]
        if q==Qmax:
            IDX = np.digitize(DNN_input_BS[(q-Qmin)*size_per_Q:,:],PARTITION)            
        else:
            IDX = np.digitize(DNN_input_BS[(q-Qmin)*size_per_Q:q*size_per_Q,:],PARTITION)
        if q ==Qmin:            
            DNN_input_BS_Q = CODEBOOK[IDX]
        else:
            DNN_input_BS_Q = np.concatenate([DNN_input_BS_Q,CODEBOOK[IDX]],axis=0)       
            
    return DNN_input_BS_Q            
        
######################################################################
def ReLU(x):
    return x * (x > 0)        