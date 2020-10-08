import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.linalg as sci
import matplotlib.pyplot as plt
from keras.layers import Dense, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
import matplotlib.cm as cm
from generate_batch_data import generate_batch_data
from generate_MRT_ZF_performance import generate_MRT_ZF_performance
from generate_batch_data_specializedB import generate_batch_data_specializedB
from mult_mod import mult_mod_complex
from tensorflow.python.keras.layers import *
'System Parameters'
M = 64 #Number of BS antennas
P = 1 #Power
K =  2 #Number of users
L = 8 #Number of pilots
Lp = 2 #Number of paths
S = 8 #Number of real parameters that we quantize (B=QS)
Qmin = 6 #Minimum Q #To get the final result for each Q set the Qmmin and Qmax equal to Q
Qmax = 6 #Maximum Q
#########################
LSF_UE = np.array(0.0*np.ones(K),dtype=np.float32) #Mean of path gains for K users
Mainlobe_UE= np.array(0.0*np.ones(K),dtype=np.float32) #Center of the AoD range for K users
HalfBW_UE = np.array(30*np.ones(K),dtype=np.float32) #Half of the AoD range for K users

snr_dl = 10 #SNR in dB
snr_max_train = 10 #max training SNR in dB
snr_min_train = 10 #min training SNR in dB
noise_std_dl = np.float32(np.sqrt(1/2)*np.sqrt(P/10**(snr_dl/10))) #STD of the Gaussian noise (per real dim.)
#####################################################
'Learning Parameters'
initial_run = 0
n_epochs = 0
learning_rate = 0.0001
batch_size = 1024
test_size = 10003
batch_per_epoch = 20 #Numbers of mini-batches per epoch
######################################################
he_init = tf.variance_scaling_initializer()
tf.reset_default_graph()        
######################################################
#Place holders
hR = tf.placeholder(tf.float32, shape=(None,M,K), name="hR")
hI = tf.placeholder(tf.float32, shape=(None,M,K), name="hI")
DNN_input_BS = tf.placeholder(tf.float32, shape=(None,K*S), name="DNN_input_BS")
###################### Weights/Biases for ENCs
BB = sio.loadmat('Data_K1_weight_bias.mat')
A1 = BB['A1']
A2 = BB['A2']
A3 = BB['A3']
A4 = BB['A4']
b1 = BB['b1']
b2 = BB['b2']
b3 = BB['b3']
b4 = BB['b4']
Xp_r = BB['Xp_r']
Xp_i = BB['Xp_i']
###################### NETWORK
# For the simplification of implementation based on Keras, we use a lambda layer to compute the rate
# Thus, the output of the model is actually the loss.
def Rate_func(temp):
    hr, hi, vr, vi, noise_power, K, M, k_index = temp
    vr = tf.transpose(tf.reshape(vr,[-1,K,M]),perm=[0, 2, 1])
    vi = tf.transpose(tf.reshape(vi,[-1,K,M]),perm=[0, 2, 1])
    for kk in range(K):
        hrvr = tf.keras.backend.batch_dot(hr, tf.transpose(a=vr[:,:,kk], perm=[1, 0]))  #Computes x.dot(y.T)
        hivi = tf.keras.backend.batch_dot(hi, tf.transpose(a=vi[:,:,kk], perm=[1, 0]))  #Computes x.dot(y.T)
        hrvi = tf.keras.backend.batch_dot(hr, tf.transpose(a=vi[:,:,kk], perm=[1, 0]))  #Computes x.dot(y.T)
        hivr = tf.keras.backend.batch_dot(hi, tf.transpose(a=vr[:,:,kk], perm=[1, 0]))  #Computes x.dot(y.T)
        real_part = hrvr - hivi
        imag_part = hrvi + hivr
        norm2_hv = tf.pow(tf.abs(real_part), 2) + tf.pow(tf.abs(imag_part), 2)
        if kk == k_index:
            nom = norm2_hv
        if kk == 0:
            nom_denom = norm2_hv + noise_power
        else:
            nom_denom = nom_denom + norm2_hv
    denom = nom_denom - nom 
    rate = tf.math.log(1 + tf.divide(nom,denom)) / tf.math.log(2.0)
    return -rate

with tf.name_scope("DL_training_phase"):
    lay = {}
    lay['noise_std'] = tf.constant(noise_std_dl)
    ############### User's operations 
    ##              --> The encoders' operations are conducated outside of the Tensorflow  
    ##              --> The outputs of the ENCs are stacked in DNN_input_BS
    power_X = (tf.reduce_sum(tf.pow(tf.abs(Xp_r), 2) + tf.pow(tf.abs(Xp_i), 2),axis=1,keepdims=True)) #just to check!
 
#######################################################################################
with tf.name_scope("BS_operations"):
    dense_layer4 = Dense(units=1024, activation='relu')(DNN_input_BS)
    dense_layer4 = BatchNormalization()(dense_layer4)
    dense_layer5 = Dense(units=512, activation='relu')(dense_layer4)
    dense_layer5 = BatchNormalization()(dense_layer5)
    dense_layer6 = Dense(units=512, activation='relu')(dense_layer5)
    dense_layer6 = BatchNormalization()(dense_layer6)
        
    V_r = Dense(units=M*K, activation='linear')(dense_layer6)
    V_i = Dense(units=M*K, activation='linear')(dense_layer6)
    norm_V = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(V_r), 2) + tf.pow(tf.abs(V_i), 2),axis=1,keepdims=True))
    V_r = np.sqrt(P)*tf.divide(V_r,norm_V)
    V_i = np.sqrt(P)*tf.divide(V_i,norm_V)                
    power_V = tf.reduce_sum(tf.pow(tf.abs(V_r), 2) + tf.pow(tf.abs(V_i), 2),axis=1,keepdims=True) #Just to check!

    rate = {0:0}
    for kk in range(K):
        rate[kk] = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))([hR[:,:,kk],hI[:,:,kk],V_r,V_i,2*lay['noise_std']**2,K,M,kk])
           
#####################################################################################
######## Loss Function
for kk in range(K):
    if kk==0:
        rate_total = rate[kk]
    else:
        rate_total = rate_total + rate[kk]
loss = tf.reduce_mean(rate_total)
######### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
######################################################################
###########  Validation Set
h_act_test, hR_act_test, hI_act_test = generate_batch_data(test_size,M,K,Lp,LSF_UE,Mainlobe_UE,HalfBW_UE)
DNN_input_BS_Q_test = generate_batch_data_specializedB(h_act_test,Xp_r,Xp_i,noise_std_dl,K,\
                                     A1,A2,A3,A4,b1,b2,b3,b4,Qmin,Qmax)
feed_dict_test = {DNN_input_BS: DNN_input_BS_Q_test,
                  hR: hR_act_test,
                  hI: hI_act_test,
                  lay['noise_std']: noise_std_dl}
rate_MRT, rate_ZF = generate_MRT_ZF_performance(h_act_test,noise_std_dl,P)
########### Load Final Data Set
AA = sio.loadmat('Data_test_K{}M{}Lp{}_{}HBW_withParams.mat'.format(K,M,Lp,int(HalfBW_UE[0])))
h_act_test_Final = AA['h_act_test_Final']
hR_act_test_Final = AA['hR_act_test_Final']
hI_act_test_Final = AA['hI_act_test_Final']
DNN_input_BS_Q_test_Final = generate_batch_data_specializedB(h_act_test_Final,Xp_r,Xp_i,noise_std_dl,K,\
                                     A1,A2,A3,A4,b1,b2,b3,b4,Qmin,Qmax)
feed_dict_test_Final = {DNN_input_BS: DNN_input_BS_Q_test_Final,
                        hR: hR_act_test_Final,
                        hI: hI_act_test_Final,
                        lay['noise_std']: noise_std_dl}
#############  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './params')
    best_loss = sess.run(loss, feed_dict=feed_dict_test)
    print(-best_loss)
    print(tf.test.is_gpu_available())

    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            snr_temp = np.random.uniform(low=snr_min_train, high=snr_max_train, size=[1])
            noise_std_temp = np.float32(np.sqrt(1/2)*np.sqrt(P/10**(snr_temp/10)))
            h_act_batch, hR_act_batch, hI_act_batch =\
                                    generate_batch_data(batch_size,M,K,Lp,LSF_UE,Mainlobe_UE,HalfBW_UE)
            DNN_input_BS_Q_batch = generate_batch_data_specializedB(h_act_batch,Xp_r,Xp_i,noise_std_dl,K,\
                                     A1,A2,A3,A4,b1,b2,b3,b4,Qmin,Qmax)                        
            feed_dict_batch = {DNN_input_BS: DNN_input_BS_Q_batch,
                               hR: hR_act_batch,
                               hI: hI_act_batch,
                               lay['noise_std']: noise_std_temp[0]}                        
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1
            
        if epoch%10==0:
            loss_test = sess.run(loss, feed_dict=feed_dict_test)
            if loss_test < best_loss:
                save_path = saver.save(sess, './params')
                best_loss = loss_test
            print('epoch:',epoch)
            print('         loss_test:%2.5f'%-loss_test,'  best_test:%2.5f'%-best_loss,'  Percntage ZF:%1.3f'%(-best_loss/rate_ZF),\
                          ' Percentage::%1.3f'%(-best_loss/rate_MRT))
    if Qmin==Qmax:        
        loss_test_Final = sess.run(loss, feed_dict=feed_dict_test_Final)
        rate_MRT, rate_ZF = generate_MRT_ZF_performance(h_act_test_Final,noise_std_dl,P)
        print(-loss_test_Final/rate_ZF)    
        sio.savemat('Data_K{}M{}Lp{}_resultBL{}_K1_B1to7_Q{}.mat'.format(K,M,Lp,L,Qmin),dict(loss_test_Final=loss_test_Final,\
                                                rate_MRT=rate_MRT, rate_ZF=rate_ZF,L=L))        