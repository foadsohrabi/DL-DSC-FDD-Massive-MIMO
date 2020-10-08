import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.linalg as sci
import matplotlib.pyplot as plt
from keras.layers import Dense, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
import matplotlib.cm as cm
from generate_batch_data import generate_batch_data
from mult_mod import mult_mod_complex
from tensorflow.python.keras.layers import *
'System Parameters'
M = 64 #Number of BS antennas
P = 1 #Power
K =  2 #Number of users
L = 8 #Number of pilots
Lp = 2 #Number of paths
B = 30 #Number of feedback bits per user
LSF_UE = np.array([0.0,0.0],dtype=np.float32) #Mean of path gains for K users
Mainlobe_UE= np.array([0,0],dtype=np.float32) #Center of the AoD range for K users
HalfBW_UE = np.array([30.0,30.0],dtype=np.float32) #Half of the AoD range for K users

snr_dl = 10 #SNR in dB
snr_max_train = 10 #max training SNR in dB
snr_min_train = 10 #min training SNR in dB
noise_std_dl = np.float32(np.sqrt(1/2)*np.sqrt(P/10**(snr_dl/10))) #STD of the Gaussian noise (per real dim.)
#####################################################
'Learning Parameters'
initial_run = 0 #1: starts training from scratch; 0: resumes training 
n_epochs = 0 #Number of training epochs, for observing the current performance set it to 0
learning_rate = 0.0001 #Learning rate
batch_size = 1024 #Mini-batch size
test_size = 10000 #Size of the validation/test set
batch_per_epoch = 20 #Numbers of mini-batches per epoch
anneal_param = 1.0 #Initial annealing parmeter
annealing_rate = 1.001 #Annealing rate 
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Determine the DNN initialization method
#Pilot matrix initialization (The random Gaussian pilots also work well!)
DFT_Matrix = sci.dft(M) 
X_init = DFT_Matrix[0::int(np.ceil(M/L)),:] 
Xp_init = np.sqrt(P/M)*X_init
Xp_r_init = np.float32(np.real(Xp_init))
Xp_i_init = np.float32(np.imag(Xp_init))
######################################################
#Place holders
hR = tf.placeholder(tf.float32, shape=(None,M,K), name="hR")
hI = tf.placeholder(tf.float32, shape=(None,M,K), name="hI")
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
    lay['noise_std'] = tf.constant(noise_std_dl) #STD of the Gaussian noise (per real dim.)
    lay['aneal'] = tf.constant(1.0) #Annealing parameter
    ############### User's operations
    Xp_r = tf.get_variable("Xp_r",  dtype=tf.float32, initializer= Xp_r_init)
    Xp_i = tf.get_variable("Xp_i",  dtype=tf.float32, initializer= Xp_i_init)
    #Normailzing the pilot matrix to satisfy the power constraint
    norm_X = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(Xp_r), 2) + tf.pow(tf.abs(Xp_i), 2),axis=1,keepdims=True)) 
    Xp_r = np.sqrt(P)*tf.divide(Xp_r,norm_X)
    Xp_i = np.sqrt(P)*tf.divide(Xp_i,norm_X)
    power_X = (tf.reduce_sum(tf.pow(tf.abs(Xp_r), 2) + tf.pow(tf.abs(Xp_i), 2),axis=1,keepdims=True)) #just to check!

    y_nless = {0:0}
    y_noisy = {0:0}
    for kk in range(K):
            hR_temp = tf.reshape(hR[:,:,kk],[-1,M,1])
            hI_temp = tf.reshape(hI[:,:,kk],[-1,M,1])
            y_nless_r,y_nless_i = mult_mod_complex(hR_temp,hI_temp,Xp_r,Xp_i,'l')            
            y_nless[kk] = tf.concat([tf.reshape(y_nless_r,[-1,L]),tf.reshape(y_nless_i,[-1,L])],axis=1)
            y_noisy[kk] = y_nless[kk] + tf.random_normal(shape = tf.shape(y_nless[kk]), mean = 0.0,\
                         stddev = lay['noise_std'], dtype='float32')
#######################################################################################                        
with tf.name_scope("UEs_operations"):
    InfoBits = {0:0}
    for kk in range(K):
        DNN_input = BatchNormalization()(y_noisy[kk])
        dense_layer1 = Dense(units=1024, activation='relu')(DNN_input)
        dense_layer1 = BatchNormalization()(dense_layer1)
        dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
        dense_layer2 = BatchNormalization()(dense_layer2)
        dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)
        dense_layer3 = BatchNormalization()(dense_layer3)
        InfoBits_linear = Dense(units=B, activation='linear')(dense_layer3)
        InfoBits_tanh = tf.tanh(lay['aneal']*InfoBits_linear)
        InfoBits_sign = tf.sign(InfoBits_linear)
        InfoBits[kk] = InfoBits_tanh + tf.stop_gradient(InfoBits_sign - InfoBits_tanh) #Straight through technique
        if kk == 0:
            DNN_input_BS = InfoBits[kk]
        else:
            DNN_input_BS = tf.concat([DNN_input_BS,InfoBits[kk]],axis=1)     
#######################################################################################
with tf.name_scope("BS_operations"):
    dense_layer4 = Dense(units=1024, activation='relu')(DNN_input_BS)
    dense_layer4 = BatchNormalization()(dense_layer4)
    dense_layer5 = Dense(units=512, activation='relu')(dense_layer4)
    dense_layer5 = BatchNormalization()(dense_layer5)
    dense_layer6 = Dense(units=512, activation='relu')(dense_layer5)
    dense_layer6 = BatchNormalization()(dense_layer6)
        
    V_r = Dense(units=M*K, activation='linear')(dense_layer6) #Precoder real part (not normalized yet) 
    V_i = Dense(units=M*K, activation='linear')(dense_layer6) #Precoder imag part (not normalized yet) 
    norm_V = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(V_r), 2) + tf.pow(tf.abs(V_i), 2),axis=1,keepdims=True))
    V_r = np.sqrt(P)*tf.divide(V_r,norm_V) #Precoder real part (normalized)
    V_i = np.sqrt(P)*tf.divide(V_i,norm_V) #Precoder real part (normalized)               
    power_V = tf.reduce_sum(tf.pow(tf.abs(V_r), 2) + tf.pow(tf.abs(V_i), 2),axis=1,keepdims=True) #Just to check!

    rate = {0:0}
    for kk in range(K):
        rate[kk] = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))([hR[:,:,kk],hI[:,:,kk],V_r,V_i,2*lay['noise_std']**2,K,M,kk])

    ################################# MRT-baseline
    rate_UP = {0:0}
    for kk in range(K):
        h_temp = tf.complex(hR[:,:,kk],hI[:,:,kk])
        V_MRT = tf.conj(h_temp)
        if kk == 0:
            V_MRT_r = tf.real(V_MRT)
            V_MRT_i = tf.imag(V_MRT)
        else:
            V_MRT_r = tf.concat([V_MRT_r,tf.real(V_MRT)],axis=1)
            V_MRT_i = tf.concat([V_MRT_i,tf.imag(V_MRT)],axis=1)
    norm_V_MRT = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(V_MRT_r), 2) + tf.pow(tf.abs(V_MRT_i), 2),axis=1,keepdims=True))
    V_MRT_r = np.sqrt(P)*tf.divide(V_MRT_r,norm_V_MRT)
    V_MRT_i = np.sqrt(P)*tf.divide(V_MRT_i,norm_V_MRT)                 
    for kk in range(K):        
        rate_UP[kk] = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))\
                        ([hR[:,:,kk],hI[:,:,kk],V_MRT_r,V_MRT_i,2*lay['noise_std']**2,K,M,kk])              
#####################################################################################
######## Loss Function
for kk in range(K):
    if kk==0:
        rate_total = rate[kk]
        rate_UP_total = rate_UP[kk]
    else:
        rate_total = rate_total + rate[kk]
        rate_UP_total = rate_UP_total + rate_UP[kk]
loss = tf.reduce_mean(rate_total)
loss_UP = tf.reduce_mean(rate_UP_total)
######### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
######################################################################
###########  Validation Set
h_act_test, hR_act_test, hI_act_test = generate_batch_data(test_size,M,K,Lp,LSF_UE,Mainlobe_UE,HalfBW_UE)
feed_dict_test = {hR: hR_act_test,
                  hI: hI_act_test,
                  lay['noise_std']: noise_std_dl,
                  lay['aneal']: 1.0}
########### Load Final Data Set
AA = sio.loadmat('Data_test_K2M64Lp2L8_withParams.mat')
hR_act_test_Final = AA['hR_act_test_Final']
hI_act_test_Final = AA['hI_act_test_Final']
feed_dict_test_Final = {hR: hR_act_test_Final,
                  hI: hI_act_test_Final,
                  lay['noise_std']: noise_std_dl,
                  lay['aneal']: 1.0}
############  Training
with tf.Session() as sess:
    if initial_run == 1: #Training from scratch 
        init.run()
    else:  #Resume training
        saver.restore(sess, './params')
    best_loss, loss_UP_test = sess.run([loss,loss_UP], feed_dict=feed_dict_test)
    print(-best_loss)
    print(tf.test.is_gpu_available()) #Check whether or not the GPU is available

    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            snr_temp = np.random.uniform(low=snr_min_train, high=snr_max_train, size=[1])
            noise_std_temp = np.float32(np.sqrt(1/2)*np.sqrt(P/10**(snr_temp/10)))
            h_act_batch, hR_act_batch, hI_act_batch =\
                                    generate_batch_data(batch_size,M,K,Lp,LSF_UE,Mainlobe_UE,HalfBW_UE)
            feed_dict_batch = {hR: hR_act_batch,
                               hI: hI_act_batch,
                               lay['noise_std']: noise_std_temp[0],
                               lay['aneal']: anneal_param}                        
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1
            
        if epoch%10==0: #Every 10 epochs if the DNN obtains a better performance on Validation set, then save the current parameters
            loss_test = sess.run(loss, feed_dict=feed_dict_test)
            if loss_test < best_loss:
                save_path = saver.save(sess, './params')
                best_loss = loss_test
            else:
                anneal_param = anneal_param*annealing_rate
            print('epoch',epoch,' anneal_param:%4.4f'%anneal_param)
            print('         loss_test:%2.5f'%-loss_test,'  best_test:%2.5f'%-best_loss,'  best_possible:%2.5f'%-loss_UP_test,\
                          ' Percentage::%1.3f'%(best_loss/loss_UP_test))
    loss_test_Final_B30, loss_UP_test_Final = sess.run([loss,loss_UP], feed_dict=feed_dict_test_Final)
    print(loss_test_Final_B30/loss_UP_test_Final)    
    print(-loss_UP_test_Final)
    sio.savemat('Data_K2M64Lp2L8_resultB30.mat',dict(loss_test_Final_B30=loss_test_Final_B30,\
                                            loss_UP_test_Final=loss_UP_test_Final))               