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
from mult_mod import mult_mod_complex
from tensorflow.python.keras.layers import *
'System Parameters'
M = 64 #Number of BS antennas
P = 1 #Power
K =  3 #Number of users
L = 8 #Number of pilots
Lp = 2 #Number of paths
B = 30 #Number of feedback bits per user
LSF_UE = np.array(0.0*np.ones(K),dtype=np.float32)#Mean of path gains for K users
Mainlobe_UE= np.array(0.0*np.ones(K),dtype=np.float32) #Center of the AoD range for K users
HalfBW_UE = np.array(60*np.ones(K),dtype=np.float32) #Half of the AoD range for K users

snr_dl = 10 #SNR in dB
snr_max_train = 10 #max training SNR in dB
snr_min_train = 10 #min training SNR in dB
noise_std_dl = np.float32(np.sqrt(1/2)*np.sqrt(P/10**(snr_dl/10))) #STD of the Gaussian noise (per real dim.)
#####################################################
'Learning Parameters'
initial_run = 0 #1: starts training from scratch; 0: resumes training
n_epochs = 00000000 #Number of training epochs, for observing the current performance set it to 0
learning_rate = 0.0001 #Learning rate
batch_size = 1024 #Mini-batch size
test_size = 10000 #Size of the validation/test set
batch_per_epoch = 20 #Numbers of mini-batches per epoch
anneal_param = 1.0 #Initial annealing parmeter
annealing_rate = 1.001 #Annealing rate
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Determine the DNN initialization method       
######################################################
#Place holders
hR = tf.placeholder(tf.float32, shape=(None,M,K), name="hR")
hI = tf.placeholder(tf.float32, shape=(None,M,K), name="hI")
###################### Load weights/biases for the common user-side DNN
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
    lay['noise_std'] = tf.constant(noise_std_dl) #STD of the Gaussian noise (per real dim.)
    lay['aneal'] = tf.constant(1.0) #Annealing parameter
    ############### User's operations
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
        DNN_input = y_noisy[kk]
        dense_layer1 = tf.nn.relu(tf.add(tf.matmul(DNN_input,A1),b1))
        dense_layer1 = BatchNormalization()(dense_layer1)
        dense_layer2 = tf.nn.relu(tf.add(tf.matmul(dense_layer1,A2),b2))
        dense_layer2 = BatchNormalization()(dense_layer2)
        dense_layer3 = tf.nn.relu(tf.add(tf.matmul(dense_layer2,A3),b3))
        dense_layer3 = BatchNormalization()(dense_layer3)
        InfoBits_linear = tf.add(tf.matmul(dense_layer3,A4),b4)
        InfoBits_tanh = tf.tanh(lay['aneal']*InfoBits_linear)
        InfoBits_sign = tf.sign(InfoBits_linear)
        InfoBits[kk] = InfoBits_tanh + tf.stop_gradient(InfoBits_sign - InfoBits_tanh)
        if kk == 0:
            DNN_input_BS = InfoBits[kk]
        else:
            DNN_input_BS = tf.concat([DNN_input_BS,InfoBits[kk]],axis=1)     
#######################################################################################
with tf.name_scope("BS_operations"):
    dense_layer4 = Dense(units=2048, activation='relu')(DNN_input_BS)
    dense_layer4 = BatchNormalization()(dense_layer4)
    dense_layer5 = Dense(units=1024, activation='relu')(dense_layer4)
    dense_layer5 = BatchNormalization()(dense_layer5)
    dense_layer6 = Dense(units=512, activation='relu')(dense_layer5)
    dense_layer6 = BatchNormalization()(dense_layer6)
    dense_layer7 = Dense(units=512, activation='relu')(dense_layer6)
    dense_layer7 = BatchNormalization()(dense_layer7)
        
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
feed_dict_test = {hR: hR_act_test,
                  hI: hI_act_test,
                  lay['noise_std']: noise_std_dl,
                  lay['aneal']: 1.0}
rate_MRT, rate_ZF = generate_MRT_ZF_performance(h_act_test,noise_std_dl,P)

AA = sio.loadmat('Data_test_K{}M{}Lp{}_{}HBW_withParams.mat'.format(K,M,Lp,int(HalfBW_UE[0])))
h_act_test_Final = AA['h_act_test_Final']
hR_act_test_Final = AA['hR_act_test_Final']
hI_act_test_Final = AA['hI_act_test_Final']
feed_dict_test_Final = {hR: hR_act_test_Final,
                  hI: hI_act_test_Final,
                  lay['noise_std']: noise_std_dl,
                  lay['aneal']: 1.0}
############  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './params')
    best_loss = sess.run(loss, feed_dict=feed_dict_test)
    print(-best_loss)
    print(tf.test.is_gpu_available())
    #rate_batch = np.zeros((batch_per_epoch))
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
            #rate_batch[batch_iter] = -sess.run(loss, feed_dict=feed_dict_batch)
            batch_iter += 1
            
        if epoch%10==0:
            loss_test = sess.run(loss, feed_dict=feed_dict_test)
            if loss_test < best_loss:
                save_path = saver.save(sess, './params')
                best_loss = loss_test
            else:
                anneal_param = anneal_param*1.001
            #print('epoch',epoch,' train_acc:%0.5f'%np.mean(rate_batch),' anneal_param:%4.4f'%anneal_param)
            print('epoch',epoch,' anneal_param:%4.4f'%anneal_param)
            print('         loss_test:%2.5f'%-loss_test,'  best_test:%2.5f'%-best_loss,'  Percntage ZF:%1.3f'%(-best_loss/rate_ZF),\
                          ' Percentage::%1.3f'%(-best_loss/rate_MRT))
    loss_test_Final = sess.run(loss, feed_dict=feed_dict_test_Final)
    rate_MRT, rate_ZF = generate_MRT_ZF_performance(h_act_test_Final,noise_std_dl,P)
    print(-loss_test_Final/rate_ZF)    
    sio.savemat('Data_K{}M{}Lp{}_resultB{}L{}_K1.mat'.format(K,M,Lp,B,L),dict(loss_test_Final=loss_test_Final,\
                                            rate_MRT=rate_MRT, rate_ZF=rate_ZF,B=B,L=L))        