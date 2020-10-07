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
'System Parameters'
M = 64 #Number of antennas
P = 1 #Power
K =  1 #Number of users
L = 8 #Number of pilots
Lp = 2 #Number of paths
B = 30 #Number of feedback bits per user
LSF_UE = np.array([0.0],dtype=np.float32) #Mean of path gains
Mainlobe_UE= np.array([0],dtype=np.float32) #Mean of the AoD range
HalfBW_UE = np.array([30.0],dtype=np.float32) #Half of the AoD range

snr_dl = 10 #SNR in dB
snr_max_train = 10 #max training SNR in dB
snr_min_train = 10 #min training SNR in dB
noise_std_dl = np.float32(np.sqrt(1/2)*np.sqrt(P/10**(snr_dl/10))) #STD of the Gaussian noise (per real dim.)
#####################################################
'Learning Parameters'
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
tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()
######################################################
Xp_init = np.random.normal(0.0,1/np.sqrt(2),(L,M)) \
                + 1j*np.random.normal(0.0,1/np.sqrt(2),(L,M))
Xp_r_init = np.float32(np.real(Xp_init)) 
Xp_i_init = np.float32(np.imag(Xp_init))         
######################################################
hR = tf.placeholder(tf.float32, shape=(None,M,K), name="hR")
hI = tf.placeholder(tf.float32, shape=(None,M,K), name="hI")
###################### NETWORK
with tf.name_scope("DL_training_phase"):
    lay = {}
    lay['noise_std'] = tf.constant(noise_std_dl)
    lay['aneal'] = tf.constant(1.0)
    ############### User's operations
    Xp_r = tf.get_variable("Xp_r",  dtype=tf.float32, initializer= Xp_r_init)
    Xp_i = tf.get_variable("Xp_i",  dtype=tf.float32, initializer= Xp_i_init)
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
        DNN_input = y_noisy[kk]
        dense_layer1 = Dense(units=1024, activation='relu')(DNN_input)
        dense_layer1 = BatchNormalization()(dense_layer1)
        dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
        dense_layer2 = BatchNormalization()(dense_layer2)
        dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)
        dense_layer3 = BatchNormalization()(dense_layer3)
        InfoBits_linear = Dense(units=B, activation='linear')(dense_layer3)
        InfoBits_tanh = tf.tanh(lay['aneal']*InfoBits_linear)
        InfoBits_sign = tf.sign(InfoBits_linear)
        InfoBits[kk] = InfoBits_tanh + tf.stop_gradient(InfoBits_sign - InfoBits_tanh)
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
        
    hhat = Dense(units=2*M, activation='linear')(dense_layer6)         
#####################################################################################
######## Loss Function
hvec = tf.reshape(tf.concat([hR,hI],axis=1),[-1,2*M])
loss = tf.reduce_mean(tf.keras.losses.MSE(hvec, hhat))
######### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#######################################################################
############  Validation Set
h_act_test, hR_act_test, hI_act_test = generate_batch_data(test_size,M,K,Lp,LSF_UE,Mainlobe_UE,HalfBW_UE)
feed_dict_test = {hR: hR_act_test,
                  hI: hI_act_test,
                  lay['noise_std']: noise_std_dl,
                  lay['aneal']: 1.0}
#############  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './params')
    best_loss = sess.run(loss, feed_dict=feed_dict_test)
    print(best_loss)
    print(tf.test.is_gpu_available())
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
            
        if epoch%10==0:
            loss_test = sess.run(loss, feed_dict=feed_dict_test)
            if loss_test < best_loss:
                save_path = saver.save(sess, './params')
                best_loss = loss_test
            else:
                anneal_param = anneal_param*1.000
            print('epoch',epoch,' anneal_param:%4.4f'%anneal_param)
            print('         loss_test:%2.5f'%loss_test,'  best_test:%2.5f'%best_loss)  

    AA = sio.loadmat('Data_test_K2M{}Lp{}_{}HBW_withParams.mat'.format(M,Lp,int(HalfBW_UE[0])))
    h_act_test_Final0 = np.reshape(AA['h_act_test_Final'][:,:,0],[-1,M,1])
    hR_act_test_Final0 = np.reshape(AA['hR_act_test_Final'][:,:,0],[-1,M,1])
    hI_act_test_Final0 = np.reshape(AA['hI_act_test_Final'][:,:,0],[-1,M,1])
    feed_dict_test_Final0 = {hR: hR_act_test_Final0,
                      hI: hI_act_test_Final0,
                      lay['noise_std']: noise_std_dl,
                      lay['aneal']: 1.0}
    h_act_test_Final1 = np.reshape(AA['h_act_test_Final'][:,:,1],[-1,M,1])
    hR_act_test_Final1 = np.reshape(AA['hR_act_test_Final'][:,:,1],[-1,M,1])
    hI_act_test_Final1 = np.reshape(AA['hI_act_test_Final'][:,:,1],[-1,M,1])
    feed_dict_test_Final1 = {hR: hR_act_test_Final1,
                      hI: hI_act_test_Final1,
                      lay['noise_std']: noise_std_dl,
                      lay['aneal']: 1.0}
             
    loss_test_Final0,hhat0 = sess.run([loss,hhat], feed_dict=feed_dict_test_Final0)
    loss_test_Final1,hhat1 = sess.run([loss,hhat], feed_dict=feed_dict_test_Final1)
    sio.savemat('hhat_data_B{}_L{}.mat'.format(B,L),dict(h_act_test_Final0=h_act_test_Final0,\
                                     h_act_test_Final1=h_act_test_Final1,\
                                     hhat0=hhat0,\
                                     hhat1=hhat1,L=L,B=B))          
     