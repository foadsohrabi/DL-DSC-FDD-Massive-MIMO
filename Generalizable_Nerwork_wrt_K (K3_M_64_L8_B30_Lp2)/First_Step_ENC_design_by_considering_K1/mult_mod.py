import tensorflow as tf

def mult_mod(M,N,left_right):
    tensor_shape = M.shape
    dims = N.shape 
    if left_right == 'r':
        #M tensor of size (batch_size, n, m)
        #N tensor of size (m, p)
        n = tensor_shape[1].value
        m = dims[0]
        p = dims[1]
        y = tf.reshape(tf.reshape(M, [-1, m]) @ N, [-1, n, p])
    elif left_right == 'l':
        #M tensor of size (batch_size, n, m)
        #N tensor of size (p, n)
        m = tensor_shape[2].value
        p = dims[0]
        n = dims[1]        
        MT = tf.matrix_transpose(M)
        NT = tf.matrix_transpose(N)
        MTNT = tf.reshape(tf.reshape(MT, [-1, n]) @ NT, [-1, m, p])
        y = tf.matrix_transpose(MTNT)
    
    return(y)
    
def mult_mod_complex(Mr,Mi,Nr,Ni,left_right):
    yr = mult_mod(Mr,Nr,left_right) -  mult_mod(Mi,Ni,left_right)
    yi = mult_mod(Mr,Ni,left_right) +  mult_mod(Mi,Nr,left_right)
    
    return(yr,yi)    
    
