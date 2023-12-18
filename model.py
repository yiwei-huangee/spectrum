import tensorflow as tf
import numpy as np
from args import args

# Query, Key and Value
class Value(tf.Module):
    def __init__(self, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = tf.keras.layers.Dense(dim_val,activation='relu')
    def __call__(self, x):
        x = self.fc1(x)       
        return x

class Key(tf.Module):
    def __init__(self, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = tf.keras.layers.Dense(dim_attn,activation='relu')
    
    def __call__(self, x):
        x = self.fc1(x)        
        return x

class Query(tf.Module):
    def __init__(self, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn        
        self.fc1 = tf.keras.layers.Dense(dim_attn,activation='relu')
        
    def __call__(self, x):        
        x = self.fc1(x)
        return x


def time_step_a_norm(Q, K):
    m = tf.matmul(Q, tf.transpose(K,perm=[0,2,1]))
    m /= tf.math.sqrt(tf.convert_to_tensor(Q.shape[-1],dtype=tf.float32))
    return tf.nn.softmax(m,-1)

def time_step_attention(Q, K, V):
    a = time_step_a_norm(Q, K)  
    return  tf.matmul(a,  V)

class time_step_AttentionBlock(tf.Module):
    def __init__(self, dim_val, dim_attn):
        super(time_step_AttentionBlock, self).__init__()
        self.value = Value(dim_val)
        self.key = Key(dim_attn)
        self.query = Query(dim_attn)
    
    def __call__(self, x, kv = None):
        if(kv is None):
            return time_step_attention(self.query(x), self.key(x), self.value(x))        
        return time_step_attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(tf.Module):  
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(MultiHeadAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.dim_val = dim_val
        self.dim_attn = dim_attn
        self.heads = []
        for _ in range(n_heads):
            self.heads.append(time_step_AttentionBlock(tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),  tf.cast(self.dim_attn/self.n_heads,dtype=tf.int32)))
        self.fc = tf.keras.layers.Dense(dim_val)
                      
        
    def __call__(self, x, kv = None):
        a = []
        x_split = tf.reshape(x, [-1,x.shape[1],tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),self.n_heads])
        for h in self.heads:
            i = 0
            a.append(h(x_split[:,:,:,i], kv = kv))
            i = i + 1
        a = tf.concat(a[0:self.n_heads],axis=-1)
        x = self.fc(a)
        return x

def ConvLSTM(inputs,kernal_size=3,dropout_prob=0.0):
    # 1D Convolutional LSTM Block
    x = tf.keras.layers.ConvLSTM1D(512, kernal_size, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.ConvLSTM1D(512, kernal_size, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    convlstm = tf.keras.layers.Activation('relu')(x)
    convlstm = tf.keras.layers.Dropout(dropout_prob)(convlstm)
    return convlstm

def Conv_Block1(inputs, kernel_size=3, dropout_prob=0.0):
    # 1D Convolutional Block
    inputs = tf.expand_dims(inputs, axis=-1)
    x = tf.keras.layers.Conv1D(8, kernel_size, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(8, kernel_size, padding='causal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    conv = tf.keras.layers.Activation('relu')(x)

    conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    return conv

def Conv_Block2(inputs, kernel_size=3, dropout_prob=0.0):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(16, kernel_size, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(16, kernel_size, padding='causal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    conv = tf.keras.layers.Activation('relu')(x)

    conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    return conv

def Conv_Block3(inputs, kernel_size=3, dropout_prob=0.0):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(32, kernel_size, padding='causal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(32, kernel_size, padding='causal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    conv = tf.keras.layers.Activation('relu')(x)

    conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    return conv

def Conv_T(inputs, kernel_size=3, strides = 4, dropout_prob=0.0):
    # 1D Convolutional Block
    Y = []
    for i in range(args.N):
        x = tf.keras.layers.Conv1DTranspose(128, kernel_size, strides, padding='same')(inputs[:,i,:,:])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1DTranspose(32, kernel_size, strides, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1DTranspose(1, kernel_size, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        conv = tf.keras.layers.Dropout(dropout_prob)(x)
        Y.append(conv)
    conv = tf.concat(Y[0:args.N],axis=1)
    conv = tf.reshape(conv,[-1,args.N,2048,1])
    return conv

def Maxpool (inputs, pool_size=2, strides=4, dropout_prob=0.0):
    # 1D Convolutional Block
    Y = []
    for i in range(args.N):
        x = tf.keras.layers.MaxPool1D(pool_size, strides)(inputs[:,i,:,:])
        x = tf.keras.layers.MaxPool1D(pool_size, strides)(x)
    Y.append(x)
    y = tf.concat(Y[0:args.N],axis=1)
    y = tf.reshape(y,[-1,args.N,64,16])
    return y

class spectrum_estimate(tf.Module):
    def __init__(self,args):
        super(spectrum_estimate,self).__init__()
        
        self.MultiHeadAttentionBlock = MultiHeadAttentionBlock(args.dim_val, args.dim_attn, args.n_heads)
        
        self.maxpool3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)
        self.dense1 = tf.keras.layers.Dense(1638,activation='relu')
        self.dense2 = tf.keras.layers.Dense(64,activation='relu')

        self.dense3 = tf.keras.layers.Dense(8000,activation='relu')
        self.dense4 = tf.keras.layers.Dense(args.N,activation='relu')
        self.dense5 = tf.keras.layers.Dense(1638,activation='relu')

    def __call__(self, input):
        # encoder
        y = Conv_Block1(input[:,:args.N,:])
        embedding = tf.keras.layers.Concatenate(axis=3)([tf.expand_dims(input[:,args.N:,:], axis=-1), y])
        y = Conv_Block2(embedding)
        y = Conv_Block3(y)
        y = Maxpool(y)
        # y = self.MultiHeadAttentionBlock(y) # 256x64
        
        # decoder
        y=tf.keras.layers.ConvLSTM1D(64,3,return_sequences=True,padding='same')(y) # 64,512
        y = Conv_T(y) # 2048,1
        
        y = self.dense1(tf.squeeze(y,axis=-1))
        y = self.dense2(y) # 1,1638
        y = tf.keras.layers.Flatten()(y)
        out = self.dense5(self.dense4(self.dense3(y)))
        return out
    
def model_test():
    inputs = tf.keras.Input(shape=(args.N *2,args.input_size))
    # separate the input into two parts, first half is the signal, second half is the arrival time
    outputs = spectrum_estimate(args)(inputs)
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    return model