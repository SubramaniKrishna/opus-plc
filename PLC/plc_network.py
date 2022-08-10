import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, GaussianNoise, Layer, Conv2D, AveragePooling2D, Concatenate
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
import numpy as np
from tf_imdct_helpers import *

def quant_regularizer(x):
    Q = 128
    Q_1 = 1./Q
    #return .01 * tf.reduce_mean(1 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))
    return .01 * tf.reduce_mean(K.sqrt(K.sqrt(1.0001 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))))


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        # Ensure that abs of adjacent weights don't sum to more than 127. Otherwise there's a risk of
        # saturation when implementing dot products with SSSE3 or AVX2.
        return self.c*p/tf.maximum(self.c, tf.repeat(tf.abs(p[:, 1::2])+tf.abs(p[:, 0::2]), 2, axis=1))
        #return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

constraint = WeightClip(0.992)

def new_lpcnet_plc_model(rnn_units=256, nb_used_features=21, cond_size = 20,nb_burg_features=0, batch_size=128, training=False,quantize = False):
    
    feat = Input(shape=(None, nb_used_features+nb_burg_features), batch_size=batch_size)
    # Decorrelate features using DCT
    feat = tf.signal.dct(feat,type = 4)
    lost = Input(shape=(None, 1), batch_size=batch_size)

    fdense1 = Dense(cond_size, activation='tanh', name='plc_dense1')

    cfeat = Concatenate()([feat, lost])
    cfeat = fdense1(cfeat)
    #cfeat = Conv1D(cond_size, 3, padding='causal', activation='tanh', name='plc_conv1')(cfeat)

    quant = quant_regularizer if quantize else None

    if training:
        rnn = CuDNNGRU(rnn_units, return_sequences=True, return_state=True, name='plc_gru1', stateful=True,
              kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant)
        rnn2 = CuDNNGRU(rnn_units, return_sequences=True, return_state=True, name='plc_gru2', stateful=True,
              kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant)
    else:
        rnn = GRU(rnn_units, return_sequences=True, return_state=True, recurrent_activation="sigmoid", reset_after='true', name='plc_gru1', stateful=True,
                kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant)
        rnn2 = GRU(rnn_units, return_sequences=True, return_state=True, recurrent_activation="sigmoid", reset_after='true', name='plc_gru2', stateful=True,
                kernel_constraint=constraint, recurrent_constraint = constraint, kernel_regularizer=quant, recurrent_regularizer=quant)

    gru_out1, _ = rnn(cfeat)
    gru_out1 = GaussianNoise(.005)(gru_out1)
    gru_out2, _ = rnn2(gru_out1)
    
    out_dense = Dense(nb_used_features, activation='linear', name='plc_out')
    plc_out = out_dense(gru_out2)
    
    model = Model([feat, lost], plc_out)
    model.rnn_units = rnn_units
    model.cond_size = cond_size
    model.nb_used_features = nb_used_features
    model.nb_burg_features = nb_burg_features

    return model


def plc_shape_dense(nb_cepstral_features = 56, nb_used_features=0, cond_size = 4000,nb_burg_features=0, batch_size=128, band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8):
    
    input_shape = (band_defs[-1] - band_defs[0],100,1) # more than 10, 64 or 128
    feat = Input(shape=input_shape, batch_size=batch_size)
    temp = renormalizer(band_defs,'in')(feat)
    temp = tf.squeeze(temp,-1)
    temp = tf.transpose(temp,[0,2,1])
    # print(temp.shape,tf.roll(temp,-1,1).shape)
    temp = tf.concat((temp,tf.roll(temp,-1,1)),-1)
    fdense1 = Dense(cond_size, activation='tanh', name='plc_shape_dense1')
    fdense2 = Dense(cond_size, activation='tanh', name='plc_shape_dense2')
    fdense3 = Dense(cond_size, activation='tanh', name='plc_shape_dense3')
    fdenseF = Dense(nb_cepstral_features, activation='linear', use_bias = False,name='plc_out')

    temp = fdense3(fdense2(fdense1(temp)))
    plc_out = fdenseF(temp)
    plc_out = tf.expand_dims(tf.transpose(plc_out,[0,2,1]),-1)
    plc_out = tf.squeeze(renormalizer(band_defs,'out')(plc_out))
    model = Model(feat, plc_out)

    # band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8
    # band_defs = np.array([0,1,2,3,4,5,6,7])*8
    # feat = Input(shape=(2*(nb_cepstral_features + nb_used_features + nb_burg_features)), batch_size=batch_size)
    # plc_out = bandwise_FF(band_defs, cond_size, batch_size)(feat)
    # feat = tf.squeeze(Reshape((-1,2*(nb_cepstral_features + nb_used_features + nb_burg_features)))(feat))
    # Decorrelate features using DCT
    # feat = tf.signal.dct(feat,type = 4)
    # print(feat.shape)
    # fdense1 = Dense(cond_size, activation='tanh', name='plc_shape_dense1')
    # fdense2 = Dense(cond_size, activation='tanh', name='plc_shape_dense2')
    # fdense3 = Dense(cond_size, activation='tanh', name='plc_shape_dense3')
    # fdenseF = Dense(nb_cepstral_features, activation='linear', name='plc_out')

    # plc_out = fdense3(fdense2(fdense1(feat)))
    # plc_out = fdenseF(plc_out)
    # print(plc_out.shape)
    # interm = fdense2(fdense1(feat))
    # plc_out = bandwise_FF(band_defs)(interm)

    # plc_out = renormalizer(band_defs)(plc_out)
    # plc_out = tf.linalg.normalize(plc_out,axis = -1)[0] #Single band normalization
    # model = Model(feat, plc_out)
    return model

# Renormalizing Layer
class renormalizer(Layer):
    def __init__(self, band_defs,choice = 'out'):
        super(renormalizer, self).__init__()
        self.band_defs = band_defs
        self.choice = choice
 
    def call(self, inputs):  
        l = []
        if self.choice == 'out':
            for i in range(self.band_defs.shape[0] - 1):
                l.append(tf.linalg.normalize(inputs[:,self.band_defs[i] - self.band_defs[0]:self.band_defs[i+1] - self.band_defs[0],:,:],axis = 1)[0])
                # l.append(inputs[:,self.band_defs[i] - self.band_defs[0]:self.band_defs[i+1] - self.band_defs[0],:,:])
            M = tf.concat(l,1)
        else:
            for i in range(self.band_defs.shape[0] - 1):
                N = np.sqrt(self.band_defs[i+1] - self.band_defs[i])
                l.append(N*inputs[:,self.band_defs[i] - self.band_defs[0]:self.band_defs[i+1] - self.band_defs[0],:,:])
            M = tf.concat(l,1)
        return M
    
    def get_config(self):
        config = super(renormalizer, self).get_config()
        return config

# Feedforward to autoencode each band
class bandwise_FF(Layer):
    def __init__(self, band_defs, cond_size, batch_size):
        super(bandwise_FF, self).__init__()
        self.band_defs = band_defs
        self.nets = []
        for i in range(band_defs.shape[0] - 1):
            L = tf.keras.Sequential()
            L.add(Input(shape=(2*(band_defs[i + 1] - band_defs[i]),), batch_size=batch_size))
            L.add(Dense(cond_size, activation='tanh', name='plcshape_cond1_' + str(i)))
            L.add(Dense(cond_size, activation='tanh', name='plcshape_cond2_' + str(i)))
            L.add(Dense(band_defs[i + 1] - band_defs[i], activation='linear', name='plcshape_bandwise_' + str(i)))
            self.nets.append(L)
 
    def call(self, inputs):  
        l = []
        for i in range(self.band_defs.shape[0] - 1):
            inp = tf.concat([inputs[:,self.band_defs[i]- self.band_defs[0]:self.band_defs[i + 1]- self.band_defs[0]],inputs[:,self.band_defs[-1] + self.band_defs[i]- self.band_defs[0]:self.band_defs[-1] + self.band_defs[i + 1]- self.band_defs[0]]],-1)
            # print(inp.shape)
            l.append(self.nets[i](inp))
            # l.append(self.nets[i](tf.signal.dct(inputs,4)))
        M = tf.concat(l,-1)
        return M
    
    def get_config(self):
        config = super(bandwise_FF, self).get_config()
        return config

def plc_shape_cnn(nb_cepstral_features = 800, nb_used_features=0, cond_size = 4000,nb_burg_features=0, batch_size=128, band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8):
    
    # band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8
    # band_defs = np.array([0,1])*8
    feat = Input(shape=(2*(nb_cepstral_features + nb_used_features + nb_burg_features)), batch_size=batch_size)
    feat = Reshape((1,2,nb_cepstral_features))(feat)
    feat = renormalizer(band_defs,'in')(feat)
    # print(feat.shape)
    fconv1 = Conv2D(2000, (3,2), (1,1), padding = 'same', activation='tanh')
    fconv2 = Conv2D(2000, (3,2), (1,1), padding = 'same', activation='tanh')
    fconv3 = Conv2D(1000, (3,2), (1,1), padding = 'same', activation='tanh')
    ap = AveragePooling2D(pool_size=(2, 2),strides=(2, 2),padding = 'same')
    op1 = fconv1(feat)
    op1 = ap(op1)
    # op1 = Concatenate(axis = 2)([op1,feat])
    op1 = fconv2(op1)
    op1 = ap(op1)
    op1 = fconv3(op1)
    cnn_out = tf.squeeze(ap(op1))
    # cnn_out = fconv3(Concatenate(axis = 2)([op1,op2]))
    # cnn_out = tf.squeeze(ap(cnn_out))
    # print(op2.shape,cnn_out.shape)
    # cnn_out = tf.squeeze(fconv3(fconv2(ap(fconv1(feat)))))
    if batch_size == 1:
        cnn_out = tf.expand_dims(cnn_out,0)
    # print(fconv1(feat).shape)
    # print(tf.squeeze(ap(fconv1(feat))).shape)
    fdense1 = Dense(1000, activation='tanh', name='plc_shape_dense1')
    fdense2 = Dense(nb_cepstral_features, activation='linear', name='plc_shape_dense2')
    plc_out = fdense2(fdense1(cnn_out))
    # plc_out = bandwise_FF(band_defs, 100, batch_size)(cnn_out)
    # print(fdense2(fdense1(ap(fconv1(feat)).shape)))
    # plc_out = bandwise_FF(band_defs, cond_size, batch_size)(feat)
    # feat = Reshape((-1,2*821))(feat)
    # Decorrelate features using DCT
    # feat = tf.signal.dct(feat,type = 4)

    # fdense1 = Dense(cond_size, activation='tanh', name='plc_shape_dense1')
    # fdense2 = Dense(cond_size, activation='tanh', name='plc_shape_dense2')
    # fdense3 = Dense(cond_size, activation='tanh', name='plc_shape_dense3')
    # fdenseF = Dense(nb_cepstral_features, activation='linear', name='plc_out')

    # plc_out = fdense3(fdense2(fdense1(feat)))
    # plc_out = fdenseF(plc_out)

    # interm = fdense2(fdense1(feat))
    # plc_out = bandwise_FF(band_defs)(interm)

    plc_out = renormalizer(band_defs)(plc_out)
    model = Model(feat, plc_out)
    return model

def plc_shape_cnnfull(nb_cepstral_features = 800, nb_used_features=0, cond_size = 4000,nb_burg_features=0, batch_size=128, band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8):
    
    # if mode == "train":
    input_shape = (band_defs[-1] - band_defs[0],100,1) # more than 10, 64 or 128
    feat = Input(shape=input_shape, batch_size=batch_size)
    temp = feat
    # temp = renormalizer(band_defs,'in')(feat)
    # plc_out = bandwise_CNN(band_defs,1,batch_size)(feat)

    # lost = Input(shape=(1,100,1), batch_size=batch_size)
    # temp = Concatenate(axis = 1)([temp, lost])
    
    fconv1 = Conv2D(128, (8,2), 1, padding = 'same', activation='tanh', data_format = 'channels_last', bias_initializer=tf.keras.initializers.GlorotNormal())
    fconv2 = Conv2D(128, (8,1), 1, padding = 'same', activation='tanh', data_format = 'channels_last', bias_initializer=tf.keras.initializers.GlorotNormal())
    fconv3 = Conv2D(128, (8,1), 1, padding = 'same', activation='tanh', data_format = 'channels_last', bias_initializer=tf.keras.initializers.GlorotNormal())
    fconvF = Conv2D(1, (8,1), 1, padding = 'same', activation='linear',use_bias = False, data_format = 'channels_last')
    temp = fconv3(fconv2(fconv1(temp)))
    plc_out = fconvF(temp)

    plc_out = tf.squeeze(plc_out)
    # plc_out = tf.squeeze(renormalizer(band_defs,'out')(plc_out))
    # plc_out = tf.concat((plc_out,tf.zeros((batch_size,960 - band_defs[-1],100))),1)
    # plc_out = tf.squeeze(tf.roll(feat,-2,-2))
    model = Model(feat, plc_out)
    return model

def plc_shape_cnnsmoothed(nb_cepstral_features = 800, nb_used_features=0, cond_size = 4000,nb_burg_features=0, batch_size=128, band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8):
    
    # if mode == "train":
    input_shape = (band_defs[-1] - band_defs[0],100,1) # more than 10, 64 or 128
    feat = Input(shape=input_shape, batch_size=batch_size)
    temp = feat
    # temp = renormalizer(band_defs,'in')(feat)
    # plc_out = bandwise_CNN(band_defs,1,batch_size)(feat)

    # lost = Input(shape=(1,100,1), batch_size=batch_size)
    # temp = Concatenate(axis = 1)([temp, lost])
    
    fconv1 = Conv2D(128, (8,2), 1, padding = 'same', activation='tanh', data_format = 'channels_last', bias_initializer=tf.keras.initializers.GlorotNormal())
    fconv2 = Conv2D(128, (8,1), 1, padding = 'same', activation='tanh', data_format = 'channels_last', bias_initializer=tf.keras.initializers.GlorotNormal())
    fconv3 = Conv2D(128, (8,1), 1, padding = 'same', activation='tanh', data_format = 'channels_last', bias_initializer=tf.keras.initializers.GlorotNormal())
    fconv_mu = Conv2D(1, (8,1), 1, padding = 'same', activation='linear',use_bias = False, data_format = 'channels_last')
    fconv_sigma = Conv2D(1, (8,1), 1, padding = 'same', activation='linear', data_format = 'channels_last')
    temp = fconv3(fconv2(fconv1(temp)))
    plc_out_mu = fconv_mu(temp)
    plc_out_sigma = fconv_sigma(temp)

    plc_out = tf.concat((plc_out_mu,plc_out_sigma),-1)
    # plc_out = tf.squeeze(renormalizer(band_defs,'out')(plc_out))
    # plc_out = tf.concat((plc_out,tf.zeros((batch_size,960 - band_defs[-1],100))),1)
    # plc_out = tf.roll(temp,-2,-2)
    # plc_out = tf.concat((plc_out,tf.zeros_like(plc_out) + 1.0e-6),-1)
    model = Model(feat, plc_out)
    return model


# CNN to autoencode each band
class bandwise_CNN(Layer):
    def __init__(self, band_defs, cond_size, batch_size):
        super(bandwise_CNN, self).__init__()
        self.band_defs = band_defs
        self.nets = []
        for i in range(band_defs.shape[0] - 1):
            L = tf.keras.Sequential()
            input_shape = (band_defs[i+1] - band_defs[i],100,1)
            L.add(Input(shape=input_shape, batch_size=batch_size))

            L.add(Conv2D(256, (8,2), 1, padding = 'same', activation='tanh', data_format = 'channels_last'))
            L.add(Conv2D(256, (8,1), 1, padding = 'same', activation='tanh', data_format = 'channels_last'))
            L.add(Conv2D(128, (8,1), 1, padding = 'same', activation='tanh', data_format = 'channels_last'))
            L.add(Conv2D(1, (8,1), 1, padding = 'same', activation='linear',use_bias = False, data_format = 'channels_last'))

            self.nets.append(L)
 
    def call(self, inputs):  
        l = []
        for i in range(self.band_defs.shape[0] - 1):
            inp = inputs[:,self.band_defs[i]- self.band_defs[0]:self.band_defs[i + 1]- self.band_defs[0],:,:]
            l.append(self.nets[i](inp))
        M = tf.concat(l,1)
        return M
    
    def get_config(self):
        config = super(bandwise_CNN, self).get_config()
        return config