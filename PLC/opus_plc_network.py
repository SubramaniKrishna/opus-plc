import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, GaussianNoise, Layer, Conv2D, AveragePooling2D, Concatenate
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback

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

def plc_energy(rnn_units=256, nb_used_features=21, cond_size = 20, batch_size=128, training=False,quantize = False):
    
    feat = Input(shape=(None, nb_used_features), batch_size=batch_size)
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
    
    return model

def plc_shape(nb_cepstral_features = 800, nb_used_features=0, cond_size = 4000,nb_burg_features=0, batch_size=128, band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8):
    
    # if mode == "train":
    input_shape = (band_defs[-1] - band_defs[0],None,1) # more than 10, 64 or 128
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