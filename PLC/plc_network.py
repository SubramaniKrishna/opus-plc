import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, GaussianNoise, Layer
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
import numpy as np

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


def plc_shape(nb_cepstral_features = 800, nb_used_features=0, cond_size = 4000,nb_burg_features=0, batch_size=128):
    
    feat = Input(shape=(2*(nb_cepstral_features + nb_used_features + nb_burg_features)), batch_size=batch_size)
    # feat = Reshape((-1,2*821))(feat)
    # Decorrelate features using DCT
    feat = tf.signal.dct(feat,type = 4)

    fdense1 = Dense(cond_size, activation='tanh', name='plc_shape_dense1')
    fdense2 = Dense(cond_size, activation='tanh', name='plc_shape_dense2')
    fdense3 = Dense(cond_size, activation='tanh', name='plc_shape_dense3')
    fdenseF = Dense(nb_cepstral_features, activation='linear', name='plc_out')

    plc_out = fdense3(fdense2(fdense1(feat)))
    plc_out = fdenseF(plc_out)
    band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8
    plc_out = renormalizer(band_defs)(plc_out)
    model = Model(feat, plc_out)
    return model

# Renormalizing Layer
class renormalizer(Layer):
    def __init__(self, band_defs):
        super(renormalizer, self).__init__()
        self.band_defs = band_defs
 
    def call(self, inputs):  
        l = []
        for i in range(self.band_defs.shape[0] - 1):
            l.append(tf.linalg.normalize(inputs[:,self.band_defs[i]:self.band_defs[i+1]],axis = -1)[0])
        M = tf.concat(l,-1)
        return M
    
    def get_config(self):
        config = super(renormalizer, self).get_config()
        return config