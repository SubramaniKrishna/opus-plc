import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import argparse
from plc_loader import *
from tf_imdct_helpers import *
ebandMeans = np.array([
      6.437500, 6.250000, 5.750000, 5.312500, 5.062500,
      4.812500, 4.500000, 4.375000, 4.875000, 4.687500,
      4.562500, 4.437500, 4.875000, 4.625000, 4.312500,
      4.500000, 4.375000, 4.625000, 4.750000, 4.437500,
      3.750000, 3.750000, 3.750000, 3.750000, 3.750000
]).astype('float')

parser = argparse.ArgumentParser(description='Train a PLC model')

parser.add_argument('features_logE', metavar='<features file>', help='binary features file containing log-Energies(float32)')
parser.add_argument('features_mdct', metavar='<features file>', help='binary features file containing normalized mdcts (float32)')
parser.add_argument('lost_file', metavar='<packet loss file>', help='packet loss traces (int8)')
parser.add_argument('output', metavar='<output>', help='trained model file (.h5)')
parser.add_argument('--model', metavar='<model>', default='plc_network', help='PLC model python definition (without .py)')
group1 = parser.add_mutually_exclusive_group()
group1.add_argument('--quantize', metavar='<input weights>', help='quantize model')
group1.add_argument('--retrain', metavar='<input weights>', help='continue training model')
parser.add_argument('--gru-size', metavar='<units>', default=256, type=int, help='number of units in GRU (default 256)')
parser.add_argument('--cond-size', metavar='<units>', default=128, type=int, help='number of units in conditioning network (default 128)')
parser.add_argument('--epochs', metavar='<epochs>', default=120, type=int, help='number of epochs to train for (default 120)')
parser.add_argument('--batch-size', metavar='<batch size>', default=128, type=int, help='batch size to use (default 128)')
parser.add_argument('--seq-length', metavar='<sequence length>', default=1000, type=int, help='sequence length to use (default 1000)')
parser.add_argument('--lr', metavar='<learning rate>', type=float, help='learning rate')
parser.add_argument('--decay', metavar='<decay>', type=float, help='learning rate decay')
parser.add_argument('--band-loss', metavar='<weight>', default=1.0, type=float, help='weight of band loss (default 1.0)')
parser.add_argument('--loss-bias', metavar='<bias>', default=0.0, type=float, help='loss bias towards low energy (default 0.0)')
parser.add_argument('--logdir', metavar='<log dir>', help='directory for tensorboard log files')


args = parser.parse_args()

import importlib
lpcnet = importlib.import_module(args.model)

import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import h5py

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
import tensorflow_io as tfio
#if gpus:
#  try:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#  except RuntimeError as e:
#    print(e)

nb_epochs = args.epochs

# Try reducing batch_size if you run out of memory on your GPU
batch_size = args.batch_size

quantize = args.quantize is not None
retrain = args.retrain is not None

if quantize:
    lr = 0.00003
    decay = 0
    input_model = args.quantize
else:
    lr = 1.0e-2
    decay = 0

if args.lr is not None:
    lr = args.lr

if args.decay is not None:
    decay = args.decay

if retrain:
    input_model = args.retrain


def plc_mdct_loss():
    def loss(y_true,y_pred):
        # mask = y_true[:,:,-1:]
        # y_true = y_true[:,:,:-1]
        # e = (y_pred - y_true)*mask
        # l1_loss = K.mean(K.abs(e))

        e = (y_pred - y_true)
        rms_loss = K.sqrt(K.sum(K.square(e),axis = -1))
        return rms_loss
    return loss

band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8
# band_defs = np.array([0,1])*8
def plc_mdctshape_loss_bandwise(alpha,band_defs):
    def loss(y_true,y_pred):
        e = (y_pred - y_true)
        abs_e = K.abs(y_pred) - K.abs(y_true)
        rms_loss = 0
        abs_rms = 0
        # print(y_pred.shape)
        # bw = np.flip(np.arange(1,band_defs.shape[0]))
        # bw = bw/bw.sum()
        for i in range(band_defs.shape[0] - 1):
            rms_loss+= K.sqrt(K.sum(K.square(e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1))
            abs_rms+= K.sqrt(K.sum(K.square(abs_e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1))
        # rms_loss = K.tanh(K.mean(rms_loss)/2)
        rms_loss = K.mean(rms_loss)
        abs_rms = K.mean(abs_rms)
        return ((1 - alpha)*rms_loss + alpha*abs_rms)
    return loss

def plc_mdctshape_spectralloss():
    def loss(y_true,y_pred):
        gt_E = tf.roll(y_true[:,:,:21],-2,1)
        gt_mdct = tf.concat((tf.transpose(tf.roll(y_true[:,:,21:],0,1),[0,2,1]),tf.zeros((16,160,100))),1)
        gt_audio = tf_imdct_ola(gt_mdct,gt_E,960*2,480*2,960*100,band_defs,ebandMeans)
        pred_audio = tf_imdct_ola(y_pred,gt_E,960*2,480*2,960*100,band_defs,ebandMeans)
        spec_gt = tfio.audio.spectrogram(gt_audio,960*2,960*2,480*2)
        spec_pred = tfio.audio.spectrogram(pred_audio,960*2,960*2,480*2)

        e = (spec_gt - spec_pred)**2
        # print("LOSS",K.mean(e).shape)
        return K.mean(e)
    return loss

def plcmdct_metric(y_true,y_pred):
        alpha = 0
        e = (y_pred - y_true)
        abs_e = K.abs(y_pred) - K.abs(y_true)
        rms_loss = 0
        abs_rms = 0
        # print(y_pred.shape)
        # bw = np.flip(np.arange(1,band_defs.shape[0]))
        # bw = bw/bw.sum()
        for i in range(band_defs.shape[0] - 1):
            rms_loss+= K.sqrt(K.sum(K.square(e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1))
            abs_rms+= K.sqrt(K.sum(K.square(abs_e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1))
        # rms_loss = K.tanh(K.mean(rms_loss)/2)
        rms_loss = K.mean(rms_loss)
        abs_rms = K.mean(abs_rms)
        return ((1 - alpha)*rms_loss + alpha*abs_rms)

def plc_loss(alpha=1.0, bias=0.):
    def loss(y_true,y_pred):
        mask = y_true[:,:,-1:]
        y_true = y_true[:,:,:-1]
        e = (y_pred - y_true)*mask
        e_bands = tf.signal.idct(e[:,:,:-2], norm='ortho')
        bias_mask = K.minimum(1., K.maximum(0., 4*y_true[:,:,-1:]))
        l1_loss = K.mean(K.abs(e)) + 0.1*K.mean(K.maximum(0., -e[:,:,-1:])) + alpha*K.mean(K.abs(e_bands) + bias*bias_mask*K.maximum(0., e_bands)) + K.mean(K.minimum(K.abs(e[:,:,18:19]),1.)) + 8*K.mean(K.minimum(K.abs(e[:,:,18:19]),.4))
        return l1_loss
    return loss

def plc_l1_loss():
    def L1_loss(y_true,y_pred):
        mask = y_true[:,:,-1:]
        y_true = y_true[:,:,:-1]
        e = (y_pred - y_true)*mask
        l1_loss = K.mean(K.abs(e))
        return l1_loss
    return L1_loss

def plc_ceps_loss():
    def ceps_loss(y_true,y_pred):
        mask = y_true[:,:,-1:]
        y_true = y_true[:,:,:-1]
        e = (y_pred - y_true)*mask
        l1_loss = K.mean(K.abs(e[:,:,:-2]))
        return l1_loss
    return ceps_loss

def plc_band_loss():
    def L1_band_loss(y_true,y_pred):
        mask = y_true[:,:,-1:]
        y_true = y_true[:,:,:-1]
        e = (y_pred - y_true)*mask
        e_bands = tf.signal.idct(e[:,:,:-2], norm='ortho')
        l1_loss = K.mean(K.abs(e_bands))
        return l1_loss
    return L1_band_loss

def plc_pitch_loss():
    def pitch_loss(y_true,y_pred):
        mask = y_true[:,:,-1:]
        y_true = y_true[:,:,:-1]
        e = (y_pred - y_true)*mask
        l1_loss = K.mean(K.minimum(K.abs(e[:,:,18:19]),.4))
        return l1_loss
    return pitch_loss

opt = Adam(lr, decay=decay, beta_2=0.99)
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# with strategy.scope():
# Energy Training
# model = lpcnet.new_lpcnet_plc_model(rnn_units=args.gru_size, batch_size=batch_size, training=True, quantize=quantize, cond_size=args.cond_size)
# model.compile(optimizer=opt, loss=plc_mdct_loss(), metrics=[plc_mdct_loss()])
# model.summary()

# Shape Training
# model = lpcnet.plc_shape(nb_cepstral_features = int(band_defs[-1] - band_defs[0]),batch_size=batch_size, cond_size=args.cond_size, band_defs = band_defs)
# model = lpcnet.plc_shape_cnn(nb_cepstral_features = int(band_defs[-1] - band_defs[0]), batch_size=batch_size, band_defs = band_defs)
# model = lpcnet.plc_shape_dense(nb_cepstral_features = int(band_defs[-1] - band_defs[0]),batch_size=batch_size, cond_size=args.cond_size, band_defs = band_defs) # Dense
model = lpcnet.plc_shape_cnnfull(nb_cepstral_features = int(band_defs[-1] - band_defs[0]), batch_size=batch_size, band_defs = band_defs) # CNN
model.compile(optimizer=opt, loss=plc_mdctshape_loss_bandwise(0.0,band_defs), metrics = [plcmdct_metric]) #Bandwise CNN Loss
# model.compile(optimizer=opt, loss=plc_mdctshape_spectralloss(), metrics = [plcmdct_metric]) #Spectral Loss
# model.compile(optimizer=opt, loss=plc_mdct_loss())
model.summary()


feature_file_logE = args.features_logE
feature_file_mdct = args.features_mdct
nb_features = 21
# nb_features = model.nb_used_features + model.nb_burg_features
# nb_used_features = model.nb_used_features
# nb_burg_features = model.nb_burg_features
# sequence_size = args.seq_length

# u for unquantised, load 16 bit PCM samples and convert to mu-law
# features = np.load("./feature_dump.npy")
# nb_sequences = features.shape[0]//sequence_size
# features = features[:nb_sequences*sequence_size,:]
# features = np.reshape(features, (nb_sequences, sequence_size, nb_features))
# lost = np.random.randint(2, size=(nb_sequences,sequence_size,1))
# print(features.shape,lost.shape)

# features = np.memmap(feature_file, dtype='float32', mode='r')
# nb_sequences = len(features)//(nb_features*sequence_size)//batch_size*batch_size
# features = features[:nb_sequences*sequence_size*nb_features]

# features = np.reshape(features, (nb_sequences, sequence_size, nb_features))

# features = features[:, :, :nb_used_features+model.nb_burg_features]

# lost = np.memmap(args.lost_file, dtype='int8', mode='r')

features_E = np.memmap(feature_file_logE, dtype='float32', mode='r')
features_mdct = np.memmap(feature_file_mdct, dtype='float32', mode='r')

sequence_size_shape = 100
nb_features_cepstral = 960
nb_sequences_shape = len(features_mdct)//(nb_features_cepstral*sequence_size_shape)//batch_size*batch_size
features_mdct = features_mdct[:nb_sequences_shape*sequence_size_shape*nb_features_cepstral]
features_mdct = np.reshape(features_mdct, (nb_sequences_shape, sequence_size_shape, nb_features_cepstral))
features_mdct = features_mdct[:(int)(1*nb_sequences_shape), :, band_defs[0]:band_defs[-1]]

nb_sequences_E = len(features_E)//(nb_features*sequence_size_shape)//batch_size*batch_size
features_E = features_E[:nb_sequences_E*sequence_size_shape*nb_features]
features_E = np.reshape(features_E, (nb_sequences_E, sequence_size_shape, nb_features))
features_E = features_E[:(int)(1*nb_sequences_E), :, :]
# print(nb_sequences_shape,nb_sequences_E)
# features_inp_shape = np.concatenate((features_mdct,features_E),axis = -1)

# features = features[:nb_sequences*sequence_size*nb_features]

# features = np.reshape(features, (nb_sequences, sequence_size, nb_features))

# features = features[:, :, :nb_used_features+model.nb_burg_features]


# dump models to disk as we go
checkpoint = ModelCheckpoint('{}_{}_{}_{}.h5'.format(args.output, args.gru_size, args.cond_size, '{epoch:02d}'), save_freq = "epoch")

# if args.retrain is not None:
#     model.load_weights(args.retrain)

# if quantize or retrain:
#     #Adapting from an existing model
#     model.load_weights(input_model)

model.save_weights('{}_{}_{}_initial.h5'.format(args.output, args.gru_size, args.cond_size))
# print(features_E.shape,features_mdct.shape)
# loader = PLCLoader(features, lost, nb_burg_features, batch_size)
loader = PLCLoader_shape(features_mdct,features_E,sequence_size_shape,batch_size,0)

csv_logger = CSVLogger('train_loss.csv', append=False, separator=',')
callbacks = [checkpoint, csv_logger]
if args.logdir is not None:
    logdir = '{}/{}_{}_{}_logs'.format(args.logdir, args.output, args.gru_size, args.cond_size)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks.append(tensorboard_callback)

model.fit(loader, epochs=nb_epochs, validation_split=0.0, callbacks=callbacks)

