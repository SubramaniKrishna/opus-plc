"""
Common code to train both the energy and shape networks
Comment out the respective model loader and data loader code to select the model to train
"""

import os
# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import argparse
import numpy as np
# Utility arrays defining the band mean energies from CELT and the band boundary definitions
ebandMeans = np.array([
      6.437500, 6.250000, 5.750000, 5.312500, 5.062500,
      4.812500, 4.500000, 4.375000, 4.875000, 4.687500,
      4.562500, 4.437500, 4.875000, 4.625000, 4.312500,
      4.500000, 4.375000, 4.625000, 4.750000, 4.437500,
      3.750000, 3.750000, 3.750000, 3.750000, 3.750000
]).astype('float')
band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8

parser = argparse.ArgumentParser(description='Train a PLC model')

parser.add_argument('features_logE', metavar='<features file>', help='binary features file containing log-Energies(float32)')
parser.add_argument('features_mdct', metavar='<features file>', help='binary features file containing normalized mdcts (float32)')
parser.add_argument('lost_file', metavar='<packet loss file>', help='packet loss traces (int8)')
parser.add_argument('output', metavar='<output>', help='trained model file (.h5)')
parser.add_argument('--model', metavar='<model>', default='opus_plc_network', help='PLC model python definition (without .py)')
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
parser.add_argument('--logdir', metavar='<log dir>', help='directory for tensorboard log files')
# parser.add_argument('--band-loss', metavar='<weight>', default=1.0, type=float, help='weight of band loss (default 1.0)')
# parser.add_argument('--loss-bias', metavar='<bias>', default=0.0, type=float, help='loss bias towards low energy (default 0.0)')


args = parser.parse_args()

import importlib
opus_plc = importlib.import_module(args.model)

import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import h5py

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
import tensorflow_io as tfio
from soft_smooth import *
#if gpus:
#  try:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#  except RuntimeError as e:
#    print(e)

# Import custom loader and loss functions
import opus_plc_loader
import opus_plc_lossfuncs

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
    lr = 1.0e-3
    decay = 0

if args.lr is not None:
    lr = args.lr

if args.decay is not None:
    decay = args.decay

if retrain:
    input_model = args.retrain


opt = Adam(lr, decay=decay, beta_2=0.99)

# Energy PLC Model
# model = opus_plc.plc_energy(rnn_units=args.gru_size, batch_size=batch_size, training=True, quantize=quantize, cond_size=args.cond_size)
# model.compile(optimizer=opt, loss=opus_plc_lossfuncs.plc_energy_log_L1()) # Jan Spectral Shaping Loss (MSE)

# Shape PLC Model
model = opus_plc.plc_shape(nb_cepstral_features = int(band_defs[-1] - band_defs[0]), batch_size=batch_size, band_defs = band_defs)
model.compile(optimizer=opt, loss=opus_plc_lossfuncs.plc_mdctshape_loss_bandwise(0.0,band_defs)) # Jan Spectral Shaping Loss (MSE)

model.summary()

# Fraction of data to use for training
frac_data = 0.4

feature_file_logE = args.features_logE
feature_file_mdct = args.features_mdct
nb_features = 21

lost = np.memmap(args.lost_file, dtype='int8', mode='r')

features_E = np.memmap(feature_file_logE, dtype='float32', mode='r')
features_mdct = np.memmap(feature_file_mdct, dtype='float32', mode='r')

sequence_size_shape = args.seq_length
nb_features_cepstral = 960
nb_sequences_shape = len(features_mdct)//(nb_features_cepstral*sequence_size_shape)//batch_size*batch_size
features_mdct = features_mdct[:nb_sequences_shape*sequence_size_shape*nb_features_cepstral]
features_mdct = np.reshape(features_mdct, (nb_sequences_shape, sequence_size_shape, nb_features_cepstral))
features_mdct = features_mdct[:(int)(frac_data*nb_sequences_shape), :, band_defs[0]:band_defs[-1]] # CNN

nb_sequences_E = len(features_E)//(nb_features*sequence_size_shape)//batch_size*batch_size
features_E = features_E[:nb_sequences_E*sequence_size_shape*nb_features]
features_E = np.reshape(features_E, (nb_sequences_E, sequence_size_shape, nb_features))
features_E = features_E[:(int)(frac_data*nb_sequences_E), :, :]

# Smoothing the above features for shape PLC
features_mdct_smooth = shape_spectrum((features_E + ebandMeans[:21]),features_mdct,2,8)

checkpoint = ModelCheckpoint('{}_{}_{}.h5'.format(args.output, args.gru_size, '{epoch:02d}'))

if args.retrain is not None:
    model.load_weights(args.retrain)

if quantize or retrain:
    #Adapting from an existing model
    model.load_weights(input_model)

model.save_weights('{}_{}_initial.h5'.format(args.output, args.gru_size))

# Energy PLC Network Loader
# loader = opus_plc_loader.PLCLoader_energy(features_E, lost, batch_size)

# Shape PLC Network Loader
loader = opus_plc_loader.PLCLoader_shape(features_mdct_smooth,features_E,sequence_size_shape,batch_size,0,lost)

csv_logger = CSVLogger('train_loss_shape_PLC.csv', append=False, separator=',')
callbacks = [checkpoint, csv_logger]
if args.logdir is not None:
    logdir = '{}/{}_{}_logs'.format(args.logdir, args.output, args.gru_size)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks.append(tensorboard_callback)

model.fit(loader, epochs=nb_epochs, validation_split=0.0, callbacks=callbacks)