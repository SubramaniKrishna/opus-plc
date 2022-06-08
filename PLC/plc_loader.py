import numpy as np
from tensorflow.keras.utils import Sequence

class PLCLoader(Sequence):
    def __init__(self, features, lost, nb_burg_features, batch_size):
        self.batch_size = batch_size
        self.nb_batches = features.shape[0]//self.batch_size
        self.features = features[:self.nb_batches*self.batch_size, :, :]
        self.lost = lost.astype('float')
        self.lost = self.lost[:(len(self.lost)//features.shape[1]-1)*features.shape[1]]
        self.nb_burg_features = nb_burg_features
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches*self.batch_size)
        np.random.shuffle(self.indices)
        offset = np.random.randint(0, high=self.features.shape[1])
        self.lost_offset = np.reshape(self.lost[offset:-self.features.shape[1]+offset], (-1, self.features.shape[1]))
        self.lost_indices = np.random.randint(0, high=self.lost_offset.shape[0], size=self.nb_batches*self.batch_size)

    def __getitem__(self, index):
        features = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        #lost = (np.random.rand(features.shape[0], features.shape[1]) > .2).astype('float')
        lost = self.lost_offset[self.lost_indices[index*self.batch_size:(index+1)*self.batch_size], :]
        lost = np.reshape(lost, (features.shape[0], features.shape[1], 1))
        lost_mask = np.tile(lost, (1,1,features.shape[2]))
        in_features = features*lost_mask
        
        # For the first frame after a loss, we don't have valid features, but the Burg estimate is valid.
        # in_features[:,1:,self.nb_burg_features:] = in_features[:,1:,self.nb_burg_features:]*lost_mask[:,:-1,self.nb_burg_features:]
        out_lost = np.copy(lost)
        out_lost[:,1:,:] = out_lost[:,1:,:]*out_lost[:,:-1,:]

        out_features = np.concatenate([features[:,:,self.nb_burg_features:], 1.-out_lost], axis=-1)
        inputs = [in_features*lost_mask, lost]
        outputs = [out_features]
        return (inputs, outputs)

    def __len__(self):
        return self.nb_batches

# class PLCLoader(Sequence):
#     def __init__(self, features, lost, nb_burg_features, batch_size):
#         self.batch_size = batch_size
#         self.nb_batches = features.shape[0]//self.batch_size
#         self.features = features[:self.nb_batches*self.batch_size, :, :]
#         self.lost = lost[:self.nb_batches*self.batch_size,:,:].astype('float')
#         # self.lost = self.lost[:(len(self.lost)//features.shape[1]-1)*features.shape[1]]
#         self.nb_burg_features = nb_burg_features
#         self.on_epoch_end()

#     def on_epoch_end(self):
#         self.indices = np.arange(self.nb_batches*self.batch_size)
#         np.random.shuffle(self.indices)
#         # offset = np.random.randint(0, high=self.features.shape[1])
#         # self.lost_offset = np.reshape(self.lost[offset:-self.features.shape[1]+offset], (-1, self.features.shape[1]))
#         # self.lost_indices = np.random.randint(0, high=self.lost_offset.shape[0], size=self.nb_batches*self.batch_size)

#     def __getitem__(self, index):
#         features = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
#         lost = self.lost[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
#         #lost = (np.random.rand(features.shape[0], features.shape[1]) > .2).astype('float')
#         # lost = self.lost_offset[self.lost_indices[index*self.batch_size:(index+1)*self.batch_size], :]
#         # lost = np.reshape(lost, (features.shape[0], features.shape[1], 1))
#         # lost_mask = np.tile(lost, (1,1,features.shape[2]))
#         # in_features = features*lost_mask
        
#         #For the first frame after a loss, we don't have valid features, but the Burg estimate is valid.
#         # in_features[:,1:,self.nb_burg_features:] = in_features[:,1:,self.nb_burg_features:]*lost_mask[:,:-1,self.nb_burg_features:]
#         # out_lost = np.copy(lost)
#         # out_lost[:,1:,:] = out_lost[:,1:,:]*out_lost[:,:-1,:]

#         # features = np.concatenate([features[:,:,self.nb_burg_features:], 1.-lost], axis=-1)
#         inputs = [features, lost]
#         outputs = [np.concatenate([features[:,:,self.nb_burg_features:], 1.-lost], axis=-1)]
#         return (inputs, outputs)

#     def __len__(self):
#         return self.nb_batches