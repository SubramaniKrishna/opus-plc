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

class PLCLoader_shape(Sequence):
    # Expects concatenated energy and unit vectors as features
    def __init__(self, features_mdct,features_E, seq_size,batch_size,choice = 0):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.nb_batches = features_mdct.shape[0]//self.batch_size
        self.features_mdct = features_mdct[:self.nb_batches*self.batch_size, :, :]
        self.features_E = features_E[:self.nb_batches*self.batch_size, :, :]
        self.on_epoch_end()
        self.choice = choice

    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches*self.batch_size)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        features_mdct = self.features_mdct[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        features_E = self.features_E[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        # features = np.concatenate((features_mdct,features_E),axis = -1)
        # inputs = features_mdct[:,:2,:].squeeze()
        # inputs = inputs.reshape(self.batch_size,2*inputs.shape[-1]) #Feedforward
        # inputs = inputs.reshape(self.batch_size,1,2,inputs.shape[-1]) #CNN
        # outputs = features_mdct[:,2:,:].squeeze()
        # print(inputs.shape,outputs.shape)
        # inputs = features_mdct.reshape(self.batch_size,features_mdct.shape[-1],self.seq_size,1) #CNN
        inputs = np.expand_dims(np.transpose(features_mdct,[0,2,1]),-1)
        outputs = np.roll(inputs.squeeze(),-2,-1)
        # Augment 0,1 for odd/even mdcts as "positional encodings"
        # zarr = (np.arange(inputs.shape[1])%2).astype('float')
        # zarr[zarr == 0] = -1
        # zarr = np.stack([zarr for i in range(self.seq_size)])
        # zarr = zarr.transpose([1,0])
        # zarr = np.stack([zarr for i in range(self.batch_size)],0)
        # zarr = np.expand_dims(zarr,-1)
        # inputs = np.concatenate([inputs,zarr],-1)
        # inputs = inputs*zarr
        # outputs = inputs.squeeze()
        # print(inputs.shape,outputs.shape)
        if self.choice == 0:
            return (inputs, outputs)
        else:
            # outputs = np.concatenate([features_E,np.transpose(outputs,[0,2,1])],-1)
            # print(inputs.shape,outputs.shape)
            return (inputs, features_E, outputs)

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