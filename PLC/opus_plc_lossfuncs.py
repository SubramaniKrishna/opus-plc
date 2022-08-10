import tensorflow.keras.backend as K

def plc_energy_log_L1():
    def loss(y_true,y_pred):
        mask = y_true[:,:,-1:]
        y_true = y_true[:,:,:-1]
        e = (y_pred - y_true)*mask
        l1_loss = K.mean(K.abs(e))

        return l1_loss
        # e = (y_pred - y_true)
        # rms_loss = K.sqrt(K.sum(K.square(e),axis = -1))
        # return rms_loss
    return loss

def plc_mdctshape_loss_bandwise(alpha,band_defs):
    def loss(y_true,y_pred):
        # mask = y_true[:,-1:,:]
        # y_true = y_true[:,:-1,:]
        e = (y_pred - y_true)
        abs_e = (K.abs(y_pred) - K.abs(y_true))
        rms_loss = 0
        abs_rms = 0
        # print(y_pred.shape)
        # bw = np.flip(np.arange(1,band_defs.shape[0]))
        # bw = bw/bw.sum()
        for i in range(band_defs.shape[0] - 1):
            rms_loss+= K.sqrt(K.sum(K.square(e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1))
            # rms_loss+= K.sum(K.square(e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1)
            abs_rms+= K.sqrt(K.sum(K.square(abs_e[:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0],:]),axis = 1))
        # rms_loss = K.tanh(K.mean(rms_loss)/2)
        rms_loss = K.mean(rms_loss)
        abs_rms = K.mean(abs_rms)
        return ((1 - alpha)*rms_loss + alpha*abs_rms)
    return loss

def plc_mdctshape_nll():
    def loss(y_true,y_pred):
        mask = y_true[:,-1:,:]
        y_true = y_true[:,:-1,:]
        mu_pred = y_pred[:,:,:,0]
        log_sigma_pred = y_pred[:,:,:,1]
        sigma_pred = K.exp(log_sigma_pred)
        # nll = 0
        # for i in range(band_defs.shape[0] - 1):
        nll = K.sum(2*0.5*log_sigma_pred,axis = 1) + K.sum(((0.5*K.square(mu_pred - y_true))/K.square(sigma_pred)),axis = 1)
        # nll += K.sum(0.5*log_sigma_pred,axis = 1)
        return K.mean(nll)
    return loss