"""
Supports batch processing
"""
import tensorflow as tf
import numpy as np

def tf_imdct_frame(x):
    N = 960*2
    I = tf.eye(N//4)
    J = tf.reverse(I,[-1])
    Z = tf.zeros((N//4,N//4))
    Q = tf.concat((tf.concat((Z,Z,-J,-I),1),tf.concat((I,-J,Z,Z),1)),0)
    Qt = tf.cast(tf.transpose(Q,[1,0]),'float32')
    
    # 120 Sample Vorbis Window
    Nw = 120
    vw = tf.sin((np.pi/2)*(tf.sin((np.pi/(2*Nw))*(tf.range(Nw,dtype = 'float') + 0.5)))**2)
    w = tf.concat((tf.zeros(420),vw,tf.ones(840),tf.reverse(vw,[-1]),tf.zeros(420)),-1)
    w = tf.cast(w,'float32')
    # dct4 = tf.signal.dct(tf.concat((x,tf.zeros((16,N))),-1), type = 2, norm = None)[:,1::2]
    dct4 = tf.signal.dct(x, type = 2, norm = None)
    imdct = tf.matmul(Qt,tf.transpose(dct4,[1,0]))*0.5
    imdct = imdct*tf.expand_dims(w,-1)
    return tf.transpose(imdct,[1,0])

def tf_mdctbands2mdct(mdct_bands,list_E,band_defs):
    """
    Given energies and normalized MDCT return the original un-normalized MDCT coefficients
    (zero out coefficients 800:960 in accordance with CELT)
    """
    
    l = []
    for i in range(band_defs.shape[0] - 1):
        E = tf.expand_dims(list_E[:,:,i],-1)
        l.append(E*mdct_bands[:,:,band_defs[i] - band_defs[0]:band_defs[i+1] - band_defs[0]])
    M = tf.concat(l,-1)
    M = tf.concat((M,tf.zeros((16,100,960 - band_defs[-1]),dtype = M.dtype)),-1)
    return M

def tf_imdct_ola(x,list_E,N,H,Nog,band_defs,ebandMeans):
    """
    Overlap-add reconstruction from short-time windowed MDCT
    """
    x = tf.squeeze(x)
    x = tf.transpose(x,[0,2,1])
    
    band_E = 2**(tf.squeeze(list_E) + ebandMeans[:21])
    # Un-normalize mdct with available energies
    x = tf_mdctbands2mdct(x,band_E,band_defs)
    l = []
    for i in range(x.shape[1]):
        l.append(tf_imdct_frame(x[:,i,:]))
    
    M = tf.stack(l,-1)
    M = tf.transpose(M,[0,2,1])
    M = tf.squeeze(tf.signal.overlap_and_add(M,960))
    return M