"""
Helper functions to compute MDCT/IMDCT
"""

import numpy as np
from scipy.fftpack import dct,idct

def mdct_frame(x):
    """
    Computes MDCT of 20 ms frame (at 48000Hz) using Vorbis Flat-top Window (OPUS specs)
    Effective frame size is 40ms (but because of flat-top only 960 "new" samples)
    """
    N = x.shape[0]
    I = np.eye(N//4)
    J = np.fliplr(I)
    Z = np.zeros((N//4,N//4))
    Q = np.concatenate((np.concatenate((Z,Z,-J,-I),1),np.concatenate((I,-J,Z,Z),1)),0)
    
    # 120 Sample Vorbis Window
    Nw = 120
    vw = np.sin((np.pi/2)*(np.sin((np.pi/(2*Nw))*(np.arange(Nw) + 0.5)))**2)
    # vw = np.ones(Nw)
    w = np.concatenate((np.zeros(420),vw,np.ones(840),np.flip(vw),np.zeros(420)))

    x = w*x
    mdct = dct(Q@x, type = 4, norm =None)*(2.0/N)
    
    return mdct

def imdct_frame(x):
    """
    Computes i-MDCT (inverse of above function) of 20 ms frame (at 48000Hz)
    """
    N = x.shape[0]*2
    I = np.eye(N//4)
    J = np.fliplr(I)
    Z = np.zeros((N//4,N//4))
    Q = np.concatenate((np.concatenate((Z,Z,-J,-I),1),np.concatenate((I,-J,Z,Z),1)),0)
    
    # 120 Sample Vorbis Window
    Nw = 120
    vw = np.sin((np.pi/2)*(np.sin((np.pi/(2*Nw))*(np.arange(Nw) + 0.5)))**2)
    w = np.concatenate((np.zeros(420),vw,np.ones(840),np.flip(vw),np.zeros(420)))
    
    imdct = Q.T@idct(x, type = 4, norm = None)*(0.5)
    
    return w*imdct

def mdct_ola(x,N,H):
    """
    Short-Time windowed MDCT for input audio
    """
    l = []
    # xzp = np.concatenate((x,np.zeros(N//2)),0)
    for i in range(0,x.shape[0] - N,H):
        l.append(mdct_frame(x[i:i+N]))
    
    return np.stack(l,0)

def imdct_ola(x,N,H,Nog):
    """
    Overlap-add reconstruction from short-time windowed MDCT
    """
    x_tilde = np.zeros(Nog + N)
    
    for i in range(x.shape[0]):
        x_tilde[i*H:i*H+N] = x_tilde[i*H:i*H+N] + imdct_frame(x[i,:])
    
    return x_tilde[N//2:Nog + N//2]

def mdct2bands(mdct,band_defs):
    """
    Split MDCT into perceptual bands and return energy and normalized MDCT
    (zero out coefficients 800:960 in accordance with CELT)
    """
    mdct_norm = np.copy(mdct)
    list_E = []

    for i in range(band_defs.shape[0] - 1):
        list_E.append(np.linalg.norm(mdct[:,band_defs[i]:band_defs[i+1]],axis = 1))
        mdct_norm[:,band_defs[i]:band_defs[i+1]] = mdct_norm[:,band_defs[i]:band_defs[i+1]]/list_E[-1][:,None]
    
    mdct_norm[:,800:] = 0
    list_E = np.stack(list_E,axis = 1)
    
    return list_E,mdct_norm

def mdctbands2mdct(mdct_bands,list_E,band_defs):
    """
    Given energies and normalized MDCT return the original un-normalized MDCT coefficients
    (zero out coefficients 800:960 in accordance with CELT)
    """
    mdct_bands_unnormalized = np.copy(mdct_bands)

    for i in range(band_defs.shape[0] - 1):
        mdct_bands_unnormalized[:,band_defs[i]:band_defs[i+1]] = mdct_bands_unnormalized[:,band_defs[i]:band_defs[i+1]]*list_E[:,i][:,None]
    mdct_bands_unnormalized[:,800:] = 0
    return mdct_bands_unnormalized

def mdctbands2mdct_withplc(mdct_bands,list_E,band_defs,loss_trace,choice = 0):
    """
    Given energies, normalized MDCTs and trace file for lost packets, replace lost packets with previous packet mdcts
    If choice = 0, use previous packet mdct
    If choice = 1, randomly generate a unit vector for the mdct
    If choice = 2, reconstruct using previous frame energy but current mdct
    """
    mdct_bands_unnormalized = np.copy(mdct_bands)
    listE = np.copy(list_E)

    for i in range(band_defs.shape[0] - 1):
        mdct_bands_unnormalized[:,band_defs[i]:band_defs[i+1]] = mdct_bands_unnormalized[:,band_defs[i]:band_defs[i+1]]*list_E[:,i][:,None]
    
    inds =  np.where(loss_trace == 0)[0]
    if inds.shape[0]!=0:
        inds = np.delete(inds,0)
    
    # Prevent cheating for consecutive zeros with ground truth
    for i in range(1,loss_trace.shape[0]):
        if loss_trace[i] == 1:
            continue
        else:
            listE[i,:] = listE[i - 1,:]

    if choice == 0:
        mdct_bands_unnormalized[inds,:] = mdct_bands_unnormalized[inds - 1,:]
    elif choice == 1:
        for i in range(band_defs.shape[0] - 1):
            rV = np.random.randn(mdct_bands_unnormalized.shape[0],band_defs[i+1] - band_defs[i])
            rV = rV/np.linalg.norm(rV,axis = 0)
            mdct_bands_unnormalized[inds,band_defs[i]:band_defs[i+1]] = rV[inds,:]*listE[inds,i][:,None]
    else:
        for i in range(band_defs.shape[0] - 1):
            mdct_bands_unnormalized[inds,band_defs[i]:band_defs[i+1]] = mdct_bands[inds,band_defs[i]:band_defs[i+1]]*listE[inds,i][:,None]

    return mdct_bands_unnormalized

def mdctbands2mdct_with_neuralplc(mdct_bands,list_E,list_E_neural,band_defs,loss_trace, choice = 0):
    """
    Given energies, normalized MDCTs and trace file for lost packets, along with corresponding 
    If choice = 0, use current packet mdct
    If choice = 1, randomly generate a unit vector for the mdct
    """
    mdct_bands_unnormalized = np.copy(mdct_bands)

    for i in range(band_defs.shape[0] - 1):
        mdct_bands_unnormalized[:,band_defs[i]:band_defs[i+1]] = mdct_bands_unnormalized[:,band_defs[i]:band_defs[i+1]]*list_E[:,i][:,None]
    
    inds =  np.where(loss_trace == 0)[0]
    if inds.shape[0]!=0:
        inds = np.delete(inds,0)

    if choice == 0:
        for i in range(band_defs.shape[0] - 1):
            mdct_bands_unnormalized[inds,band_defs[i]:band_defs[i+1]] = mdct_bands[inds,band_defs[i]:band_defs[i+1]]*list_E_neural[inds,i][:,None]
    else:
        for i in range(band_defs.shape[0] - 1):
            rV = np.random.randn(mdct_bands_unnormalized.shape[0],band_defs[i+1] - band_defs[i])
            rV = rV/np.linalg.norm(rV,axis = 0)
            mdct_bands_unnormalized[inds,band_defs[i]:band_defs[i+1]] = rV[inds,:]*list_E_neural[inds,i][:,None]

    return mdct_bands_unnormalized