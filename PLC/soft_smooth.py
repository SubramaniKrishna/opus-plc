"""
Code to smooth the energy vector and the mdct-normalizing coefficients
Energy inputs of size [batch,seq,21] and Shape inputs of size [batch,seq,800]
"""

import numpy as np

ebandMeans = np.array([
      6.437500, 6.250000, 5.750000, 5.312500, 5.062500,
      4.812500, 4.500000, 4.375000, 4.875000, 4.687500,
      4.562500, 4.437500, 4.875000, 4.625000, 4.312500,
      4.500000, 4.375000, 4.625000, 4.750000, 4.437500,
      3.750000, 3.750000, 3.750000, 3.750000, 3.750000
]).astype('float')
band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8

def smooth_vector(vec, k, lowpass='pyramid'):
    """ applies smoothing with rect kernel of size 2k + 1 """
    
    if lowpass == 'rect':
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
    elif lowpass == 'pyramid':
        kernel = np.concatenate((np.arange(1, k+2), np.arange(k, 0, -1)))
        kernel = kernel / np.sum(kernel)
    
    # extend vector by repetition
    extended_vec = np.concatenate(
        (
            vec[0] * np.ones(k),
            vec,
            vec[-1] * np.ones(k)
        ))
    
    smoothed_vec = np.convolve(extended_vec, kernel, mode='valid')
    
    return smoothed_vec

def denormalize_mdct(input_norm_log, input_mdct_normalized):
    input_norm_linear = 2**(input_norm_log)

    mdct = np.copy(input_mdct_normalized)
    
    for i in range(band_defs.shape[0] - 1):
        mdct[:,:, band_defs[i] : band_defs[i + 1]] *= input_norm_linear[:,:,i:i+1]
    # mdct[:,:, band_defs[-1]:] *= np.tile(input_norm_linear[:,:,-1:],[1,1,160])

    return mdct

def normalize_mdct(input_norm_log, mdct):
    input_norm_linear = 2**(input_norm_log)

    mdct_normalized = np.copy(mdct)
    
    for i in range(band_defs.shape[0] - 1):
        mdct_normalized[:,:, band_defs[i] : band_defs[i + 1]] /= input_norm_linear[:,:,i:i+1]
    # mdct_normalized[:,:, band_defs[-1]:] /= np.tile(input_norm_linear[:,:,-1:],[1,1,160])

    return mdct_normalized

def get_scale_factors(input_norm_log,k1 = 2,k2 = 8):
    """
    Add ebandmeans to input 
    """

    input_energy_linear = 2**(2*(input_norm_log))
    input_energy_log = np.log(input_energy_linear)

    input_energy_log_smoothed = np.apply_along_axis(lambda inp: smooth_vector(inp, k1, 'pyramid'), axis=-1, arr=input_energy_log)
    
    smoothing_coefficients = np.zeros((input_norm_log.shape[0],input_norm_log.shape[1],800))
    for i in range(band_defs.shape[0] - 1):
        smoothing_coefficients[:,:, band_defs[i] : band_defs[i + 1]] = np.tile(input_energy_log_smoothed[:,:,i:i+1],[1,1,band_defs[i+1] - band_defs[i]])
    
    scale_factors = np.apply_along_axis(lambda inp: smooth_vector(inp, k2, 'pyramid'), axis=-1, arr=smoothing_coefficients)

    return (np.exp(scale_factors/2) + 1e-8)

def shape_spectrum(input_norm_log, input_mdct_normalized,k1 = 2, k2 = 8):

    mdct = denormalize_mdct(input_norm_log, input_mdct_normalized)
    scale_factors = get_scale_factors(input_norm_log,k1,k2)

    return mdct / scale_factors

def invert_shape_spectrum(input_norm_log, input_mdct_shaped,k1 = 2, k2 = 8):
    scale_factors = get_scale_factors(input_norm_log,k1,k2)
    input_mdct_unnormalized = input_mdct_shaped * scale_factors
    input_mdct_normalized = normalize_mdct(input_norm_log, input_mdct_unnormalized)

    return input_mdct_normalized





