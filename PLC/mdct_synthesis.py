"""
Performing MDCT Synthesis from log Energies and Unnormalized coefficients
"""

from mdct_funcs import *
from scipy.io import wavfile
from scipy.signal import lfilter


# MDCT Parameters
N = 960*2
H = 480*2
band_defs = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100])*8
ebandMeans = np.array([
      6.437500, 6.250000, 5.750000, 5.312500, 5.062500,
      4.812500, 4.500000, 4.375000, 4.875000, 4.687500,
      4.562500, 4.437500, 4.875000, 4.625000, 4.312500,
      4.500000, 4.375000, 4.625000, 4.750000, 4.437500,
      3.750000, 3.750000, 3.750000, 3.750000, 3.750000
])

# Load log-Energies and unnormalized coefficients
log_tD = np.load("./comp_mono48_logE.npy")
mdct_normalized = np.load("./comp_mono48_normalized_mdct.npy")

# Go from log-E to E
band_E_recon = 2**(log_tD + ebandMeans[:21])
mdct_recon_x = mdctbands2mdct(mdct_normalized,band_E_recon,band_defs)
recon_x = imdct_ola(mdct_recon_x,N,H,log_tD.shape[0]*N)

# Invert DC Rejection and Pre-emphasis
c = (6.3*3/48000)
recon_x = recon_x/32768.0
recon_x = lfilter([1],[1,-0.85],recon_x)
recon_x = lfilter([1,-(1-c)],[1,-1],recon_x) # DC Reject

wavfile.write("./comp_mono48_recon.wav",48000,(recon_x*32768.0).astype(np.short))





