"""
Performing MDCT Analysis of Input
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

# Load Audio
x = np.fromfile("/Users/krisub/Desktop/Datasets/comp_mono48.raw",dtype = np.short)/32768.0

# DC Rejection and Pre-emphasis
c = (6.3*3/48000)
x = lfilter([1,-1],[1,-(1-c)],x) # DC Reject
x = lfilter([1,-0.85],[1],x)

# Shift by 420 to account for 2N windowing
mdct_x = mdct_ola(x[420:]*32768.0,N,H)
tD,mdct_normalized = mdct2bands(mdct_x,band_defs)
log_tD = np.log2(tD) - ebandMeans[:21]

# Save log Energies and Normalized MDCTs
np.save("./comp_mono48_logE.npy",log_tD)
np.save("./comp_mono48_normalized_mdct.npy",mdct_normalized)





