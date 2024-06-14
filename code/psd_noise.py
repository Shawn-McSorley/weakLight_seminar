import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def fftnoise(f):
    ''' Function to generate noise with a given frequency spectrum. '''
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1, psd_lvl=1):
    ''' Function to generate band-limited noise. Parameters: min_freq, max_freq, samples, samplerate. '''
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples, dtype='complex')
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    T = samples/samplerate
    f[idx] = np.sqrt(psd_lvl * T / 2) * (np.cos(2*np.pi*np.random.rand(len(idx))) + 1j * np.sin(2*np.pi*np.random.rand(len(idx))))
    return fftnoise(f)

# Example usage
fs = 100000
N = fs * 1  # 1 second of data
min_freq = 0
max_freq = fs / 2
psd_lvl = 1e-3

noise = band_limited_noise(min_freq, max_freq, N, fs, psd_lvl)

f, Pxx = signal.welch(noise, fs=fs, nperseg=N//8)
plt.figure()
plt.plot(f, Pxx, label='Welch')
plt.axhline(psd_lvl, color='r', linestyle='--', label='True PSD Level')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.title('Power spectral density of generated noise')
plt.xscale('log')
plt.yscale('log')
plt.show()
