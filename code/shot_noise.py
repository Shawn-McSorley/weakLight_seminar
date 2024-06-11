import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def shot_noise(P, waveLength, T, fs, eta = 1):
    ''' This function simulates the Poisson statistics of an optical detector. 
        P: Optical power in dBm. Make sure it's low.
        waveLength: Wavelength in meters
        T: Observation time in seconds
        fs: Sampling frequency in Hz
        eta: Quantum efficiency of the detector
        I think this function should be called repeatedly to simulate multiple observations.
    '''
    # Constants
    h = 6.62607015e-34
    c = 299792458
    q = 1.60217662e-19
    f = c/waveLength
    # Convert power to Watts
    P = 10**(P/10)/1000
    print(f'Power: {P*1e15} fW')
    # Calculate the number of photons
    flux = P/(h*f)
    N = int(flux*T*eta) # Mean number of photo-electrons generated
    
    # Simulate a shot on the detector as a dirac delta function multiplied by q. The distribution of shots is Poisson.
    size = int(T*fs)
    if(N > size):
        print('Error: Number of shots is greater than the size of the array')
        return 0, 0
    shots = np.zeros(size)
    t = np.linspace(0, T, size)
    num_shots = np.random.poisson(N)
    shot_times = np.random.randint(0, size, num_shots)
    shots[shot_times] = q
    I = (q/T)*N
    SI = 2*q*I*np.ones(size)
    return t, shots, SI
    

# Example
P = -120
waveLength = 1550e-9

fs = 250e6
T = 0.1
t, shots, qI = shot_noise(P, waveLength, T, fs)

plt.figure()
plt.step(t, shots)
plt.show()

# Calculate the autocorrelation of the shot noise. As density
f, Pxx = welch(shots, fs, nperseg=1024) 
plt.figure()
plt.semilogy(f, Pxx*fs**2)
plt.plot(f, qI[0]*np.ones(len(f)), 'r')
plt.show()

## Good enough for now.