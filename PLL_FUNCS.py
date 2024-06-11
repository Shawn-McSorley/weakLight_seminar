import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import integrate
import shutil
import os
from scipy.fft import fft, fftfreq
from numba import jit

def phaseToIQ(data, tVals, f0, A, unit = 'cycles'):
    ''' Function to convert phase data to IQ data. f0 is the carrier frequency. Default unit for phase data is cycles.'''
    if unit == 'cycles':
        scale = 2*np.pi
    else:
        scale = 1
    IQ = A * np.exp(1j * (2*np.pi*f0*tVals + scale * data))
    return IQ

def PLL_TF(paramPI = {'P':0.2, 'I':0.05, 'I2' : 0.005}, fs = 1):
    ''' Function to calculate the transfer function of a PLL. '''
    P = paramPI['P']
    I = paramPI['I']
    I2 = paramPI['I2']
    
    # Phase out = phase_in * (P + I/(1-z^-1) + I2/(1-z^-1)^2)
    # Transfer function = (P + I/(1-z^-1) + I2/(1-z^-1)^2)
    f = np.linspace(0.00001, fs/2, 100000)
    z = np.exp(2j*np.pi*f/fs)
    den = 1 - z**-1

    ## PII2 Controller
    PII2_TF = P + I/den + I2/den**2
    NCO = 2*np.pi/den

    LOOP_GAIN = PII2_TF * NCO
    FORWARD_LOOP_TF = PII2_TF * NCO/ (1 + LOOP_GAIN) # Phase_out / Phase_in (1 is ideal phase lock)
    ERROR_TF = 1 / (1 + LOOP_GAIN) # Error / Phase_in (0 is ideal phase lock)

    ## Controller Tuning Functions

    P_GAIN = P * NCO
    I_GAIN = I/den * NCO
    I2_GAIN = (I2/den**2) * NCO


    return f, P_GAIN, I_GAIN, I2_GAIN, LOOP_GAIN, FORWARD_LOOP_TF, ERROR_TF

@jit(nopython=True)
def FUNC_DELAY(timeseries, time_delay, fs):
    ''' Function to delay a time series f(t) of length N by the instantaneous time delay T(t) of length N (i.e. f(t-T(t))). '''
    N = len(timeseries)
    t = np.arange(N)/fs
    delayed_timeseries = np.zeros(N)  # Create a new array to store the delayed time series
    
    for i in range(N):
        if time_delay[i] > 0:
            t_delayed = t[i] - time_delay[i]
            if t_delayed < 0 or t_delayed > t[-1]:  # Handle boundary conditions
                delayed_timeseries[i] = 0
            else:
                delayed_timeseries[i] = np.interp(t_delayed, t, timeseries)
        else:
            delayed_timeseries[i] = timeseries[i]
    
    return delayed_timeseries
    
@jit(nopython=True)
def PLL(dataIQ, P, I, I2):
    ''' Function for Phase Locked Loop (PLL) for IQ data (in complex form, i.e. I+jQ) '''
    N = len(dataIQ)
    adjustment = np.zeros(N, dtype=np.complex64)
    phase_error = np.zeros(N)
    freq_error = np.zeros(N)
    phase = 0
    sum_error = 0
    sum_sum_error = 0  
    for i in range(N):
        adjustment[i] = dataIQ[i] * np.exp(-1j*phase)
        error = np.imag(adjustment[i])
        sum_error += error
        sum_sum_error += sum_error
        freq_error[i]  = P * error + I * sum_error + I2 * sum_sum_error
        phase += 2*np.pi*freq_error[i]
        phase_error[i] = phase/(2*np.pi)
    return adjustment, phase_error


    
def fftnoise(f):
    ''' Function to generate noise with a given frequency spectrum. '''
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    ''' Function to generate band-limited noise. Parameters: min_freq, max_freq, samples, samplerate. '''
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

# Code to run if main
if __name__ == '__main__':

    # Test the delay function
    fs = 10000  # Sampling frequency
    timeseries = np.sin(2 * np.pi * 0.5 * np.arange(fs) / 100)  # Example sine wave
    time_delay = 0.01 * np.sin(2 * np.pi * 0.1 * np.arange(fs) / 100)  # Example delay
    

    delayed_timeseries = FUNC_DELAY(timeseries, -time_delay, fs)
    plt.figure()
    plt.plot(timeseries, label='Original')
    plt.plot(delayed_timeseries, label='Delayed')
    plt.show()
    # # Test the PLL TF function
    # P = 2**-8
    # I = 2**-17
    # I2 = 2**-30
    # fs = 250e6
    # f, P_GAIN, I_GAIN, I2_GAIN, LOOP_GAIN, FORWARD_LOOP_TF, ERROR_TF = PLL_TF({'P':P, 'I':I, 'I2':I2}, fs)
    
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(P_GAIN)), label = 'P')
    # ax0.plot(f, 20*np.log10(np.abs(I_GAIN)), label = 'I')
    # ax0.plot(f, 20*np.log10(np.abs(I2_GAIN)), label = 'I2')

    # unity_gain_freq = f[np.argmin(np.abs(np.abs(P_GAIN) - 1))]
    # ax0.axvline(unity_gain_freq, color='r', linestyle='--')

    # ax0.legend()
    # ax0.set_title('LOOP GAINs Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax0.grid()
    # ax1.plot(f, np.angle(P_GAIN))
    # ax1.plot(f, np.angle(I_GAIN))
    # ax1.plot(f, np.angle(I2_GAIN))
    # ax1.set_title('LOOP GAIN Phase Response')
    

    # # Plot LOOP GAIN magnitude and phase, and show unity gain frequency and phase margin
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(LOOP_GAIN)))
    # ax0.set_title('LOOP GAIN Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax1.plot(f, np.angle(LOOP_GAIN))
    # ax1.set_title('LOOP GAIN Phase Response')
    
    # # Find unity gain frequency and phase margin
    # unity_gain_freq = f[np.argmin(np.abs(np.abs(LOOP_GAIN) - 1))]
    # phase_margin = np.angle(LOOP_GAIN[np.argmin(np.abs(f - unity_gain_freq))])
    # # Draw unity gain frequency line and phase margin line
    # ax0.axvline(unity_gain_freq, color='r', linestyle='--')
    # ax1.axvline(unity_gain_freq, color='r', linestyle='--')
    # ax1.axhline(phase_margin, color='r', linestyle='--')
    # ax1.axhline(- np.pi, color='r', linestyle='--')
    # # print phase margin in degrees
    # print('Unity Gain Frequency: ', unity_gain_freq)
    # print('Phase Margin: ', 180 + phase_margin*180/np.pi)

    # # Plot Magnitude and Phase response in subplots vertically
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(FORWARD_LOOP_TF)))
    # ax0.set_title('Forward Loop Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax1.plot(f, np.angle(FORWARD_LOOP_TF))
    # ax1.set_title('Phase Response')
    # ax1.set_xscale('log')
    # ax1.set_yscale('linear')
    # ax1.set_xlabel('Frequency (Hz)')
    # ax0.grid()
    # ax1.grid()

    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(ERROR_TF)))
    # ax0.set_title('Error Loop Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax1.plot(f, np.angle(ERROR_TF))
    # ax1.set_title('Phase Response')
    # ax1.set_xscale('log')
    # ax1.set_yscale('linear')
    # ax1.set_xlabel('Frequency (Hz)')
    # ax0.grid()
    # ax1.grid()

    # # plot the error response

    # plt.show()

