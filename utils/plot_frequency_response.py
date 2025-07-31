
import torch 
import torchaudio
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq

# def calculate_frequency_response(freqs, position):
#     """
#     Calculate the frequency response of the wah-wah effect at a given position.
    
#     Args:
#         freqs: Array of frequencies to analyze
#         position: Wah-wah pedal position (0 to 1)
    
#     Returns:
#         magnitude_response: Array of magnitude responses in dB
#     """
#     # Get filter coefficients for the analog prototype
#     f = min_freq * np.exp(position * np.log(max_freq/min_freq))
#     f_khz = f / 1000
#     Q = Q_1k * f_khz ** eQ
    
#     # Calculate analog filter coefficients
#     b0_1k = 0 if LT == -60 else 10 ** (LT * 0.05)
#     b1_1k = 0 if LM == -60 else 10 ** (LM * 0.05)
#     b2_1k = 0 if LB == -60 else 10 ** (LB * 0.05)
    
#     b0_a = b0_1k * f_khz ** eT
#     b1_a = b1_1k * f_khz ** eM
#     b2_a = b2_1k * f_khz ** eB
    
#     # Apply frequency pre-warping like in the digital implementation
#     fw = f * np.tan(pit * f) / (pit * f)
    
#     # Calculate frequency response with pre-warped frequencies
#     # Pre-warp the analysis frequencies too
#     freqs_w = freqs * np.tan(np.pi * freqs / sr) / (np.pi * freqs / sr)
    
#     # Transfer function H(f) = (b2 + b1*jf/(Q*fw) - b0*f^2/fw^2) / (1 + jf/(Q*fw) - f^2/fw^2)
#     numerator = b2_a + b1_a * (1j * freqs_w)/(Q * fw) - b0_a * (freqs_w**2)/(fw**2)
#     denominator = 1 + (1j * freqs_w)/(Q * fw) - (freqs_w**2)/(fw**2)
    
#     H = numerator / denominator
    
#     # Convert to dB
#     magnitude_db = 20 * np.log10(np.abs(H))
#     return magnitude_db

# def measure_frequency_response(freqs, position):
#     """
#     Measure the actual frequency response of the wah-wah effect using z-transform analysis.
    
#     Args:
#         freqs: Array of frequencies to analyze
#         position: Wah-wah pedal position (0 to 1)
    
#     Returns:
#         freq_points: Array of actual frequency points from measurement
#         magnitude_response: Array of magnitude responses in dB
#     """
#     # Get the resonant frequency for this position
#     f = min_freq * np.exp(position * np.log(max_freq/min_freq))
#     f_khz = f / 1000
#     Q = Q_1k * f_khz ** eQ
    
#     # Pre-warp the resonant frequency
#     fw = f * np.tan(pit * f) / (pit * f)
    
#     # Get Q warping
#     aux = pit * f / np.sin(2 * pit * f) * np.log((np.sqrt(1 + 4 * Q * Q) + 1) / (np.sqrt(1 + 4 * Q * Q) - 1))
#     kqw = np.exp(aux) - np.exp(-aux)
    
#     # Get coefficients
#     k = pit * fw
#     kdiv = 1 / (1 + k * (k + kqw))
#     kf = kqw + k
    
#     # Get mixing coefficients
#     b0_1k = 0 if LT == -60 else 10 ** (LT * 0.05)
#     b1_1k = 0 if LM == -60 else 10 ** (LM * 0.05)
#     b2_1k = 0 if LB == -60 else 10 ** (LB * 0.05)
    
#     b0 = b0_1k * f_khz ** eT
#     kb1 = b1_1k * f_khz ** eM * kqw
#     b2 = b2_1k * f_khz ** eB
    
#     # Convert frequencies to normalized frequency
#     w = 2 * np.pi * freqs / sr
#     z = np.exp(1j * w)
    
#     # State-variable filter transfer functions
#     # For the integrator: k/(1-z^(-1))
#     int1 = k / (1 - z**(-1))
    
#     # Calculate transfer functions
#     # From the difference equations:
#     # hp = kdiv * (x - kf*s1 - s2)
#     # bp = aux + s1; s1 = aux + bp  where aux = k*hp
#     # lp = aux + s2; s2 = aux + lp  where aux = k*bp
    
#     # This means:
#     # bp = 2k*hp/(1-z^(-1))
#     # lp = 2k*bp/(1-z^(-1))
    
#     # Solve the system:
#     denom = 1 + kdiv * kf * 2 * k/(1-z**(-1)) + kdiv * (2*k/(1-z**(-1)))**2
#     H_hp = kdiv / denom
#     H_bp = 2 * k * H_hp / (1-z**(-1))
#     H_lp = 2 * k * H_bp / (1-z**(-1))
    
#     # Total response
#     H = b0 * H_hp + kb1 * H_bp + b2 * H_lp
    
#     # Convert to dB
#     mag_db = 20 * np.log10(np.abs(H))
    
#     return freqs, mag_db

# def plot_frequency_response(positions=[1.0], save_path='wahwah_frequency_response.png'):
#     """
#     Plot both theoretical and measured frequency response of the wah-wah effect at different positions.
    
#     Args:
#         positions: List of wah-wah pedal positions to analyze
#         save_path: Path to save the plot
#     """
#     # Generate frequency points (logarithmically spaced)
#     freqs = np.logspace(1, 4.5, 1000)  # 10 Hz to 31.6 kHz
    
#     plt.figure(figsize=(12, 8))
    
#     # Colors for different positions
#     colors = ['blue', 'green', 'red']
#     measured_colors = ['cyan', 'lime', 'magenta']
    
#     # Plot frequency response for each position
#     for pos, theo_color, meas_color in zip(positions, colors, measured_colors):
#         # Theoretical response
#         theoretical_db = calculate_frequency_response(freqs, pos)
#         plt.semilogx(freqs, theoretical_db, color=theo_color, linestyle='-', label=f'Theoretical Pos {pos:.1f}')
        
#         # Measured response
#         meas_freqs, measured_db = measure_frequency_response(freqs, pos)
#         plt.semilogx(meas_freqs, measured_db, color=meas_color, linestyle='--', label=f'Measured Pos {pos:.1f}')
    
#     plt.grid(True, which="both", ls="-", alpha=0.6)
#     plt.xlabel('Frequency / Hz')
#     plt.ylabel('Gain / dB')
#     plt.title('Wah-Wah Frequency Response: Theoretical vs Measured')
#     plt.legend()
#     plt.ylim(-40, 25)
#     plt.xlim(10, 24000)
    
#     # Add vertical lines at characteristic frequencies
#     for freq in [400, 1000, 2500]:
#         plt.axvline(freq, color='gray', linestyle='--', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()


def plot_frequency_response(x, y, fs):
    # Ensure both signals are the same length
    N = min(len(x), len(y))
    x = x[:N]
    y = y[:N]
    
    # FFT
    X = fft(x)
    Y = fft(y)
    
    # Frequency axis
    freqs = fftfreq(N, 1/fs)
    
    # Positive frequencies only
    idx = np.where(freqs >= 0)
    freqs = freqs[idx]
    H = Y[idx] / X[idx]  # Transfer function H(f) = Y(f)/X(f)
    
    # Magnitude and phase
    magnitude = 20 * np.log10(np.abs(H) + 1e-12)  # in dB
    phase = np.angle(H)

    # Plot
    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # plt.plot(freqs, magnitude)
    # plt.title('Frequency Response (Magnitude)')
    # plt.ylabel('Magnitude (dB)')
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(freqs, phase)
    # plt.title('Frequency Response (Phase)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Phase (radians)')
    # plt.grid()

    # plt.tight_layout()
    # plt.show()
    plt.savefig('frequency_response.png', dpi=300, bbox_inches='tight')   
    plt.close()


def plot_measured_vs_gt_frequency_response(dry_path, wet_path, gt_path, save_path=None, show=True):
    """
    Plot measured (wet/dry) and groundtruth (gt/dry) frequency responses on the same plot.
    Args:
        dry_path: Path to dry audio file (reference input)
        wet_path: Path to wet audio file (measured output)
        gt_path: Path to groundtruth audio file (theoretical output)
        save_path: If provided, save the plot to this path
        show: If True, display the plot
    """
    dry, sr = torchaudio.load(dry_path, backend='soundfile')
    wet, _ = torchaudio.load(wet_path, backend='soundfile')
    gt, _ = torchaudio.load(gt_path, backend='soundfile')

    dry = dry.numpy().squeeze()
    wet = wet.numpy().squeeze()
    gt = gt.numpy().squeeze()

    N = min(len(dry), len(wet), len(gt))
    dry = dry[:N]
    wet = wet[:N]
    gt = gt[:N]

    X = fft(dry)
    Y_wet = fft(wet)
    Y_gt = fft(gt)
    freqs = fftfreq(N, 1/sr)
    idx = np.where(freqs >= 0)
    freqs = freqs[idx]
    H_measured = Y_wet[idx] / X[idx]
    H_gt = Y_gt[idx] / X[idx]
    mag_measured = 20 * np.log10(np.abs(H_measured) + 1e-12)
    mag_gt = 20 * np.log10(np.abs(H_gt) + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mag_measured, label="Measured (Wet/Dry)", alpha=0.7)
    plt.plot(freqs, mag_gt, label="Groundtruth (GT/Dry)", alpha=0.7)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency Response: Measured vs Groundtruth")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xscale("log")
    plt.xlim(20, sr//2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Plot both theoretical and measured responses
    import torchaudio

    di = "/Users/michael/Desktop/guitar data/EGDB subset/one sample dry/233.wav"
    wet = "/Users/michael/Desktop/ddsp_code/basics/fx_data/NA_WahFilter_7.5_Power_True_Bypass_False/233.wav"

    x, _ = torchaudio.load(di, backend='soundfile')
    y, _ = torchaudio.load(wet, backend='soundfile')

    sample_rate = 44100.

    plot_frequency_response(x.numpy(), y.numpy(), sample_rate)