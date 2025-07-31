import librosa
import numpy as np

def compare_waveforms(file1_path, file2_path, sr=22050):
    """
    Compare two audio files by their waveforms.
    """
    # Load audio files
    y1, sr = librosa.load(file1_path, sr=sr, mono=True)
    y2, sr = librosa.load(file2_path, sr=sr, mono=True)
    
    # Ensure same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # Compute mean squared error
    mse = np.mean((y1 - y2) ** 2)
    
    # Return True if MSE is very small
    return mse < 1e-6, mse

# Example usage
file1_path = "/Users/michael/Desktop/ddsp_code/basics/spotify_pedalboard/results/autowah_test1_old/test1/one sample dry/233.wav"
file2_path = "/Users/michael/Desktop/ddsp_code/basics/spotify_pedalboard/results/autowah_test2_old/test1/one sample dry/233.wav"
# file2_path = "/Users/michael/Desktop/ddsp_code/basics/spotify_pedalboard/results/07F87767-D1B0-4C79-AD9B-096BD704B432_eq_0_delay_0/Sing With Tremolo/one sample dry/233.wav"
try:
    are_same, mse = compare_waveforms(file1_path, file2_path)
    print(f"Are the audio files the same? {are_same}")
    print(f"Mean Squared Error: {mse:.6f}")
except Exception as e:
    print(f"Error processing files: {e}")