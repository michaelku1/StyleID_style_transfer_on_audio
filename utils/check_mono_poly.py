import librosa
import numpy as np
import torchaudio

y, sr = torchaudio.load('/Users/michael/Desktop/guitar data/dataset1/train/dry/233.wav')
y = y.squeeze(0)
y = y.numpy()

harmonic, _ = librosa.effects.hpss(y)
pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr)

print(magnitudes.shape)

exit()

threshold = 0.1

active_pitches = np.sum(magnitudes > threshold, axis=0)  # Number of pitches per frame
avg_polyphony = np.mean(active_pitches)
print(f"Average number of active pitches: {avg_polyphony}")
if avg_polyphony > 1.5:
    print("Likely polyphonic")
else:
    print("Likely monophonic")