import torchaudio


target_sr = 44100

audio, sr = torchaudio.load("/mnt/gestalt/home/mku666/vocal2guitar/vocals/233_sinsy.wav")
print(f"Loaded audio from {sr}Hz")

audio = torchaudio.functional.resample(audio, sr, target_sr)

torchaudio.save("233_sinsy_resampled.wav", audio, target_sr)
print(f"Resampled from {sr}Hz to {target_sr}Hz")






