from utils.plot_spectrogram import plot_spectrogram, plot_single_spectrogram

# EGDB and sinsy vocal
# style_path = "/mnt/gestalt/home/mku666/wah_emulation/EGDB_DI/233.wav"
# content_path = "/mnt/gestalt/home/mku666/vocal2guitar/vocals/233_sinsy.wav"
# output_path = "/home/mku666/riffusion-hobby/output.wav" 

# musicTI dataset
style_path = "/mnt/gestalt/home/mku666/musicTI_audios/timbre/accordion/accordion1.wav"
content_path = "/mnt/gestalt/home/mku666/musicTI_audios/content/piano/piano1.wav"
output_path = "/home/mku666/riffusion-hobby/results/musicTI_piano_accordion_test_t_step_5.wav"

# audio fx dataset
# style_path = "/mnt/gestalt/home/mku666/wah_emulation/NA_Wah_75/NA_WahFilter_7.5_Power_True_Bypass_False/223.wav"
# content_path = "/mnt/gestalt/home/mku666/wah_emulation/EGDB_DI/233.wav"
# output_path = "/home/mku666/riffusion-hobby/results/wah_wah_effect_test.wav"
# output_path = "/home/mku666/riffusion-hobby/results/wah_wah_effect_test_t_step_5.wav"

# Use a two-second segment starting from 0 seconds
start_time = 0  # Start from the beginning
duration = 2.0  # Two seconds duration

plot_spectrogram(style_path, content_path, save_path='spectrogram.png', name1='Style Spectrogram', name2='Content Spectrogram', start_time=start_time, duration=duration)
plot_single_spectrogram(output_path, save_path='output_spectrogram.png', title='Output Spectrogram', start_time=start_time, duration=duration)


