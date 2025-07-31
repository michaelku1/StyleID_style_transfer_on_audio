from stableaudio.pipelines import StyleIDStableAudioPipeline

# Load the pipeline
pipeline = StyleIDStableAudioPipeline.load_checkpoint("stable-audio-model")

# Perform audio style transfer
pipeline.styleid_diffuse(
    content_audio_path="content.wav",
    style_audio_path="style.wav", 
    output_path="output.wav",
    prompt="electronic music",
    gamma=0.75,
    T=1.5,
    start_step=49
)