# Chunked Inference for Riffusion

This module provides chunked inference capabilities for Riffusion, allowing you to process long audio files by splitting them into smaller chunks, processing each chunk with the Riffusion model, and then stitching the results back together.

## Features

- **Automatic chunking**: Split long audio files into configurable chunks
- **Overlap handling**: Smooth transitions between chunks with configurable overlap
- **Custom chunk boundaries**: Define custom chunk boundaries for specific use cases
- **Progress tracking**: Monitor processing progress with callbacks
- **Memory efficient**: Process chunks sequentially to manage memory usage
- **Crossfade stitching**: Smooth audio reconstruction with crossfading

## Installation

The chunked inference module is part of the Riffusion codebase. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from riffusion.chunked_inference import create_chunked_inference
from riffusion.datatypes import InferenceInput, PromptInput
from PIL import Image

# Create chunked inference pipeline
chunked_inference = create_chunked_inference(
    model_path="riffusion/riffusion-model-v1",
    chunk_duration_ms=10000,  # 10 seconds per chunk
    overlap_ms=1000,          # 1 second overlap
    fade_duration_ms=500,     # 500ms fade
    device="cuda"
)

# Create inference inputs
prompt_input = PromptInput(
    prompt="funky synth solo",
    seed=42,
    denoising=0.75,
    guidance=7.5,
)

inputs = InferenceInput(
    start=prompt_input,
    end=prompt_input,
    alpha=0.0,
    num_inference_steps=50,
    seed_image_id="vibes",
)

# Load seed image
init_image = Image.open("seed_images/vibes.png").convert("RGB")

# Process audio file
processed_audio = chunked_inference.process_audio_file(
    audio_path="input.wav",
    inputs=inputs,
    init_image=init_image,
    output_path="output.wav"
)
```

### Command Line Interface

Use the provided CLI script for easy command-line processing:

```bash
python chunked_inference_cli.py \
    --input input.wav \
    --output output.wav \
    --prompt "funky synth solo" \
    --chunk-duration-ms 10000 \
    --overlap-ms 1000 \
    --denoising 0.75
```

### Custom Chunk Boundaries

For more control, you can define custom chunk boundaries:

```python
# Define custom chunk boundaries (in milliseconds)
chunk_boundaries_ms = [0, 5000, 15000, 25000, 35000]

processed_audio = chunked_inference.process_audio_with_custom_chunks(
    audio_path="input.wav",
    inputs=inputs,
    init_image=init_image,
    chunk_boundaries_ms=chunk_boundaries_ms,
    output_path="output.wav"
)
```

Or via CLI:

```bash
python chunked_inference_cli.py \
    --input input.wav \
    --output output.wav \
    --prompt "electronic beat" \
    --custom-chunks 0 5000 15000 25000 35000
```

## Parameters

### Chunking Parameters

- **`chunk_duration_ms`**: Duration of each chunk in milliseconds (default: 10000)
- **`overlap_ms`**: Overlap between chunks in milliseconds (default: 1000)
- **`fade_duration_ms`**: Duration of fade in/out for smooth transitions (default: 500)

### Inference Parameters

- **`denoising`**: Denoising strength (0.0 to 1.0, default: 0.75)
- **`guidance`**: Guidance scale (default: 7.5)
- **`num_inference_steps`**: Number of inference steps (default: 50)
- **`seed`**: Random seed for reproducible results (default: 42)

## CLI Options

```bash
python chunked_inference_cli.py --help
```

### Required Arguments

- `--input, -i`: Input audio file path
- `--output, -o`: Output audio file path
- `--prompt, -p`: Text prompt for audio generation

### Optional Arguments

- `--model-path`: Path to the Riffusion model (default: "riffusion/riffusion-model-v1")
- `--seed-image`: Seed image ID (default: "vibes")
- `--chunk-duration-ms`: Duration of each chunk in milliseconds (default: 10000)
- `--overlap-ms`: Overlap between chunks in milliseconds (default: 1000)
- `--fade-duration-ms`: Fade duration in milliseconds (default: 500)
- `--denoising`: Denoising strength (default: 0.75)
- `--guidance`: Guidance scale (default: 7.5)
- `--num-steps`: Number of inference steps (default: 50)
- `--seed`: Random seed (default: 42)
- `--device`: Device to run inference on (default: "cuda")
- `--custom-chunks`: Custom chunk boundaries in milliseconds
- `--verbose, -v`: Enable verbose output

## Examples

### Process a 5-minute audio file

```bash
python chunked_inference_cli.py \
    --input long_audio.wav \
    --output processed_audio.wav \
    --prompt "ambient electronic music" \
    --chunk-duration-ms 15000 \
    --overlap-ms 2000 \
    --denoising 0.8
```

### Use custom chunk boundaries based on beat detection

```bash
python chunked_inference_cli.py \
    --input song.wav \
    --output processed_song.wav \
    --prompt "rock guitar solo" \
    --custom-chunks 0 8000 16000 24000 32000 40000
```

### Process with different parameters for each chunk

```python
# This would require custom implementation
# You could modify the chunked_inference.py to support different
# parameters for each chunk if needed
```

## Memory Management

The chunked inference approach helps manage memory usage by:

1. **Sequential processing**: Only one chunk is processed at a time
2. **Configurable chunk size**: Adjust chunk duration based on available memory
3. **Automatic cleanup**: Memory is freed after each chunk is processed

### Memory Recommendations

- **GPU Memory < 8GB**: Use 5-10 second chunks
- **GPU Memory 8-16GB**: Use 10-15 second chunks
- **GPU Memory > 16GB**: Use 15-20 second chunks

## Performance Tips

1. **Chunk size**: Larger chunks are more efficient but require more memory
2. **Overlap**: 10-20% overlap provides good transitions
3. **Fade duration**: 5-10% of chunk duration for smooth transitions
4. **Batch processing**: Process multiple files sequentially to avoid memory issues

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce chunk duration or use CPU
2. **Poor transitions**: Increase overlap or fade duration
3. **Long processing time**: Reduce number of inference steps
4. **Audio artifacts**: Adjust fade duration or overlap

### Debug Mode

Use the `--verbose` flag for detailed output:

```bash
python chunked_inference_cli.py \
    --input input.wav \
    --output output.wav \
    --prompt "test" \
    --verbose
```

## Advanced Usage

### Custom Progress Callback

```python
def my_progress_callback(chunk_num, total_chunks):
    print(f"Processing chunk {chunk_num}/{total_chunks}")

processed_audio = chunked_inference.process_audio_file(
    audio_path="input.wav",
    inputs=inputs,
    init_image=init_image,
    progress_callback=my_progress_callback
)
```

### Integration with Existing Pipeline

```python
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.chunked_inference import ChunkedRiffusionInference

# Load existing pipeline
pipeline = RiffusionPipeline.load_checkpoint("riffusion/riffusion-model-v1")

# Create chunked inference wrapper
chunked_inference = ChunkedRiffusionInference(
    pipeline=pipeline,
    chunk_duration_ms=10000,
    overlap_ms=1000,
    fade_duration_ms=500
)
```

## File Structure

```
riffusion/
├── chunked_inference.py          # Main chunked inference implementation
├── riffusion_pipeline.py         # Original pipeline
└── ...

chunked_inference_cli.py          # Command-line interface
example_chunked_inference.py      # Example usage
README_CHUNKED_INFERENCE.md       # This documentation
```

## Contributing

To extend the chunked inference functionality:

1. Modify `riffusion/chunked_inference.py` for core changes
2. Update `chunked_inference_cli.py` for CLI changes
3. Add tests for new functionality
4. Update this documentation

## License

This module is part of the Riffusion project and follows the same license terms. 