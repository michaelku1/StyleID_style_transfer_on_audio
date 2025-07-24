# StyleID-Enhanced Riffusion: Spectrogram-Based Style Transfer

This implementation adapts StyleID techniques to the Riffusion model for training-free spectrogram-based style transfer. Instead of working with natural images, this system operates on spectrograms, enabling musical style transfer while preserving content structure.

## Overview

StyleID is a training-free approach for image style transfer that uses:
1. **KV Style Feature Injection**: Injects style information through key and value features from attention layers
2. **Query Preservation**: Preserves content structure by keeping query features from the content image
3. **Temperature Scaling**: Controls attention map sharpness for better style transfer
4. **AdaIN Initialization**: Initializes latents with style statistics using Adaptive Instance Normalization

## Key Techniques Adapted for Spectrograms

### 1. KV Style Feature Injection
- **Content Channel**: Extracts query (Q) features from content spectrograms to preserve musical structure
- **Style Channel**: Extracts key (K) and value (V) features from style spectrograms to transfer musical style
- **Injection Strategy**: 
  - Q features come from content spectrograms (preserves structure)
  - K and V features come from style spectrograms (transfers style)

### 2. Query Preservation (γ parameter)
- Controls the balance between content preservation and style transfer
- γ = 0: Full style transfer (may lose content structure)
- γ = 1: Full content preservation (may lose style)
- Default: γ = 0.75 (balanced approach)

### 3. Temperature Scaling (T parameter)
- Scales attention maps to control the sharpness of style transfer
- T > 1: Sharper attention (more focused style transfer)
- T < 1: Softer attention (more gradual style transfer)
- Default: T = 1.5

### 4. AdaIN Initialization
- Transfers style statistics from style spectrograms to content spectrograms
- Preserves content structure while applying style characteristics
- Applied at the latent level before diffusion sampling

## Implementation Details

### Architecture Modifications

The implementation extends the original Riffusion pipeline with:

```python
class StyleIDRiffusionPipeline(RiffusionPipeline):
    def __init__(self, ...):
        # StyleID parameters
        self.gamma = 0.75  # Query preservation
        self.T = 1.5      # Temperature scaling
        self.attn_layers = [6, 7, 8, 9, 10, 11]  # Injection layers
        self.start_step = 49  # Injection start step
```

### Feature Extraction Process

1. **DDIM Inversion**: Both content and style spectrograms are inverted using DDIM to extract features
2. **Attention Feature Capture**: Q, K, V features are extracted from specified attention layers
3. **Feature Storage**: Features are stored per timestep for injection during generation

### Feature Injection During Generation

```python
def feat_merge(content_feats, style_feats, start_step=0, gamma=0.75, T=1.5):
    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                # Preserve content queries
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                # Inject style keys and values
                feat_maps[i][ori_key] = sty_feat[ori_key]
```

## Installation and Setup

### Prerequisites

```bash
# Install base requirements
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install pillow matplotlib numpy einops

# Install Riffusion dependencies
cd riffusion-hobby
pip install -e .
```

### Model Setup

```python
from riffusion.styleid_riffusion_pipeline import StyleIDRiffusionPipeline

# Load the StyleID-enhanced pipeline
pipeline = StyleIDRiffusionPipeline.load_checkpoint(
    checkpoint="riffusion/riffusion-model-v1",
    device="cuda",
    dtype=torch.float16
)

# Setup feature extraction
pipeline.setup_feature_extraction()
```

## Usage Examples

### Basic Style Transfer

```python
from riffusion.datatypes import InferenceInput, PromptInput

# Create inference parameters
start = PromptInput(
    prompt="electronic music with synthesizers",
    seed=42,
    guidance=7.5,
    denoising=0.8,
)

end = PromptInput(
    prompt="jazz fusion with acoustic instruments", 
    seed=43,
    guidance=7.5,
    denoising=0.8,
)

inputs = InferenceInput(
    start=start,
    end=end,
    alpha=0.5,  # Interpolation factor
    num_inference_steps=50,
)

# Run style transfer
stylized_spectrogram = pipeline.styleid_riffuse(
    inputs=inputs,
    content_image=content_spectrogram,
    style_image=style_spectrogram,
    gamma=0.75,      # Query preservation
    T=1.5,          # Temperature scaling
    start_step=49,  # Injection start step
)
```

### Command Line Usage

```bash
python examples/styleid_spectrogram_transfer.py \
    --content_spectrogram path/to/content.png \
    --style_spectrogram path/to/style.png \
    --output_path output/ \
    --content_prompt "electronic music with synthesizers" \
    --style_prompt "jazz fusion with acoustic instruments" \
    --alpha 0.5 \
    --gamma 0.75 \
    --T 1.5 \
    --start_step 49
```

## Parameter Tuning

### Style Transfer Control

| Parameter | Range | Effect | Default |
|-----------|-------|--------|---------|
| `alpha` | 0-1 | Interpolation between content/style | 0.5 |
| `gamma` | 0-1 | Query preservation strength | 0.75 |
| `T` | 0.1-3.0 | Attention temperature scaling | 1.5 |
| `start_step` | 0-49 | When to start feature injection | 49 |

### Generation Quality

| Parameter | Range | Effect | Default |
|-----------|-------|--------|---------|
| `num_inference_steps` | 20-100 | Sampling quality vs speed | 50 |
| `guidance_scale` | 1-20 | Text adherence | 7.5 |
| `denoising_strength` | 0.1-1.0 | Style transfer intensity | 0.8 |

## Advanced Usage

### Batch Processing

```python
# Process multiple style transfers
for alpha in [0.2, 0.4, 0.6, 0.8]:
    inputs.alpha = alpha
    result = pipeline.styleid_riffuse(
        inputs=inputs,
        content_image=content_img,
        style_image=style_img,
        gamma=0.75,
        T=1.5
    )
    # Save result...
```

### Feature Precomputation

```python
# Precompute features for faster processing
content_latents, content_features = pipeline.extract_features_ddim(
    content_image, num_steps=50
)
style_latents, style_features = pipeline.extract_features_ddim(
    style_image, num_steps=50
)

# Save features for reuse
with open('content_features.pkl', 'wb') as f:
    pickle.dump(content_features, f)
```

## Technical Details

### Attention Layer Selection

The implementation targets specific attention layers in the UNet:
- Layers 6-11: Mid-level features for balanced style transfer
- Earlier layers: More structural preservation
- Later layers: More style transfer

### DDIM Inversion Process

1. **Forward Process**: Add noise to spectrogram latents
2. **Reverse Process**: Predict and remove noise step by step
3. **Feature Extraction**: Capture Q, K, V at each timestep
4. **Storage**: Organize features by timestep and layer

### Memory Optimization

- Features are stored per timestep to minimize memory usage
- Optional feature precomputation for batch processing
- Gradient checkpointing for large models

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_inference_steps` or use gradient checkpointing
2. **Poor Style Transfer**: Increase `T` or decrease `gamma`
3. **Loss of Content**: Increase `gamma` or decrease `T`
4. **Slow Processing**: Precompute features or reduce attention layers

### Performance Tips

- Use GPU acceleration when available
- Precompute features for batch processing
- Adjust `start_step` based on desired style transfer intensity
- Use mixed precision (float16) for faster inference

## Limitations and Future Work

### Current Limitations

- Requires spectrogram images as input
- Limited to the Riffusion model architecture
- Feature injection is architecture-specific
- No real-time processing capability

### Future Improvements

- Support for real-time audio input
- Adaptive parameter selection
- Multi-style transfer
- Cross-modal style transfer (audio-to-visual)

## Citation

If you use this implementation, please cite:

```bibtex
@article{styleid2023,
  title={StyleID: Training-Free Style Transfer with Attention Injection},
  author={...},
  journal={...},
  year={2023}
}

@article{riffusion2022,
  title={Riffusion: Real-time music generation with stable diffusion},
  author={...},
  journal={...},
  year={2022}
}
```

## License

This implementation follows the same license as the original Riffusion project. 