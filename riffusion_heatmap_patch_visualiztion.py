# Full script to extract and visualize Q/K/V attention from Riffusion UNet (based on Stable Diffusion v1.5)
# Requirements: diffusers, torch, umap-learn, matplotlib

import torch
import matplotlib.pyplot as plt
import umap
from diffusers import DiffusionPipeline
import numpy as np
import argparse
from PIL import Image

# -----------------------------------
# CONFIGURATION
# -----------------------------------
MODEL_ID = "riffusion/riffusion-model-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Riffusion Attention Heatmap Visualization")
    
    # Model parameters
    parser.add_argument("--model_key", default="riffusion/riffusion-model-v1", 
                       help="Model checkpoint to load")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=25, 
                       help="Number of inference steps")
    
    # Audio input
    parser.add_argument("--audio_path", type=str, required=True,
                       help="Path to input audio file for spectrogram extraction")
    
    # Visualization parameters
    parser.add_argument("--time_step", type=int, default=0,
                       help="Specific denoising time step to visualize")
    parser.add_argument("--attention_layer", type=str, default="6",
                       help="Attention layer index to capture")
    parser.add_argument("--head", type=int, default=0, 
                       help="Attention head to visualize")
    parser.add_argument("--query_token", type=int, default=33,
                       help="Query token to visualize attention for")
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Parse attention layer index
attention_layer_index = int(args.attention_layer)

# -----------------------------------
# SETUP PIPELINE FROM Riffusion
# -----------------------------------
pipe = DiffusionPipeline.from_pretrained(
    args.model_key,
    torch_dtype=torch.float16
).to(args.device)

# Import spectrogram conversion utilities
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# Initialize spectrogram converter
params = SpectrogramParams()
spectrogram_converter = SpectrogramImageConverter(params=params, device=args.device)

qkv_out = {}
current_time_step = None

# Function to extract spectrogram from audio
def extract_spectrogram_from_audio(audio_path: str) -> Image.Image:
    """Extract spectrogram from audio file."""
    print(f"Extracting spectrogram from: {audio_path}")
    
    # Load audio using pydub
    import pydub
    audio_segment = pydub.AudioSegment.from_file(audio_path)
    
    # Convert audio to spectrogram image
    spectrogram_image = spectrogram_converter.spectrogram_image_from_audio(audio_segment)
    
    print(f"Spectrogram extracted with size: {spectrogram_image.size}")
    return spectrogram_image

# Hook function to extract Q/K/V and attention weights
def register_attention_hooks(pipe, target_layer_index):
    layer_count = 0
    
    def hook_fn(module, input, output):
        global current_time_step
        hidden_states = input[0]  # shape: [B, N, C]

        q = module.attn1.to_q(hidden_states)
        k = module.attn1.to_k(hidden_states)
        v = module.attn1.to_v(hidden_states)
        _, attn_weights = module.attn1(hidden_states, return_attn=True)

        # Store with time step information
        if current_time_step is not None:
            key = f"layer_{target_layer_index}_step_{current_time_step}"
        else:
            key = f"layer_{target_layer_index}"
            
        qkv_out[key] = {
            'q': q.detach().cpu(),
            'k': k.detach().cpu(),
            'v': v.detach().cpu(),
            'attn': attn_weights.detach().cpu()
        }

    # Find and hook the specified attention layer
    for name, module in pipe.unet.named_modules():
        if "Attention" in str(type(module)):
            if layer_count == target_layer_index:
                module.register_forward_hook(hook_fn)
                print(f"Hook registered for attention layer {layer_count}: {name}")
                break
            layer_count += 1

# NOTE we need this callback to track time steps since we iterate over the time steps in the reverse diffusion process
# Custom callback to track time steps
def time_step_callback(step, timestep, latents):
    global current_time_step
    current_time_step = step
    if step == args.time_step:
        print(f"Capturing attention at time step {step}")
    return latents

register_attention_hooks(pipe, attention_layer_index)

# -----------------------------------
# EXTRACT SPECTROGRAM AND RUN GENERATION
# -----------------------------------
input_spectrogram = extract_spectrogram_from_audio(args.audio_path)

# Run generation with empty prompt and audio input
print("Running generation with audio input...")
generator = torch.manual_seed(args.seed)
_ = pipe(
    prompt="",  # Empty prompt as requested
    num_inference_steps=args.num_inference_steps, 
    generator=generator,
    callback=time_step_callback,
    image=input_spectrogram  # Use the spectrogram as input
)

# -----------------------------------
# UMAP VISUALIZATION OF Q PATCHES
# -----------------------------------
# Find the captured data for the specified time step
target_key = f"layer_{attention_layer_index}_step_{args.time_step}"
if target_key in qkv_out:
    q = qkv_out[target_key]['q'][0]  # [tokens, dim], e.g., [64, 320]
    print(f"Q shape: {q.shape}")

    # Reduce dimensionality
    reducer = umap.UMAP(n_neighbors=5)
    q_umap = reducer.fit_transform(q.numpy())  # [tokens, 2]

    # Plot UMAP
    plt.figure(figsize=(6, 6))
    plt.scatter(q_umap[:, 0], q_umap[:, 1], c='blue', alpha=0.6)
    plt.title(f"UMAP of Q vectors from Layer {attention_layer_index} at Step {args.time_step}")
    plt.show()
else:
    print(f"No Q vectors captured for {target_key}. Available keys: {list(qkv_out.keys())}")

# -----------------------------------
# ATTENTION WEIGHT HEATMAP OVER PATCH GRID
# -----------------------------------
if target_key in qkv_out:
    attn = qkv_out[target_key]['attn'][0][args.head]  # specified head: [query_tokens, key_tokens]
    query_token = args.query_token
    
    if query_token < attn.shape[0]:
        heatmap = attn[query_token].numpy()  # [64] if 8x8
        heatmap_2d = heatmap.reshape(8, 8)  # adjust if patch grid is different

        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap_2d, cmap='hot', interpolation='nearest')
        plt.title(f"Attention weights for query token #{query_token} at Layer {attention_layer_index}, Step {args.time_step}, Head {args.head}")
        plt.colorbar()
        plt.show()
    else:
        print(f"Query token {query_token} is out of range. Max token index: {attn.shape[0]-1}")
else:
    print(f"No attention weights captured for {target_key}. Available keys: {list(qkv_out.keys())}")

# Print summary
print(f"\nSummary:")
print(f"Audio input: {args.audio_path}")
print(f"Time step: {args.time_step}")
print(f"Attention layer: {attention_layer_index}")
print(f"Attention head: {args.head}")
print(f"Query token: {args.query_token}")
print(f"Total attention maps captured: {len(qkv_out)}")