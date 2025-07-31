import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import List
from PIL import Image

"""
Riffusion Attention Visualization Script

Usage examples:
    # Basic usage with default attention layers [6,7,8,9,10,11]
    python riffusion_attention_visualization.py --prompt "electronic music"
    
    # Visualize specific time step (step 10)
    python riffusion_attention_visualization.py --time_step 10 --attention_layer "6,7,8"
    
    # Custom attention layers and head
    python riffusion_attention_visualization.py --attention_layer "0,1,2,3" --head 2
    
    # Full custom example
    python riffusion_attention_visualization.py --time_step 15 --attention_layer "6,7,8,9,10,11" --head 0 --prompt "jazz piano"
    
    # Audio input with spectrogram visualization
    python riffusion_attention_visualization.py --audio_path "path/to/audio.wav" --time_step 10 --attention_layer "6,7,8"
    
    # Audio input with spectrogram saving
    python riffusion_attention_visualization.py --audio_path "path/to/audio.wav" --save_spectrogram --attention_layer "6,7,8,9,10,11"
    
    # Audio input with specific time step and custom layers
    python riffusion_attention_visualization.py --audio_path "path/to/audio.wav" --time_step 20 --attention_layer "0,1,2,3,4,5" --head 1
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Riffusion Attention Visualization")
    
    # Model parameters
    parser.add_argument("--model_key", default="riffusion/riffusion-model-v1", 
                       help="Model checkpoint to load")
    parser.add_argument("--device", default="cuda:1", help="Device to run on")
    parser.add_argument("--prompt", default="", help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=25, 
                       help="Number of inference steps")
    
    # Audio input
    parser.add_argument("--audio_path", type=str, default=None,
                       help="Path to input audio file for spectrogram extraction")
    
    # Visualization parameters
    parser.add_argument("--time_step", type=int, default=None,
                       help="Specific denoising time step to visualize")
    parser.add_argument("--attention_layer", type=str, default="6,7,8,9,10,11",
                       help="Comma-separated list of attention layer indices to capture")
    parser.add_argument("--head", type=int, default=0, 
                       help="Attention head to visualize")
    parser.add_argument("--save_spectrogram", action="store_true",
                       help="Save the input spectrogram image")
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Parse attention layer indices
attention_layer_indices = [int(x.strip()) for x in args.attention_layer.split(',')]

# Load Riffusion model
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

# Create a dictionary to store attention maps
attention_log = {}
current_time_step = None
input_spectrogram = None

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

# Hook function to capture attention weights
def hook_fn(name):
    def fn(module, input, output):
        global current_time_step
        if hasattr(module, 'to_q') and hasattr(module, 'to_k'):
            q = module.to_q(input[0])
            k = module.to_k(input[0])
            attn_scores = torch.einsum('b i d, b j d -> b i j', q, k) / q.shape[-1]**0.5
            attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
            
            # Store with time step information
            if current_time_step is not None:
                key = f"{name}_step_{current_time_step}"
                attention_log[key] = attn_probs.detach().cpu()
            else:
                attention_log[name] = attn_probs.detach().cpu()
    return fn

# Custom callback to track time steps
def time_step_callback(step, timestep, latents):
    global current_time_step
    current_time_step = step
    if args.time_step is not None and step == args.time_step:
        print(f"Capturing attention at time step {step}")
    return latents

# Register hooks only to specified attention layers
layer_count = 0
for name, module in pipe.unet.named_modules():
    if "Attention" in str(type(module)):
        if layer_count in attention_layer_indices:
            module.register_forward_hook(hook_fn(name))
            print(f"Registered hook for attention layer {layer_count}: {name}")
        layer_count += 1

# Extract spectrogram if audio path is provided
if args.audio_path:
    input_spectrogram = extract_spectrogram_from_audio(args.audio_path)
    
    # Save spectrogram if requested
    if args.save_spectrogram:
        spectrogram_path = "input_spectrogram.png"
        input_spectrogram.save(spectrogram_path)
        print(f"Saved input spectrogram to: {spectrogram_path}")
    
    # Use the spectrogram as input for generation
    print("Using extracted spectrogram as input for generation...")
    generator = torch.manual_seed(args.seed)
    _ = pipe(
        prompt=args.prompt, 
        num_inference_steps=args.num_inference_steps, 
        generator=generator,
        callback=time_step_callback if args.time_step is not None else None,
        image=input_spectrogram  # Use the spectrogram as input
    )
else:
    # Standard text-to-image generation
    print("Using text prompt for generation...")
    generator = torch.manual_seed(args.seed)
    _ = pipe(
        prompt=args.prompt, 
        num_inference_steps=args.num_inference_steps, 
        generator=generator,
        callback=time_step_callback if args.time_step is not None else None
    )

# Enhanced plotting function for spectrogram attention visualization
def plot_attention_with_spectrogram(name, head=0, spectrogram=None):
    if name not in attention_log:
        print(f"No attention recorded for: {name}")
        return
    
    attn = attention_log[name][0]  # (num_heads, tokens, tokens)
    attn_head = attn[head].numpy()
    
    # Create subplot with spectrogram and attention map
    if spectrogram is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot spectrogram
        ax1.imshow(spectrogram, cmap='viridis', aspect='auto')
        ax1.set_title("Input Spectrogram")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Frequency")
        
        # Plot attention map
        im = ax2.imshow(attn_head, cmap='inferno')
        ax2.set_title(f"Attention Map: {name} | Head: {head}")
        ax2.set_xlabel("Key tokens")
        ax2.set_ylabel("Query tokens")
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    else:
        # Standard attention plot
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_head, cmap='inferno')
        plt.title(f"Layer: {name} | Head: {head}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

# Example: list layers and plot first one
print("Logged attention layers:")
for i, key in enumerate(attention_log.keys()):
    print(f"{i}: {key}")
layer_names = list(attention_log.keys())
if layer_names:
    plot_attention_with_spectrogram(layer_names[0], head=args.head, spectrogram=input_spectrogram)
else:
    print("No attention layers captured. Check hooks or model structure.")

# Print summary of captured attention maps
print(f"\nSummary:")
print(f"Audio input: {args.audio_path}")
print(f"Time step specified: {args.time_step}")
print(f"Attention layers captured: {attention_layer_indices}")
print(f"Total attention maps captured: {len(attention_log)}")

if args.time_step is not None:
    time_step_maps = [k for k in attention_log.keys() if f"_step_{args.time_step}" in k]
    print(f"Attention maps at time step {args.time_step}: {len(time_step_maps)}")
    if time_step_maps:
        print("Available time step maps:")
        for map_name in time_step_maps:
            print(f"  - {map_name}")
