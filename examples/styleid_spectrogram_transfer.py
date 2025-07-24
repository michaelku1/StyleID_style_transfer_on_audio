#!/usr/bin/env python3
"""
Example script for StyleID-enhanced Riffusion spectrogram style transfer.

This script demonstrates how to use the StyleID techniques with Riffusion:
1. Load content and style spectrograms
2. Extract features using DDIM inversion
3. Apply StyleID techniques (KV injection, query preservation, AdaIN)
4. Generate stylized spectrograms
5. Convert back to audio

Usage:
    python styleid_spectrogram_transfer.py \
        --content_spectrogram path/to/content.png \
        --style_spectrogram path/to/style.png \
        --output_path output/ \
        --prompt "electronic music" \
        --style_prompt "jazz fusion"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the riffusion package to the path
sys.path.append(str(Path(__file__).parent.parent))

from riffusion.styleid_riffusion_pipeline import StyleIDRiffusionPipeline
from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.util import torch_util


def load_spectrogram_image(image_path: str) -> Image.Image:
    """Load and preprocess a spectrogram image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Spectrogram image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    print(f"Loaded spectrogram: {image_path} (size: {image.size})")
    return image


def create_inference_input(
    content_prompt: str,
    style_prompt: str,
    alpha: float = 0.5,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    denoising_strength: float = 0.8,
    seed: int = 42,
) -> InferenceInput:
    """Create an InferenceInput object for the StyleID pipeline."""
    
    # Create start and end prompts
    start = PromptInput(
        prompt=content_prompt,
        seed=seed,
        guidance=guidance_scale,
        denoising=denoising_strength,
    )
    
    end = PromptInput(
        prompt=style_prompt,
        seed=seed + 1,  # Different seed for variety
        guidance=guidance_scale,
        denoising=denoising_strength,
    )
    
    return InferenceInput(
        start=start,
        end=end,
        alpha=alpha,
        num_inference_steps=num_inference_steps,
    )


def save_stylized_spectrogram(
    image: Image.Image, 
    output_path: str, 
    filename: str
) -> str:
    """Save the stylized spectrogram image."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    image.save(filepath)
    print(f"Saved stylized spectrogram: {filepath}")
    return filepath


def visualize_comparison(
    content_img: Image.Image,
    style_img: Image.Image, 
    stylized_img: Image.Image,
    output_path: str,
    filename: str = "comparison.png"
):
    """Create a side-by-side comparison of content, style, and stylized images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(content_img)
    axes[0].set_title("Content Spectrogram")
    axes[0].axis('off')
    
    axes[1].imshow(style_img)
    axes[1].set_title("Style Spectrogram")
    axes[1].axis('off')
    
    axes[2].imshow(stylized_img)
    axes[2].set_title("Stylized Spectrogram")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison visualization: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="StyleID-enhanced Riffusion spectrogram style transfer"
    )
    
    # Input/output paths
    parser.add_argument(
        "--content_spectrogram", 
        type=str, 
        required=True,
        help="Path to content spectrogram image"
    )
    parser.add_argument(
        "--style_spectrogram", 
        type=str, 
        required=True,
        help="Path to style spectrogram image"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output",
        help="Output directory for results"
    )
    
    # Text prompts
    parser.add_argument(
        "--content_prompt", 
        type=str, 
        default="electronic music with synthesizers",
        help="Text prompt describing the content"
    )
    parser.add_argument(
        "--style_prompt", 
        type=str, 
        default="jazz fusion with acoustic instruments",
        help="Text prompt describing the style"
    )
    
    # StyleID parameters
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5,
        help="Interpolation factor between content and style (0-1)"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.75,
        help="Query preservation parameter (0-1)"
    )
    parser.add_argument(
        "--T", 
        type=float, 
        default=1.5,
        help="Temperature scaling parameter for attention maps"
    )
    parser.add_argument(
        "--start_step", 
        type=int, 
        default=49,
        help="Starting step for feature injection"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--denoising_strength", 
        type=float, 
        default=0.8,
        help="Denoising strength"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # StyleID options
    parser.add_argument(
        "--use_adain_init", 
        action="store_true",
        default=True,
        help="Use AdaIN initialization"
    )
    parser.add_argument(
        "--use_attn_injection", 
        action="store_true",
        default=True,
        help="Use attention feature injection"
    )
    
    # Model parameters
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="riffusion/riffusion-model-v1",
        help="Model checkpoint to use"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch_util.check_device(args.device)
    print(f"Using device: {device}")
    
    # Load spectrograms
    print("Loading spectrograms...")
    content_img = load_spectrogram_image(args.content_spectrogram)
    style_img = load_spectrogram_image(args.style_spectrogram)
    
    # Load StyleID-enhanced Riffusion pipeline
    print("Loading StyleID-enhanced Riffusion pipeline...")
    pipeline = StyleIDRiffusionPipeline.load_checkpoint(
        checkpoint=args.checkpoint,
        device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    
    # Setup feature extraction
    pipeline.setup_feature_extraction()
    
    # Create inference input
    inference_input = create_inference_input(
        content_prompt=args.content_prompt,
        style_prompt=args.style_prompt,
        alpha=args.alpha,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        denoising_strength=args.denoising_strength,
        seed=args.seed,
    )
    
    print(f"Content prompt: {args.content_prompt}")
    print(f"Style prompt: {args.style_prompt}")
    print(f"Alpha: {args.alpha}")
    print(f"Gamma: {args.gamma}")
    print(f"Temperature: {args.T}")
    print(f"Start step: {args.start_step}")
    
    # Run StyleID-enhanced riffusion
    print("Running StyleID-enhanced riffusion...")
    stylized_img = pipeline.styleid_riffuse(
        inputs=inference_input,
        content_image=content_img,
        style_image=style_img,
        use_adain_init=args.use_adain_init,
        use_attn_injection=args.use_attn_injection,
        gamma=args.gamma,
        T=args.T,
        start_step=args.start_step,
    )
    
    # Save results
    print("Saving results...")
    timestamp = int(time.time())
    
    # Save stylized spectrogram
    stylized_filename = f"stylized_spectrogram_{timestamp}.png"
    stylized_path = save_stylized_spectrogram(
        stylized_img, 
        args.output_path, 
        stylized_filename
    )
    
    # Create comparison visualization
    comparison_filename = f"comparison_{timestamp}.png"
    visualize_comparison(
        content_img,
        style_img,
        stylized_img,
        args.output_path,
        comparison_filename
    )
    
    # Save parameters
    params_filename = f"parameters_{timestamp}.txt"
    params_path = os.path.join(args.output_path, params_filename)
    with open(params_path, 'w') as f:
        f.write("StyleID-Enhanced Riffusion Parameters\n")
        f.write("=" * 40 + "\n")
        f.write(f"Content spectrogram: {args.content_spectrogram}\n")
        f.write(f"Style spectrogram: {args.style_spectrogram}\n")
        f.write(f"Content prompt: {args.content_prompt}\n")
        f.write(f"Style prompt: {args.style_prompt}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"Temperature: {args.T}\n")
        f.write(f"Start step: {args.start_step}\n")
        f.write(f"Num inference steps: {args.num_inference_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Denoising strength: {args.denoising_strength}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Use AdaIN init: {args.use_adain_init}\n")
        f.write(f"Use attention injection: {args.use_attn_injection}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
    
    print(f"Saved parameters: {params_path}")
    print("\nStyle transfer completed successfully!")
    print(f"Results saved in: {args.output_path}")
    print(f"Stylized spectrogram: {stylized_path}")
    print(f"Comparison visualization: {os.path.join(args.output_path, comparison_filename)}")


if __name__ == "__main__":
    main() 