#!/usr/bin/env python3
"""
Script to analyze UNet memory consumption in Riffusion pipeline.
"""

import torch
import torch.nn as nn
import gc
import psutil
import os
from typing import Optional, Dict, Any
import sys

# Add the riffusion directory to the path
sys.path.append('.')

try:
    from riffusion.riffusion_pipeline import RiffusionPipeline
    from riffusion.util import torch_util
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size_mb(model: nn.Module, dtype: torch.dtype = torch.float16) -> float:
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb


def analyze_unet_memory():
    """Analyze UNet memory consumption."""
    print("=== UNet Memory Analysis ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB (CPU)")
    if torch.cuda.is_available():
        print(f"Initial GPU memory usage: {get_gpu_memory_usage():.2f} MB")
    print()
    
    # Load the pipeline
    print("Loading Riffusion pipeline...")
    try:
        pipeline = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            device=device,
            dtype=dtype,
            use_traced_unet=False,  # Use regular UNet for analysis
            local_files_only=False
        )
        
        print("Pipeline loaded successfully!")
        print(f"Memory after loading: {get_memory_usage():.2f} MB (CPU)")
        if torch.cuda.is_available():
            print(f"GPU memory after loading: {get_gpu_memory_usage():.2f} MB")
        print()
        
        # Analyze UNet specifically
        unet = pipeline.unet
        print("=== UNet Analysis ===")
        
        # Parameter count
        param_counts = count_parameters(unet)
        print(f"UNet Parameters:")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Trainable: {param_counts['trainable']:,}")
        print(f"  Non-trainable: {param_counts['non_trainable']:,}")
        print()
        
        # Model size
        model_size_mb = get_model_size_mb(unet, dtype)
        print(f"UNet Model Size: {model_size_mb:.2f} MB")
        print()
        
        # Memory usage breakdown
        print("=== Memory Breakdown ===")
        
        # Check if UNet is on GPU
        if hasattr(unet, 'device'):
            print(f"UNet device: {unet.device}")
        else:
            print(f"UNet device: {next(unet.parameters()).device}")
        
        # Analyze each component
        total_params = 0
        total_size = 0
        
        for name, module in unet.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    module_size = sum(p.numel() * p.element_size() for p in module.parameters())
                    total_params += param_count
                    total_size += module_size
                    print(f"{name}: {param_count:,} params, {module_size/1024/1024:.2f} MB")
        
        print(f"\nTotal UNet parameters: {total_params:,}")
        print(f"Total UNet size: {total_size/1024/1024:.2f} MB")
        
        # Compare with other components
        print("\n=== Component Comparison ===")
        vae_params = count_parameters(pipeline.vae)
        text_encoder_params = count_parameters(pipeline.text_encoder)
        
        print(f"VAE: {vae_params['total']:,} params ({get_model_size_mb(pipeline.vae, dtype):.2f} MB)")
        print(f"Text Encoder: {text_encoder_params['total']:,} params ({get_model_size_mb(pipeline.text_encoder, dtype):.2f} MB)")
        print(f"UNet: {param_counts['total']:,} params ({model_size_mb:.2f} MB)")
        
        # Calculate percentages
        total_model_params = param_counts['total'] + vae_params['total'] + text_encoder_params['total']
        unet_percentage = (param_counts['total'] / total_model_params) * 100
        print(f"\nUNet represents {unet_percentage:.1f}% of total model parameters")
        
        # Memory usage during inference
        print("\n=== Inference Memory Analysis ===")
        
        # Create dummy inputs for inference
        batch_size = 1
        latent_height = 64
        latent_width = 64
        text_embedding_dim = 768
        
        dummy_latents = torch.randn(
            batch_size, 4, latent_height, latent_width,
            device=device, dtype=dtype
        )
        dummy_timestep = torch.tensor([500], device=device, dtype=dtype)
        dummy_text_embeddings = torch.randn(
            batch_size, 77, text_embedding_dim,
            device=device, dtype=dtype
        )
        
        # Measure memory before inference
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        memory_before = get_memory_usage()
        gpu_memory_before = get_gpu_memory_usage()
        
        print(f"Memory before inference: {memory_before:.2f} MB (CPU), {gpu_memory_before:.2f} MB (GPU)")
        
        # Run inference
        with torch.no_grad():
            output = unet(dummy_latents, dummy_timestep, dummy_text_embeddings)
        
        memory_after = get_memory_usage()
        gpu_memory_after = get_gpu_memory_usage()
        
        print(f"Memory after inference: {memory_after:.2f} MB (CPU), {gpu_memory_after:.2f} MB (GPU)")
        print(f"Memory increase: {memory_after - memory_before:.2f} MB (CPU), {gpu_memory_after - gpu_memory_before:.2f} MB (GPU)")
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return


if __name__ == "__main__":
    analyze_unet_memory() 