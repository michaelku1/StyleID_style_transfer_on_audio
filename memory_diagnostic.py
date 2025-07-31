#!/usr/bin/env python3
"""
Diagnostic script to identify root cause of 40GB+ memory usage in Colab.
"""

import torch
import gc
import sys
import os
from typing import Dict, Any

# Add the riffusion directory to the path
sys.path.append('.')

try:
    from riffusion.riffusion_pipeline import RiffusionPipeline
    from riffusion.util import torch_util
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def get_memory_info():
    """Get detailed memory information."""
    info = {}
    
    # CPU memory
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
        info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # Python memory
    try:
        import psutil
        process = psutil.Process(os.getpid())
        info['ram_used'] = process.memory_info().rss / 1024**3  # GB
    except ImportError:
        info['ram_used'] = "psutil not available"
    
    return info


def print_memory_info(label: str):
    """Print memory information with a label."""
    info = get_memory_info()
    print(f"\n=== {label} ===")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f} GB")
        else:
            print(f"{key}: {value}")


def diagnose_memory_issues():
    """Diagnose potential memory issues."""
    print("=== Memory Issue Diagnosis ===\n")
    
    # Check PyTorch version and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    
    print_memory_info("Initial State")
    
    # Test 1: Load model with different settings
    print("\n=== Test 1: Model Loading ===")
    
    # Test with traced UNet (default)
    print("\n--- Loading with traced UNet ---")
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print_memory_info("Before loading")
        
        pipeline1 = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            use_traced_unet=True,
            dtype=torch.float16,
            device="cuda",
            local_files_only=False
        )
        
        print_memory_info("After loading with traced UNet")
        
        # Test inference
        print("\n--- Testing inference ---")
        import numpy as np
        from PIL import Image
        
        # Create dummy input
        dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        # Import InferenceInput
        from riffusion.datatypes import InferenceInput, PromptInput
        
        inputs = InferenceInput(
            alpha=0.5,
            start=PromptInput(
                prompt="test prompt",
                seed=42,
                guidance=7.5,
                denoising=0.8
            ),
            end=PromptInput(
                prompt="test prompt 2", 
                seed=43,
                guidance=7.5,
                denoising=0.8
            ),
            num_inference_steps=50
        )
        
        print_memory_info("Before inference")
        
        with torch.no_grad():
            result = pipeline1.riffuse(inputs, dummy_image)
        
        print_memory_info("After inference")
        
        del pipeline1
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error with traced UNet: {e}")
    
    # Test 2: Load without traced UNet
    print("\n--- Loading without traced UNet ---")
    try:
        print_memory_info("Before loading")
        
        pipeline2 = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            use_traced_unet=False,  # Use regular UNet
            dtype=torch.float16,
            device="cuda",
            local_files_only=False
        )
        
        print_memory_info("After loading without traced UNet")
        
        del pipeline2
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error without traced UNet: {e}")
    
    # Test 3: Check for memory leaks
    print("\n=== Test 3: Memory Leak Detection ===")
    
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        try:
            print_memory_info(f"Before iteration {i+1}")
            
            pipeline = RiffusionPipeline.load_checkpoint(
                checkpoint="riffusion/riffusion-model-v1",
                use_traced_unet=True,
                dtype=torch.float16,
                device="cuda",
                local_files_only=False
            )
            
            print_memory_info(f"After loading iteration {i+1}")
            
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
            
            print_memory_info(f"After cleanup iteration {i+1}")
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
    
    # Test 4: Check attention memory usage
    print("\n=== Test 4: Attention Memory Analysis ===")
    
    try:
        pipeline = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            use_traced_unet=False,  # Need regular UNet for attention analysis
            dtype=torch.float16,
            device="cuda",
            local_files_only=False
        )
        
        unet = pipeline.unet
        
        # Count attention layers
        attention_layers = 0
        for name, module in unet.named_modules():
            if 'attn' in name and hasattr(module, 'to_q'):
                attention_layers += 1
                print(f"Found attention layer: {name}")
        
        print(f"Total attention layers: {attention_layers}")
        
        # Estimate attention memory
        # Each attention layer stores Q, K, V matrices
        # For 64x64 latents with 8 heads and 768 dim context
        attention_memory_estimate = attention_layers * 64 * 64 * 8 * 768 * 2 / (1024**3)  # GB
        print(f"Estimated attention memory: {attention_memory_estimate:.3f} GB")
        
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error in attention analysis: {e}")
    
    # Test 5: Check for gradient accumulation
    print("\n=== Test 5: Gradient Memory Check ===")
    
    try:
        pipeline = RiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            use_traced_unet=True,
            dtype=torch.float16,
            device="cuda",
            local_files_only=False
        )
        
        # Check if any parameters require gradients
        grad_params = sum(1 for p in pipeline.parameters() if p.requires_grad)
        print(f"Parameters requiring gradients: {grad_params}")
        
        # Check if any parameters are in eval mode
        pipeline.eval()
        print("Set pipeline to eval mode")
        
        print_memory_info("After setting to eval mode")
        
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error in gradient check: {e}")


def identify_potential_causes():
    """Identify potential causes of 40GB+ memory usage."""
    print("\n=== Potential Root Causes ===")
    
    causes = [
        {
            "cause": "Gradient accumulation during inference",
            "description": "PyTorch might be storing gradients even during inference",
            "solution": "Ensure torch.no_grad() is used consistently"
        },
        {
            "cause": "Attention memory explosion",
            "description": "Attention layers store Q,K,V matrices for each timestep",
            "solution": "Use attention slicing or reduce inference steps"
        },
        {
            "cause": "Model duplication",
            "description": "Multiple model instances loaded simultaneously",
            "solution": "Ensure proper cleanup between runs"
        },
        {
            "cause": "Float32 precision",
            "description": "Model loaded in float32 instead of float16",
            "solution": "Explicitly set dtype=torch.float16"
        },
        {
            "cause": "Colab GPU memory fragmentation",
            "description": "Colab's GPU memory management issues",
            "solution": "Restart runtime or use smaller batch sizes"
        },
        {
            "cause": "Cached activations",
            "description": "Activations cached for multiple forward passes",
            "solution": "Clear cache between runs with torch.cuda.empty_cache()"
        },
        {
            "cause": "Text embedding caching",
            "description": "Text embeddings cached with @functools.lru_cache",
            "solution": "Clear cache or reduce max cache size"
        },
        {
            "cause": "VAE memory leak",
            "description": "VAE encoder/decoder not properly cleaned up",
            "solution": "Ensure VAE is properly deleted"
        }
    ]
    
    for i, cause in enumerate(causes, 1):
        print(f"\n{i}. {cause['cause']}")
        print(f"   Description: {cause['description']}")
        print(f"   Solution: {cause['solution']}")


if __name__ == "__main__":
    diagnose_memory_issues()
    identify_potential_causes() 