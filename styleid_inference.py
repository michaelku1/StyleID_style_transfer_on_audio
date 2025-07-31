#!/usr/bin/env python3
"""
StyleID Riffusion Inference Script

This script applies StyleID techniques to Riffusion for audio style transfer.
It loads a pretrained diffusion model and applies StyleID attention feature injection
to transfer style characteristics from one audio spectrogram to another.

Usage:
    python styleid_inference.py --content_audio path/to/content.wav --style_audio path/to/style.wav --output_dir ./output
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy
import time
import gc

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_converter import SpectrogramConverter
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import torch_util

# StyleID imports
from riffusion.styleid_riffusion_pipeline import StyleIDRiffusionPipeline


def clear_memory():
    """Clear GPU and CPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0, 0


class StyleIDRiffusionInference:
    """
    Complete inference pipeline for StyleID-enhanced Riffusion audio style transfer.
    """
    
    def __init__(
        self,
        model_path: str = "riffusion/riffusion-model-v1",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        styleid_params: Optional[Dict] = None,
        enable_memory_optimization: bool = True
    ):
        """
        Initialize the StyleID Riffusion inference pipeline.
        
        Args:
            model_path: Path to the Riffusion model checkpoint
            device: Device to run inference on
            dtype: Data type for model weights
            styleid_params: StyleID parameters (gamma, T, start_step, etc.)
            enable_memory_optimization: Whether to enable memory optimization features
        """
        self.device = torch_util.check_device(device)
        self.dtype = dtype
        self.enable_memory_optimization = enable_memory_optimization
        
        # Memory optimization settings
        if self.enable_memory_optimization:
            # Set PyTorch memory allocation settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Enable gradient checkpointing for memory efficiency
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        # Default StyleID parameters
        self.styleid_params = {
            'gamma': 0.75,      # Query preservation parameter
            'T': 1.5,           # Temperature scaling parameter
            'start_step': 49,   # Starting step for feature injection
            'use_adain_init': True,
            'use_attn_injection': True,
            'injection_layers': [6, 7, 8, 9, 10, 11]  # Attention layers for injection
        }
        if styleid_params:
            self.styleid_params.update(styleid_params)
        
        # Load the model
        print(f"Loading Riffusion model from {model_path}...")
        print(f"Memory before loading: {get_memory_usage()[0]:.2f} GB allocated")
        
        # NOTE using default (UNet2DConditionModel)
        self.pipeline = StyleIDRiffusionPipeline.load_checkpoint(
            checkpoint=model_path,
            device=self.device,
            dtype=self.dtype,
            use_traced_unet=True
        )
        
        # Apply memory optimizations
        if self.enable_memory_optimization:
            self.pipeline.unet.eval()
            self.pipeline.vae.eval()
            self.pipeline.text_encoder.eval()
            
            # Enable gradient checkpointing for UNet
            if hasattr(self.pipeline.unet, 'enable_gradient_checkpointing'):
                self.pipeline.unet.enable_gradient_checkpointing()
        
        # Initialize spectrogram converters
        params = SpectrogramParams()
        self.spectrogram_converter = SpectrogramConverter(params=params, device=self.device)
        self.image_converter = SpectrogramImageConverter(params=params, device=self.device)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Memory after loading: {get_memory_usage()[0]:.2f} GB allocated")
    
    def audio_to_spectrogram(self, audio_path: str) -> Image.Image:
        """
        Convert audio file to spectrogram image.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            PIL Image of the spectrogram
        """
        print(f"Converting audio to spectrogram: {audio_path}")
        
        # Load audio using pydub
        import pydub
        audio_segment = pydub.AudioSegment.from_file(audio_path)

        
        # Convert audio to spectrogram image
        image = self.image_converter.spectrogram_image_from_audio(audio_segment)
        
        return image
    
    def spectrogram_to_audio(self, spectrogram_image: Image.Image, output_path: str):
        """
        Convert spectrogram image back to audio.
        
        Args:
            spectrogram_image: PIL Image of the spectrogram
            output_path: Path to save the output audio
        """
        print(f"Converting spectrogram to audio: {output_path}")
        
        # Convert image back to audio
        audio_segment = self.image_converter.audio_from_spectrogram_image(spectrogram_image)
        
        # Save audio to file
        audio_segment.export(output_path, format="wav")
    
    def extract_features_ddim(
        self, 
        image: Image.Image, 
        num_steps: int = 50
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Extract features using DDIM inversion with memory optimization.
        
        Args:
            image: Input spectrogram image
            num_steps: Number of DDIM inversion steps
            
        Returns:
            Tuple of (latent, features)
        """
        print(f"Extracting features using DDIM inversion ({num_steps} steps)...")
        
        # Clear memory before feature extraction
        if self.enable_memory_optimization:
            clear_memory()
            print(f"Memory before feature extraction: {get_memory_usage()[0]:.2f} GB allocated")
        
        # Setup feature extraction
        self.pipeline.setup_feature_extraction()
        
        # NOTE Extract features + DDIM sampling with memory monitoring
        with torch.no_grad():
            latents, features = self.pipeline.extract_features_ddim(
                image=image,
                num_steps=num_steps,
                save_feature_steps=num_steps
            )
        
        if self.enable_memory_optimization:
            print(f"Memory after feature extraction: {get_memory_usage()[0]:.2f} GB allocated")
            clear_memory()
        
        return latents, features
    
    def style_transfer(
        self,
        content_audio_path: str,
        style_audio_path: str,
        output_path: str,
        prompt_start: str = "electronic music",
        prompt_end: str = "electronic music",
        alpha: float = 0.5,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        denoising_strength: float = 0.8,
        seed: int = 42
    ) -> Image.Image:
        """
        Perform StyleID-enhanced audio style transfer with memory optimization.
        
        Args:
            content_audio_path: Path to content audio file
            style_audio_path: Path to style audio file
            output_path: Path to save output audio
            prompt_start: Starting text prompt
            prompt_end: Ending text prompt
            alpha: Interpolation parameter (0-1)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            denoising_strength: Denoising strength for img2img
            seed: Random seed for generation
            
        Returns:
            PIL Image of the stylized spectrogram
        """
        print("=" * 60)
        print("STYLEID RIFFUSION AUDIO STYLE TRANSFER")
        print("=" * 60)
        
        # Initial memory check
        if self.enable_memory_optimization:
            allocated, reserved = get_memory_usage()
            print(f"Initial memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Step 1: Convert audio to spectrograms
        print("\nStep 1: Converting audio to spectrograms...")
        content_image = self.audio_to_spectrogram(content_audio_path)
        style_image = self.audio_to_spectrogram(style_audio_path)
        
        if self.enable_memory_optimization:
            clear_memory()
            allocated, reserved = get_memory_usage()
            print(f"Memory after spectrogram conversion: {allocated:.2f} GB allocated")
        
        # Step 2: Extract features using DDIM inversion
        print("\nStep 2: Extracting content features...")

        # NOTE extract features and perform DDIM inversion
        content_latents, content_features = self.extract_features_ddim(
            content_image, num_steps=num_inference_steps
        )
        
        # Clear memory between extractions
        if self.enable_memory_optimization:
            clear_memory()
            del content_latents  # Free memory immediately
        
        print("Extracting style features...")

        # NOTE extract feature and perform DDIM inversion
        style_latents, style_features = self.extract_features_ddim(
            style_image, num_steps=num_inference_steps
        )
        
        # Clear memory after both extractions
        if self.enable_memory_optimization:
            clear_memory()
            del style_latents  # Free memory immediately
        
        # Step 3: Prepare inference inputs
        print("\nStep 3: Preparing inference parameters...")
        inputs = InferenceInput(
            alpha=alpha,
            num_inference_steps=num_inference_steps,
            start=PromptInput(
                prompt=prompt_start,
                seed=seed,
                denoising=denoising_strength,
                guidance=guidance_scale
            ),
            end=PromptInput(
                prompt=prompt_end,
                seed=seed + 1,
                denoising=denoising_strength,
                guidance=guidance_scale
            )
        )
        
        # Step 4: Perform StyleID-enhanced generation
        print("\nStep 4: Performing StyleID-enhanced generation...")
        start_time = time.time()
        
        if self.enable_memory_optimization:
            allocated, reserved = get_memory_usage()
            print(f"Memory before generation: {allocated:.2f} GB allocated")
        
        # NOTE where style injection happens
        with torch.no_grad():
            stylized_image = self.pipeline.styleid_riffuse(
                inputs=inputs,
                content_image=content_image,
                style_image=style_image,
                use_adain_init=self.styleid_params['use_adain_init'],
                use_attn_injection=self.styleid_params['use_attn_injection'],
                gamma=self.styleid_params['gamma'],
                T=self.styleid_params['T'],
                start_step=self.styleid_params['start_step']
            )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Clear memory after generation
        if self.enable_memory_optimization:
            clear_memory()
            allocated, reserved = get_memory_usage()
            print(f"Memory after generation: {allocated:.2f} GB allocated")
        


        # Step 5: Convert back to audio
        print("\nStep 5: Converting to audio...")
        self.spectrogram_to_audio(stylized_image, output_path)
        
        print(f"\nStyle transfer completed!")
        print(f"Output saved to: {output_path}")
        
        return stylized_image
    
    def batch_style_transfer(
        self,
        content_dir: str,
        style_dir: str,
        output_dir: str,
        **kwargs
    ):
        """
        Perform batch style transfer on multiple audio files with memory optimization.
        
        Args:
            content_dir: Directory containing content audio files
            style_dir: Directory containing style audio files
            output_dir: Directory to save output files
            **kwargs: Additional arguments for style_transfer
        """
        import glob
        
        # Get all audio files
        content_files = glob.glob(os.path.join(content_dir, "*.wav")) + \
                       glob.glob(os.path.join(content_dir, "*.mp3"))
        style_files = glob.glob(os.path.join(style_dir, "*.wav")) + \
                     glob.glob(os.path.join(style_dir, "*.mp3"))
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Found {len(content_files)} content files and {len(style_files)} style files")
        
        for i, content_file in enumerate(content_files):
            for j, style_file in enumerate(style_files):
                content_name = os.path.splitext(os.path.basename(content_file))[0]
                style_name = os.path.splitext(os.path.basename(style_file))[0]
                
                output_file = os.path.join(
                    output_dir, 
                    f"{content_name}_styled_by_{style_name}.wav"
                )
                
                print(f"\nProcessing {i+1}/{len(content_files)} content, {j+1}/{len(style_files)} style")
                print(f"Content: {content_name}")
                print(f"Style: {style_name}")
                
                try:
                    # Clear memory before each processing
                    if self.enable_memory_optimization:
                        clear_memory()
                    
                    self.style_transfer(
                        content_audio_path=content_file,
                        style_audio_path=style_file,
                        output_path=output_file,
                        **kwargs
                    )
                except Exception as e:
                    print(f"Error processing {content_file} with {style_file}: {e}")
                    # Clear memory on error
                    if self.enable_memory_optimization:
                        clear_memory()
                    continue


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="StyleID Riffusion Audio Style Transfer")
    
    # Model parameters
    parser.add_argument("--model_path", default="riffusion/riffusion-model-v1", 
                       help="Path to Riffusion model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"], 
                       help="Data type for model weights")
    parser.add_argument("--disable_memory_optimization", action="store_true",
                       help="Disable memory optimization features")
    
    # Input/Output paths
    parser.add_argument("--content_audio", required=True, help="Path to content audio file")
    parser.add_argument("--style_audio", required=True, help="Path to style audio file")
    parser.add_argument("--output_path", required=True, help="Path to save output audio")
    
    # StyleID parameters
    # NOTE gamma for content, (1-gamma) for style
    parser.add_argument("--gamma", type=float, default=0.75, 
                       help="Query preservation parameter (0-1)")
    parser.add_argument("--T", type=float, default=1.5, 
                       help="Temperature scaling parameter")
    parser.add_argument("--start_step", type=int, default=49, 
                       help="Starting step for feature injection")
    parser.add_argument("--no_adain_init", action="store_true", 
                       help="Disable AdaIN initialization")
    parser.add_argument("--no_attn_injection", action="store_true", 
                       help="Disable attention feature injection")
    
    # Generation parameters
    parser.add_argument("--prompt_start", default="", 
                       help="Starting text prompt")
    parser.add_argument("--prompt_end", default="", 
                       help="Ending text prompt")
    parser.add_argument("--alpha", type=float, default=0.5, 
                       help="Interpolation parameter (0-1)")
    parser.add_argument("--num_inference_steps", type=int, default=50, 
                       help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, 
                       help="Classifier-free guidance scale")
    parser.add_argument("--denoising_strength", type=float, default=0.8, 
                       help="Denoising strength for img2img")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for generation")
    
    # Batch processing
    parser.add_argument("--batch_mode", action="store_true", 
                       help="Enable batch processing mode")
    parser.add_argument("--content_dir", help="Directory containing content audio files")
    parser.add_argument("--style_dir", help="Directory containing style audio files")
    parser.add_argument("--output_dir", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Setup dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Setup StyleID parameters
    styleid_params = {
        'gamma': args.gamma,
        'T': args.T,
        'start_step': args.start_step,
        'use_adain_init': not args.no_adain_init,
        'use_attn_injection': not args.no_attn_injection
    }


    # print argparse parameters 
    print('-' * 60)
    print('Argparse parameters:')
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Disable memory optimization: {args.disable_memory_optimization}")
    print(f"Content audio: {args.content_audio}")
    print(f"Style audio: {args.style_audio}")
    print(f"Prompt start: {args.prompt_start}")
    print(f"Prompt end: {args.prompt_end}")
    print(f"Alpha: {args.alpha}")
    print(f"Num inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Denoising strength: {args.denoising_strength}")
    print(f"Seed: {args.seed}")
    print(f"Output path: {args.output_path}")
    print('-' * 60)
    
    # Initialize inference pipeline
    inference = StyleIDRiffusionInference(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        styleid_params=styleid_params,
        enable_memory_optimization=not args.disable_memory_optimization
    )
    
    if args.batch_mode:
        # Batch processing
        if not all([args.content_dir, args.style_dir, args.output_dir]):
            raise ValueError("Batch mode requires --content_dir, --style_dir, and --output_dir")
        
        inference.batch_style_transfer(
            content_dir=args.content_dir,
            style_dir=args.style_dir,
            output_dir=args.output_dir,
            prompt_start=args.prompt_start,
            prompt_end=args.prompt_end,
            alpha=args.alpha,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            denoising_strength=args.denoising_strength,
            seed=args.seed
        )
    else:
        # Single file processing
        inference.style_transfer(
            content_audio_path=args.content_audio,
            style_audio_path=args.style_audio,
            output_path=args.output_path,
            prompt_start=args.prompt_start,
            prompt_end=args.prompt_end,
            alpha=args.alpha,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            denoising_strength=args.denoising_strength,
            seed=args.seed
        )


if __name__ == "__main__":
    main() 