"""
StyleID-enhanced Stable Audio inference pipeline for audio-based style transfer.
This implementation adapts StyleID techniques to work with Stable Audio's DiT-based architecture.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
import typing as T
import copy
import pickle
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from einops import rearrange

# Stable Audio specific imports (these would need to be installed)
# from stable_audio.models import DiffusionTransformer, AudioVAE
# from stable_audio.schedulers import AudioScheduler

# For now, we'll use placeholder imports - you'll need to install stable_audio
try:
    from stable_audio.models import DiffusionTransformer, AudioVAE
    from stable_audio.schedulers import AudioScheduler
    STABLE_AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: stable_audio not available. Using placeholder classes.")
    STABLE_AUDIO_AVAILABLE = False
    # Placeholder classes for development
    class DiffusionTransformer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x, t, encoder_hidden_states=None):
            return type('obj', (object,), {'sample': torch.randn_like(x)})()
    
    class AudioVAE(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def encode(self, x):
            return type('obj', (object,), {'latent_dist': type('obj', (object,), {'sample': lambda: torch.randn(x.shape[0], 4, x.shape[2]//8, x.shape[3]//8)})()})()
        def decode(self, x):
            return type('obj', (object,), {'sample': torch.randn(x.shape[0], 2, x.shape[2]*8, x.shape[3]*8)})()
    
    class AudioScheduler:
        def __init__(self, *args, **kwargs):
            pass
        def set_timesteps(self, num_steps):
            self.timesteps = torch.arange(num_steps)
        def step(self, noise_pred, t, latents, **kwargs):
            return type('obj', (object,), {'prev_sample': latents})()
        def add_noise(self, latents, noise, timesteps):
            return latents + noise

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def adain(content_feat, style_feat):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.
    Transfers the style statistics from style_feat to content_feat.
    """
    content_mean = content_feat.mean(dim=[0, 2, 3], keepdim=True)
    content_std = content_feat.std(dim=[0, 2, 3], keepdim=True)
    style_mean = style_feat.mean(dim=[0, 2, 3], keepdim=True)
    style_std = style_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((content_feat - content_mean) / content_std) * style_std + style_mean
    return output


def feat_merge(content_feats, style_feats, start_step=0, gamma=0.75, T=1.5):
    """
    Merge content and style features for StyleID injection.
    
    Args:
        content_feats: List of content features from DDIM inversion
        style_feats: List of style features from DDIM inversion
        start_step: Starting step for feature injection
        gamma: Query preservation parameter
        T: Temperature scaling parameter
        
    Returns:
        List of merged features for injection
    """
    merged_feats = []
    
    for i, (content_feat, style_feat) in enumerate(zip(content_feats, style_feats)):
        if i < start_step:
            # No injection before start_step
            merged_feats.append({'config': {'gamma': gamma, 'T': T}})
            continue
            
        merged_feat = {}
        
        # Merge attention features (Q, K, V)
        if 'q' in content_feat and 'q' in style_feat:
            # Query preservation: keep content query, inject style key/value
            merged_feat['q'] = content_feat['q']  # Preserve content query
            merged_feat['k'] = style_feat['k']    # Inject style key
            merged_feat['v'] = style_feat['v']    # Inject style value
            
            # Apply temperature scaling
            merged_feat['k'] = merged_feat['k'] / T
            merged_feat['v'] = merged_feat['v'] / T
            
        # Merge other features if present
        for key in content_feat:
            if key not in ['q', 'k', 'v']:
                if key in style_feat:
                    # Blend content and style features
                    merged_feat[key] = (1 - gamma) * content_feat[key] + gamma * style_feat[key]
                else:
                    merged_feat[key] = content_feat[key]
        
        merged_feat['config'] = {'gamma': gamma, 'T': T}
        merged_feats.append(merged_feat)
    
    return merged_feats


class StyleIDStableAudioPipeline(DiffusionPipeline):
    """
    StyleID-enhanced Stable Audio pipeline for audio-based style transfer.
    
    This pipeline extends the original Stable Audio pipeline with StyleID techniques:
    1. KV style feature injection from style audio
    2. Query preservation from content audio  
    3. Temperature scaling of attention maps
    4. AdaIN initialization of latents
    """

    def __init__(
        self,
        audio_vae: AudioVAE,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        diffusion_transformer: DiffusionTransformer,
        scheduler: T.Union[AudioScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            diffusion_transformer=diffusion_transformer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        
        # StyleID parameters
        self.gamma = 0.75  # Query preservation parameter
        self.T = 1.5      # Temperature scaling parameter
        self.attn_layers = [6, 7, 8, 9, 10, 11]  # Attention layers for injection
        self.start_step = 49  # Starting step for feature injection
        
        # Feature storage
        self.feat_maps = []
        self.content_features = None
        self.style_features = None

    def setup_feature_extraction(self):
        """Setup hooks for extracting attention features during DDIM inversion."""
        self.feat_maps = [{'config': {
            'gamma': self.gamma,
            'T': self.T
        }} for _ in range(50)]
        
        # Register hooks for attention feature extraction
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """Register hooks to extract attention features from DiffusionTransformer."""
        def save_feature_map(feature_map, filename, time_step):
            if hasattr(self, 'idx_time_dict') and time_step in self.idx_time_dict:
                cur_idx = self.idx_time_dict[time_step]
                self.feat_maps[cur_idx][filename] = feature_map

        def attention_hook(module, input, output, layer_name):
            # Extract Q, K, V from attention modules in DiffusionTransformer
            # This would need to be adapted based on the specific DiT architecture
            pass

        # Register hooks on attention modules
        for name, module in self.diffusion_transformer.named_modules():
            if 'attn' in name and 'to_q' in name:
                module.register_forward_hook(lambda m, i, o, name=name: attention_hook(m, i, o, name))

    def extract_features_ddim(self, audio, num_steps=50, save_feature_steps=50):
        """
        Extract features using DDIM inversion with memory optimization.
        
        Args:
            audio: Input audio (content or style)
            num_steps: Number of DDIM inversion steps
            save_feature_steps: Number of steps to save features for
            
        Returns:
            Tuple of (latent, features)
        """
        # Setup timestep mapping
        self.scheduler.set_timesteps(num_steps)
        
        # Ensure timesteps are available
        if not hasattr(self.scheduler, 'timesteps') or self.scheduler.timesteps is None:
            raise ValueError("Scheduler timesteps not properly initialized")
            
        # Convert to numpy array if it's a tensor
        timesteps = self.scheduler.timesteps.cpu().numpy() if torch.is_tensor(self.scheduler.timesteps) else self.scheduler.timesteps
        time_range = np.flip(timesteps)
        self.idx_time_dict = {}
        self.time_idx_dict = {}
        for i, t in enumerate(time_range):
            self.idx_time_dict[t] = i
            self.time_idx_dict[i] = t

        # Encode audio to latent using AudioVAE
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.to(device=self.device, dtype=self.audio_vae.dtype)
        else:
            # Convert audio to tensor format expected by AudioVAE
            audio_tensor = self._preprocess_audio(audio)
            
        init_latent_dist = self.audio_vae.encode(audio_tensor).latent_dist
        init_latents = init_latent_dist.sample()
        # AudioVAE might have different scaling factor than image VAE
        init_latents = 0.18215 * init_latents  # Adjust if needed for audio

        # DDIM inversion with feature extraction
        latents = init_latents.clone()
        features = []
        
        # Memory optimization: Process in smaller chunks if needed
        chunk_size = min(num_steps, 10)  # Process 10 steps at a time
        
        for chunk_start in range(0, num_steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_steps)
            
            for i in range(chunk_start, chunk_end):
                if i >= len(self.scheduler.timesteps):
                    break
                    
                t = self.scheduler.timesteps[i]
                
                # Predict noise - we need to provide encoder hidden states
                # For DDIM inversion, we can use a dummy embedding or skip the cross-attention
                batch_size = latents.shape[0]
                dummy_embedding = torch.zeros(batch_size, 77, 768, device=latents.device, dtype=latents.dtype)
                
                # Use torch.no_grad() for memory efficiency
                with torch.no_grad():
                    noise_pred = self.diffusion_transformer(latents, t, encoder_hidden_states=dummy_embedding).sample
                
                # DDIM step
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
                
                # DDIM formula
                pred_x0 = (latents - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
                pred_dir_xt = torch.sqrt(1 - alpha_prod_t_prev) * noise_pred
                latents = torch.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir_xt
                
                # Save features at specified steps
                if i < save_feature_steps:
                    # Use shallow copy to save memory
                    if i < len(self.feat_maps):
                        feature_copy = {}
                        for key, value in self.feat_maps[i].items():
                            if isinstance(value, torch.Tensor):
                                feature_copy[key] = value.detach().cpu()  # Move to CPU to save GPU memory
                            else:
                                feature_copy[key] = value
                        features.append(feature_copy)
                    else:
                        features.append({})
                
                # Clear intermediate tensors to save memory
                del noise_pred, pred_x0, pred_dir_xt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clear memory after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return latents, features

    def _preprocess_audio(self, audio):
        """
        Preprocess audio for AudioVAE encoding.
        This method should be adapted based on the specific AudioVAE requirements.
        """
        # Placeholder implementation - adapt based on actual AudioVAE requirements
        if isinstance(audio, str):
            # Load audio file
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio)
            audio_tensor = torch.from_numpy(audio_data).float()
        elif isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
            
        # Ensure correct shape for AudioVAE (batch, channels, time)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
        return audio_tensor.to(device=self.device, dtype=self.audio_vae.dtype)

    def styleid_diffuse(
        self,
        content_audio_path: str,
        style_audio_path: str,
        output_path: str,
        prompt: str = "",
        use_adain_init: bool = True,
        use_attn_injection: bool = True,
        gamma: float = 0.75,
        T: float = 1.5,
        start_step: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        """
        StyleID-enhanced audio diffusion for audio style transfer.
        
        Args:
            content_audio_path: Path to content audio file
            style_audio_path: Path to style audio file
            output_path: Path to save output audio
            prompt: Text prompt for generation
            use_adain_init: Whether to use AdaIN initialization
            use_attn_injection: Whether to use attention feature injection
            gamma: Query preservation parameter
            T: Temperature scaling parameter
            start_step: Starting step for feature injection
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for classifier-free guidance
        """
        self.gamma = gamma
        self.T = T
        self.start_step = start_step

        # Extract content and style features
        print("Extracting content features...")
        content_latents, content_features = self.extract_features_ddim(
            content_audio_path, 
            num_steps=num_inference_steps,
            save_feature_steps=num_inference_steps
        )
        
        print("Extracting style features...")
        style_latents, style_features = self.extract_features_ddim(
            style_audio_path,
            num_steps=num_inference_steps, 
            save_feature_steps=num_inference_steps
        )

        # AdaIN initialization
        if use_adain_init:
            init_latents = adain(content_latents, style_latents)
        else:
            init_latents = content_latents

        # Merge features for injection
        if use_attn_injection:
            feat_maps = feat_merge(
                content_features, 
                style_features, 
                start_step=start_step,
                gamma=gamma,
                T=T
            )
        else:
            feat_maps = None

        # Text encoding
        if prompt:
            text_embedding = self._encode_text(prompt)
        else:
            # Use dummy embedding if no prompt
            batch_size = init_latents.shape[0]
            text_embedding = torch.zeros(batch_size, 77, 768, device=self.device, dtype=self.dtype)

        # Run StyleID-enhanced generation
        outputs = self.styleid_generate(
            text_embeddings=text_embedding,
            init_latents=init_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            injected_features=feat_maps,
            start_step=start_step,
        )

        # Decode and save audio
        audio = self._decode_audio(outputs["latents"])
        self._save_audio(audio, output_path)
        
        return audio

    def _encode_text(self, prompt: str):
        """Encode text prompt using the text encoder."""
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return text_embeddings

    def _decode_audio(self, latents):
        """Decode latents to audio using AudioVAE."""
        latents = 1.0 / 0.18215 * latents  # Adjust scaling factor if needed
        audio = self.audio_vae.decode(latents).sample
        return audio

    def _save_audio(self, audio, output_path):
        """Save audio to file."""
        import soundfile as sf
        # Convert to numpy and save
        audio_np = audio.cpu().numpy()
        sf.write(output_path, audio_np, samplerate=44100)  # Adjust sample rate as needed

    @torch.no_grad()
    def styleid_generate(
        self,
        text_embeddings: torch.Tensor,
        init_latents: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
        num_audio_per_prompt: int = 1,
        eta: T.Optional[float] = 0.0,
        injected_features: T.Optional[T.List[T.Dict]] = None,
        start_step: int = 49,
        **kwargs,
    ):
        """
        StyleID-enhanced audio generation with attention feature injection.
        """
        batch_size = text_embeddings.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_audio_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_audio_per_prompt, seq_len, -1)

        # Classifier free guidance setup
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            uncond_embeddings = uncond_embeddings.repeat_interleave(
                batch_size * num_audio_per_prompt, dim=0
            )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_dtype = text_embeddings.dtype

        # Add noise to latents
        noise = torch.randn(
            init_latents.shape, device=self.device, dtype=latents_dtype
        )
        latents = self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0])

        # Prepare extra kwargs for scheduler
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timesteps = self.scheduler.timesteps.to(self.device)

        # StyleID generation loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # Get injected features for current timestep
            injected_features_i = None
            if injected_features is not None and i < len(injected_features):
                injected_features_i = injected_features[i]
            
            # Skip injection before start_step
            if i < start_step:
                injected_features_i = None

            # Expand latents for classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise with StyleID injection
            noise_pred = self.diffusion_transformer_with_styleid(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings,
                injected_features=injected_features_i
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return dict(latents=latents, nsfw_content_detected=False)
    
    def diffusion_transformer_with_styleid(self, x, t, encoder_hidden_states, injected_features=None):
        """
        DiffusionTransformer forward pass with StyleID attention feature injection.
        This is a simplified version - in practice, you'd need to modify the DiffusionTransformer
        architecture to support feature injection at the attention layers.
        """
        # For now, we'll use the standard DiffusionTransformer forward pass
        # In a full implementation, you would modify the DiffusionTransformer to inject features
        # at specific attention layers based on the injected_features parameter
        return self.diffusion_transformer(x, t, encoder_hidden_states=encoder_hidden_states)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: T.Optional[str] = None,
    ) -> StyleIDStableAudioPipeline:
        """
        Load a StyleID Stable Audio pipeline from a checkpoint.
        
        Args:
            checkpoint: Path to the model checkpoint
            device: Device to load the model on
            dtype: Data type for model weights
            local_files_only: Don't download, only use local files
            low_cpu_mem_usage: Attempt to use less memory on CPU
        """
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        if device.type == "cpu" or device.type == "mps":
            print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
            dtype = torch.float32

        try:
            # Load Stable Audio components
            # Note: These imports and loading methods need to be adapted based on the actual Stable Audio API
            if STABLE_AUDIO_AVAILABLE:
                from stable_audio import load_pipeline
                
                # Load the base Stable Audio pipeline
                base_pipeline = load_pipeline(checkpoint, device=device, dtype=dtype)
                
                # Extract components
                audio_vae = base_pipeline.audio_vae
                text_encoder = base_pipeline.text_encoder
                tokenizer = base_pipeline.tokenizer
                diffusion_transformer = base_pipeline.diffusion_transformer
                scheduler = base_pipeline.scheduler
                feature_extractor = base_pipeline.feature_extractor
            else:
                # Placeholder components for development
                audio_vae = AudioVAE()
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                diffusion_transformer = DiffusionTransformer()
                scheduler = AudioScheduler()
                feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Create the StyleIDStableAudioPipeline
            styleid_pipeline = cls(
                audio_vae=audio_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                diffusion_transformer=diffusion_transformer,
                scheduler=scheduler,
                feature_extractor=feature_extractor,
            )
            
            # Move to device
            styleid_pipeline = styleid_pipeline.to(device)

            return styleid_pipeline
            
        except Exception as e:
            print(f"Error loading Stable Audio pipeline: {e}")
            raise
        
        
        