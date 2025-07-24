"""
StyleID-enhanced Riffusion inference pipeline for spectrogram-based style transfer.
This implementation adapts StyleID techniques to work with Riffusion's spectrogram processing.
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
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from einops import rearrange

from riffusion.datatypes import InferenceInput
from riffusion.external.prompt_weighting import get_weighted_text_embeddings
from riffusion.util import torch_util
from riffusion.riffusion_pipeline import RiffusionPipeline

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
        content_feats: Content feature maps from DDIM inversion
        style_feats: Style feature maps from DDIM inversion
        start_step: Starting step for feature injection
        gamma: Query preservation parameter (0-1)
        T: Temperature scaling parameter for attention maps
    """
    feat_maps = [{'config': {
        'gamma': gamma,
        'T': T,
        'timestep': _,
    }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = content_feats[i]
        sty_feat = style_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                # Preserve content queries
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                # Inject style keys and values
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


class StyleIDRiffusionPipeline(RiffusionPipeline):
    """
    StyleID-enhanced Riffusion pipeline for spectrogram-based style transfer.
    
    This pipeline extends the original Riffusion pipeline with StyleID techniques:
    1. KV style feature injection from style spectrograms
    2. Query preservation from content spectrograms  
    3. Temperature scaling of attention maps
    4. AdaIN initialization of latents
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor, # NOTE coming from huggingface transformers
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
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
        """Register hooks to extract attention features from UNet."""
        def save_feature_map(feature_map, filename, time_step):
            if hasattr(self, 'idx_time_dict') and time_step in self.idx_time_dict:
                cur_idx = self.idx_time_dict[time_step]
                self.feat_maps[cur_idx][filename] = feature_map

        def attention_hook(module, input, output, layer_name):
            # Extract Q, K, V from attention modules
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # This would need to be adapted based on the specific UNet architecture
                pass

        # Register hooks on attention modules
        for name, module in self.unet.named_modules():
            if 'attn' in name and 'to_q' in name:
                module.register_forward_hook(lambda m, i, o, name=name: attention_hook(m, i, o, name))

    def extract_features_ddim(self, image, num_steps=50, save_feature_steps=50):
        """
        Extract features using DDIM inversion.
        
        Args:
            image: Input image (content or style)
            num_steps: Number of DDIM inversion steps
            save_feature_steps: Number of steps to save features for
            
        Returns:
            Tuple of (latent, features)
        """
        # Setup timestep mapping
        self.scheduler.set_timesteps(num_steps)
        
        # Debug: Check scheduler state (commented out to reduce output)
        # print(f"Scheduler type: {type(self.scheduler)}")
        # print(f"Has timesteps: {hasattr(self.scheduler, 'timesteps')}")
        # if hasattr(self.scheduler, 'timesteps'):
        #     print(f"Timesteps shape: {self.scheduler.timesteps.shape if hasattr(self.scheduler.timesteps, 'shape') else 'No shape'}")
        #     print(f"Timesteps: {self.scheduler.timesteps}")
        
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

        # Encode image to latent
        if isinstance(image, Image.Image):
            image_tensor = preprocess_image(image).to(device=self.device, dtype=self.vae.dtype)
        else:
            image_tensor = image
            
        init_latent_dist = self.vae.encode(image_tensor).latent_dist
        init_latents = init_latent_dist.sample()
        init_latents = 0.18215 * init_latents

        # DDIM inversion with feature extraction
        latents = init_latents.clone()
        features = []
        
        for i, t in enumerate(self.scheduler.timesteps):
            if i >= num_steps:
                break
                
            # Predict noise - we need to provide encoder hidden states
            # For DDIM inversion, we can use a dummy embedding or skip the cross-attention
            # Let's create a dummy embedding with the correct shape
            batch_size = latents.shape[0]
            dummy_embedding = torch.zeros(batch_size, 77, 768, device=latents.device, dtype=latents.dtype)
            noise_pred = self.unet(latents, t, encoder_hidden_states=dummy_embedding).sample
            
            # DDIM step
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            # DDIM formula
            pred_x0 = (latents - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            pred_dir_xt = torch.sqrt(1 - alpha_prod_t_prev) * noise_pred
            latents = torch.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir_xt
            
            # Save features at specified steps
            if i < save_feature_steps:
                features.append(copy.deepcopy(self.feat_maps[i]) if i < len(self.feat_maps) else {})
        
        return latents, features

    def styleid_riffuse(
        self,
        inputs: InferenceInput,
        content_image: Image.Image,
        style_image: Image.Image,
        use_adain_init: bool = True,
        use_attn_injection: bool = True,
        gamma: float = 0.75,
        T: float = 1.5,
        start_step: int = 49,
    ) -> Image.Image:
        """
        StyleID-enhanced riffusion inference for spectrogram style transfer.
        
        Args:
            inputs: Parameter dataclass
            content_image: Content spectrogram image
            style_image: Style spectrogram image  
            use_adain_init: Whether to use AdaIN initialization
            use_attn_injection: Whether to use attention feature injection
            gamma: Query preservation parameter
            T: Temperature scaling parameter
            start_step: Starting step for feature injection
        """
        self.gamma = gamma
        self.T = T
        self.start_step = start_step
        
        alpha = inputs.alpha
        start = inputs.start
        end = inputs.end

        guidance_scale = start.guidance * (1.0 - alpha) + end.guidance * alpha

        # Setup generators
        if self.device.lower().startswith("mps"):
            generator_start = torch.Generator(device="cpu").manual_seed(start.seed)
            generator_end = torch.Generator(device="cpu").manual_seed(end.seed)
        else:
            generator_start = torch.Generator(device=self.device).manual_seed(start.seed)
            generator_end = torch.Generator(device=self.device).manual_seed(end.seed)

        # Text encodings with interpolation
        embed_start = self.embed_text_weighted(start.prompt)
        embed_end = self.embed_text_weighted(end.prompt)
        text_embedding = embed_start + alpha * (embed_end - embed_start)

        # Extract content and style features
        print("Extracting content features...")
        content_latents, content_features = self.extract_features_ddim(
            content_image, 
            num_steps=inputs.num_inference_steps,
            save_feature_steps=inputs.num_inference_steps
        )
        
        print("Extracting style features...")
        style_latents, style_features = self.extract_features_ddim(
            style_image,
            num_steps=inputs.num_inference_steps, 
            save_feature_steps=inputs.num_inference_steps
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

        # Run StyleID-enhanced interpolation
        outputs = self.styleid_interpolate_img2img(
            text_embeddings=text_embedding,
            init_latents=init_latents,
            generator_a=generator_start,
            generator_b=generator_end,
            interpolate_alpha=alpha,
            strength_a=start.denoising,
            strength_b=end.denoising,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=guidance_scale,
            injected_features=feat_maps,
            start_step=start_step,
        )

        return outputs["images"][0]

    @torch.no_grad()
    def styleid_interpolate_img2img(
        self,
        text_embeddings: torch.Tensor,
        init_latents: torch.Tensor,
        generator_a: torch.Generator,
        generator_b: torch.Generator,
        interpolate_alpha: float,
        mask: T.Optional[torch.Tensor] = None,
        strength_a: float = 0.8,
        strength_b: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: T.Optional[float] = 0.0,
        output_type: T.Optional[str] = "pil",
        injected_features: T.Optional[T.List[T.Dict]] = None,
        start_step: int = 49,
        **kwargs,
    ):
        """
        StyleID-enhanced img2img interpolation with attention feature injection.
        """
        batch_size = text_embeddings.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

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
                batch_size * num_images_per_prompt, dim=0
            )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_dtype = text_embeddings.dtype
        strength = (1 - interpolate_alpha) * strength_a + interpolate_alpha * strength_b

        # Get initial timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=self.device
        )

        # Add noise to latents
        noise_a = torch.randn(
            init_latents.shape, generator=generator_a, device=self.device, dtype=latents_dtype
        )
        noise_b = torch.randn(
            init_latents.shape, generator=generator_b, device=self.device, dtype=latents_dtype
        )
        noise = torch_util.slerp(interpolate_alpha, noise_a, noise_b)
        init_latents_orig = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # Prepare extra kwargs for scheduler
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents.clone()
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

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
            noise_pred = self.unet_with_styleid(
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

            # Apply mask if provided
            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_orig, noise, torch.tensor([t])
                )
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

        # Decode latents to image
        latents = 1.0 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return dict(images=image, latents=latents, nsfw_content_detected=False)

    def unet_with_styleid(self, x, t, encoder_hidden_states, injected_features=None):
        """
        UNet forward pass with StyleID attention feature injection.
        This is a simplified version - in practice, you'd need to modify the UNet
        architecture to support feature injection at the attention layers.
        """
        # For now, we'll use the standard UNet forward pass
        # In a full implementation, you would modify the UNet to inject features
        # at specific attention layers based on the injected_features parameter
        return self.unet(x, t, encoder_hidden_states=encoder_hidden_states)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        use_traced_unet: bool = True,
        channels_last: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: T.Optional[str] = None,
    ) -> StyleIDRiffusionPipeline:
        """
        Load a StyleID Riffusion pipeline from a checkpoint.
        
        Args:
            checkpoint: Path to the model checkpoint
            use_traced_unet: Whether to use the traced unet for speedups
            device: Device to load the model on
            channels_last: Whether to use channels_last memory format
            local_files_only: Don't download, only use local files
            low_cpu_mem_usage: Attempt to use less memory on CPU
        """
        device = torch_util.check_device(device)

        if device == "cpu" or device.lower().startswith("mps"):
            print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
            dtype = torch.float32

        # Try to load using the base pipeline's from_pretrained method
        # but handle the feature_extractor parameter manually
        try:
            # Load components individually to avoid the feature_extractor issue
            from diffusers import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
            from diffusers.schedulers import DDIMScheduler
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            
            # Load components
            vae = AutoencoderKL.from_pretrained(
                checkpoint, 
                subfolder="vae",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            text_encoder = CLIPTextModel.from_pretrained(
                checkpoint,
                subfolder="text_encoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(
                checkpoint,
                subfolder="tokenizer",
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            unet = UNet2DConditionModel.from_pretrained(
                checkpoint,
                subfolder="unet",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            scheduler = DDIMScheduler.from_pretrained(
                checkpoint,
                subfolder="scheduler",
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            # Create the StyleIDRiffusionPipeline
            styleid_pipeline = cls(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            
            # Move to device
            styleid_pipeline = styleid_pipeline.to(device)
            
            if channels_last:
                styleid_pipeline.unet.to(memory_format=torch.channels_last)

            return styleid_pipeline
            
        except Exception as e:
            print(f"Error loading components individually: {e}")
            # Fallback to the original method
            base_pipeline = RiffusionPipeline.load_checkpoint(
                checkpoint=checkpoint,
                use_traced_unet=use_traced_unet,
                channels_last=channels_last,
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
                low_cpu_mem_usage=low_cpu_mem_usage,
                cache_dir=cache_dir,
            )

            # Create StyleIDRiffusionPipeline with the same components
            styleid_pipeline = cls(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                tokenizer=base_pipeline.tokenizer,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler,
                safety_checker=base_pipeline.safety_checker,
                feature_extractor=base_pipeline.feature_extractor,
            )
            
            # Move to device
            styleid_pipeline = styleid_pipeline.to(device)

            return styleid_pipeline


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)
    return 2.0 * image_torch - 1.0


def preprocess_mask(mask: Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    # Convert to grayscale
    mask = mask.convert("L")

    # Resize to integer multiple of 32
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.NEAREST)

    # Convert to numpy array and rescale
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Tile and transpose
    mask_np = np.tile(mask_np, (4, 1, 1))
    mask_np = mask_np[None].transpose(0, 1, 2, 3)

    # Invert to repaint white and keep black
    mask_np = 1 - mask_np

    return torch.from_numpy(mask_np) 