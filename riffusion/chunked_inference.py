"""
Chunked inference for Riffusion to process long audio files in smaller segments.
"""

import typing as T
from pathlib import Path
import numpy as np
import pydub
import torch
from PIL import Image

from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.datatypes import InferenceInput
from riffusion.util import audio_util


class ChunkedRiffusionInference:
    """
    Chunked inference pipeline for processing long audio files with Riffusion.
    
    This class splits long audio files into chunks, processes each chunk with the
    Riffusion model, and then stitches the results back together.
    """
    
    def __init__(
        self,
        pipeline: RiffusionPipeline,
        chunk_duration_ms: int = 10000,  # 10 seconds
        overlap_ms: int = 1000,  # 1 second overlap
        fade_duration_ms: int = 500,  # 500ms fade
        device: str = "cuda",
    ):
        """
        Initialize the chunked inference pipeline.
        
        Args:
            pipeline: The Riffusion pipeline to use for inference
            chunk_duration_ms: Duration of each chunk in milliseconds
            overlap_ms: Overlap between chunks in milliseconds
            fade_duration_ms: Duration of fade in/out for smooth transitions
            device: Device to run inference on
        """
        self.pipeline = pipeline
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.fade_duration_ms = fade_duration_ms
        self.device = device
        
        # Initialize spectrogram converter
        params = SpectrogramParams()
        self.spectrogram_converter = SpectrogramImageConverter(params=params, device=device)
    
    def process_audio_file(
        self,
        audio_path: str,
        inputs: InferenceInput,
        init_image: Image.Image,
        mask_image: T.Optional[Image.Image] = None,
        output_path: str = None,
        progress_callback: T.Optional[T.Callable[[int, int], None]] = None,
    ) -> pydub.AudioSegment:
        """
        Process a long audio file by splitting it into chunks and processing each chunk.
        
        Args:
            audio_path: Path to the input audio file
            inputs: Inference parameters
            init_image: Initial spectrogram image for conditioning
            mask_image: Optional mask image
            output_path: Path to save the output audio (optional)
            progress_callback: Optional callback for progress updates (chunk, total_chunks)
            
        Returns:
            Processed audio segment
        """
        print(f"Loading audio file: {audio_path}")
        audio_segment = pydub.AudioSegment.from_file(audio_path)
        
        # Split audio into chunks
        chunks = self._split_audio_into_chunks(audio_segment)
        print(f"Split audio into {len(chunks)} chunks")
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            print(f"Processing chunk {i + 1}/{len(chunks)}")
            processed_chunk = self._process_chunk(chunk, inputs, init_image, mask_image)
            processed_chunks.append(processed_chunk)
        
        # Stitch chunks back together
        print("Stitching chunks together...")
        final_audio = self._stitch_chunks(processed_chunks)
        
        # Save output if path provided
        if output_path:
            print(f"Saving output to: {output_path}")
            final_audio.export(output_path, format="wav")
        
        return final_audio
    
    def _split_audio_into_chunks(self, audio: pydub.AudioSegment) -> T.List[pydub.AudioSegment]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio segment to split
            
        Returns:
            List of audio chunks
        """
        chunks = []
        start_ms = 0
        
        while start_ms < len(audio):
            end_ms = min(start_ms + self.chunk_duration_ms, len(audio))
            chunk = audio[start_ms:end_ms]
            
            # Add fade in/out to the chunk
            if start_ms > 0:  # Not the first chunk
                chunk = chunk.fade_in(self.fade_duration_ms)
            if end_ms < len(audio):  # Not the last chunk
                chunk = chunk.fade_out(self.fade_duration_ms)
            
            chunks.append(chunk)
            start_ms = end_ms - self.overlap_ms
        
        return chunks
    
    def _process_chunk(
        self,
        chunk: pydub.AudioSegment,
        inputs: InferenceInput,
        init_image: Image.Image,
        mask_image: T.Optional[Image.Image] = None,
    ) -> pydub.AudioSegment:
        """
        Process a single audio chunk with the Riffusion pipeline.
        
        Args:
            chunk: Audio chunk to process
            inputs: Inference parameters
            init_image: Initial spectrogram image
            mask_image: Optional mask image
            
        Returns:
            Processed audio chunk
        """
        # Convert audio chunk to spectrogram image
        spectrogram_image = self.spectrogram_converter.spectrogram_image_from_audio(chunk)
        
        # Run inference on the spectrogram
        processed_image = self.pipeline.riffuse(
            inputs=inputs,
            init_image=spectrogram_image,
            mask_image=mask_image,
        )
        
        # Convert processed spectrogram back to audio
        processed_audio = self.spectrogram_converter.audio_from_spectrogram_image(
            processed_image,
            apply_filters=True,
        )
        
        return processed_audio
    
    def _stitch_chunks(self, chunks: T.List[pydub.AudioSegment]) -> pydub.AudioSegment:
        """
        Stitch processed chunks back together with crossfade.
        
        Args:
            chunks: List of processed audio chunks
            
        Returns:
            Stitched audio segment
        """
        if not chunks:
            return pydub.AudioSegment.empty()
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Start with the first chunk
        result = chunks[0]
        
        # Crossfade with subsequent chunks
        for i in range(1, len(chunks)):
            # Calculate crossfade duration (half of the overlap)
            crossfade_duration = self.overlap_ms // 2
            
            # Apply crossfade
            result = result.overlay(
                chunks[i],
                position=len(result) - crossfade_duration,
                crossfade=True,
                crossfade_duration=crossfade_duration
            )
        
        return result
    
    def process_audio_with_custom_chunks(
        self,
        audio_path: str,
        inputs: InferenceInput,
        init_image: Image.Image,
        chunk_boundaries_ms: T.List[int],
        mask_image: T.Optional[Image.Image] = None,
        output_path: str = None,
    ) -> pydub.AudioSegment:
        """
        Process audio with custom chunk boundaries.
        
        Args:
            audio_path: Path to the input audio file
            inputs: Inference parameters
            init_image: Initial spectrogram image
            chunk_boundaries_ms: List of chunk boundaries in milliseconds
            mask_image: Optional mask image
            output_path: Path to save the output audio (optional)
            
        Returns:
            Processed audio segment
        """
        print(f"Loading audio file: {audio_path}")
        audio_segment = pydub.AudioSegment.from_file(audio_path)
        
        # Create chunks based on custom boundaries
        chunks = []
        for i in range(len(chunk_boundaries_ms) - 1):
            start_ms = chunk_boundaries_ms[i]
            end_ms = chunk_boundaries_ms[i + 1]
            chunk = audio_segment[start_ms:end_ms]
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} custom chunks")
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Processing custom chunk {i + 1}/{len(chunks)}")
            processed_chunk = self._process_chunk(chunk, inputs, init_image, mask_image)
            processed_chunks.append(processed_chunk)
        
        # Stitch chunks back together
        print("Stitching custom chunks together...")
        final_audio = self._stitch_chunks(processed_chunks)
        
        # Save output if path provided
        if output_path:
            print(f"Saving output to: {output_path}")
            final_audio.export(output_path, format="wav")
        
        return final_audio


def create_chunked_inference(
    model_path: str = "riffusion/riffusion-model-v1",
    chunk_duration_ms: int = 10000,
    overlap_ms: int = 1000,
    fade_duration_ms: int = 500,
    device: str = "cuda",
    **pipeline_kwargs
) -> ChunkedRiffusionInference:
    """
    Convenience function to create a chunked inference pipeline.
    
    Args:
        model_path: Path to the Riffusion model
        chunk_duration_ms: Duration of each chunk in milliseconds
        overlap_ms: Overlap between chunks in milliseconds
        fade_duration_ms: Duration of fade in/out for smooth transitions
        device: Device to run inference on
        **pipeline_kwargs: Additional arguments for pipeline loading
        
    Returns:
        Configured ChunkedRiffusionInference instance
    """
    # Load the pipeline
    pipeline = RiffusionPipeline.load_checkpoint(
        checkpoint=model_path,
        device=device,
        **pipeline_kwargs
    )
    
    # Create chunked inference
    return ChunkedRiffusionInference(
        pipeline=pipeline,
        chunk_duration_ms=chunk_duration_ms,
        overlap_ms=overlap_ms,
        fade_duration_ms=fade_duration_ms,
        device=device,
    ) 