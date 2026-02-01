"""Replicate Cog predictor for HOMR - Optical Music Recognition."""

import os
import shutil
import tempfile

import onnxruntime as ort
from cog import BasePredictor, Input, Path as CogPath

from homr.main import (
    ProcessingConfig,
    download_weights,
    process_image,
)
from homr.music_xml_generator import XmlGeneratorArguments
from homr.title_detection import download_ocr_weights


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download models during container build."""
        print(f"Available ONNX providers: {ort.get_available_providers()}")

        # Download GPU (FP16) models for faster inference
        print("Downloading GPU models...")
        download_weights(use_gpu_inference=True)

        # Download OCR weights for title detection
        download_ocr_weights()
        print("Model setup complete.")

    def predict(
        self,
        image: CogPath = Input(description="Sheet music image (JPG or PNG)"),
        large_page: bool = Input(
            default=False,
            description="Output MusicXML for larger page rendering",
        ),
        metronome_bpm: int = Input(
            default=None,
            description="Add metronome marking at specified BPM (optional)",
        ),
        tempo_bpm: int = Input(
            default=None,
            description="Add tempo marking at specified BPM (optional)",
        ),
    ) -> CogPath:
        """Convert sheet music image to MusicXML."""
        print(f"Running GPU inference. Providers: {ort.get_available_providers()}")

        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy input image to temp directory with a clean name
            input_path = os.path.join(tmpdir, "input.png")
            shutil.copy(str(image), input_path)

            # Configure processing with GPU inference
            config = ProcessingConfig(
                enable_debug=False,
                enable_cache=False,
                write_staff_positions=False,
                read_staff_positions=False,
                selected_staff=-1,
                use_gpu_inference=True,
            )

            xml_args = XmlGeneratorArguments(
                large_page=large_page,
                metronome=metronome_bpm,
                tempo=tempo_bpm,
            )

            # Process the image
            process_image(input_path, config, xml_args)

            # Get the output path
            output_path = os.path.join(tmpdir, "input.musicxml")

            # Copy to a persistent location for return
            final_output = "/tmp/output.musicxml"
            shutil.copy(output_path, final_output)

            return CogPath(final_output)
