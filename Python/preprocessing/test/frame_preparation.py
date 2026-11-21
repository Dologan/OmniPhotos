"""Regression check for selecting source frame indices from a video."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import yaml

import data_preprocessor


class TestFramePreparation(unittest.TestCase):
    def test_selected_output_matches_source_frames(self):
        preprocessing = Path(data_preprocessor.__file__).resolve().parent
        with TemporaryDirectory() as directory:
            project = Path(directory)
            video = project / "source.mkv"
            width, height = 16, 8
            frames = b"".join(bytes([index] * (width * height * 3)) for index in range(13))
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                 "-video_size", f"{width}x{height}", "-framerate", "30000/1001", "-i", "pipe:0",
                 "-c:v", "ffv1", "-pix_fmt", "bgr0", str(video)],
                input=frames, check=True,
            )
            config = yaml.safe_load((preprocessing / "config_omniphotos.sample.yaml").read_text())
            config.update({
                "preprocessing.step_start": 1,
                "preprocessing.step_end": 1,
                "preprocessing.input_path": video.name,
                "preprocessing.original_filename_expression": "image-%04d.png",
                "preprocessing.frame_index_start": 2,
                "preprocessing.frame_index_end": 11,
                "preprocessing.frame_fixed_number": 5,
                "preprocessing.colmap.input_camera_type": "perspective",
            })
            pipeline = project / "pipeline.yaml"
            pipeline.write_text(yaml.safe_dump(config))
            result = subprocess.run(
                [sys.executable, str(preprocessing / "main.py"), "--headless", "-c", str(pipeline)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            selected = [2, 4, 6, 8, 11]
            actual = sorted((project / "temp/trajectory_images").glob("*.png"))
            self.assertEqual([path.name for path in actual], [f"image-{index:04d}.png" for index in selected])
            for image_path, expected in zip(actual, selected):
                with Image.open(image_path) as image:
                    self.assertEqual(image.getpixel((0, 0)), (expected,) * 3)
