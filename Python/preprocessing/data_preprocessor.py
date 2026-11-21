# -*- coding: utf-8 -*-
"""
implement data preparing by rotation, extraction images & video.\
    To generate the data for trajectory reconstruction
"""
import shutil
import tempfile
import pathlib

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import ffmpeg

from envmap import EnvironmentMap

from abs_preprocessor import AbsPreprocessor


class DataPreprocessor(AbsPreprocessor):
    """
    data prepare class
    """

    def __init__(self, args):
        """
        load all configuration and set-up runtime environment (directories)
        :param args: a directory containing config info
        """
        super().__init__(args)
        self.persp_forward = self.config["preprocessing.colmap.perspective_forward"]
        self.persp_fov_v = self.config["preprocessing.colmap.perspective_fov_vertical"]
        self.persp_resolution_height = \
            self.config["preprocessing.colmap.perspective_resolution_height"]
        self.persp_resolution_width = \
            self.config["preprocessing.colmap.perspective_resolution_width"]
        self.rotation = self.config["preprocessing.image_rotation"]

    def save_images(self, output_path_list, data):
        """
        save the numpy data to image
        :param output_path_list: the output path for storage images, \
            length is same as data first dimension
        :type output_path_list: list
        :param data: output image data, dimension is [x, height, width, 3]
        :type data: numpy
        """
        if len(output_path_list) != np.shape(data)[0]:
            self.show_info("The output data size is not correct.", "error")

        if not isinstance(output_path_list[0], pathlib.Path):
            self.show_info("The path type should be pathlib.", "error")

        for i in range(0, np.shape(data)[0]):  # multi-thread
            image = Image.fromarray(data[i])
            if output_path_list[i].suffix in [".jpeg", ".jpg"]:
                image.save(str(output_path_list[i]), quality=95)
            else:
                image.save(str(output_path_list[i]))

    def rotate_image_fast(self, data):
        """
        rotation panoramic image with drift matrix
        :param data: image data will be rotated, dimension is 4 [ x , width, height, 3]
        :return : weather rotated images
        """
        if [0.0, 0.0] == self.rotation:
            return data

        phi_roll_numb = self.rotation[0] / 360.0 * np.shape(data)[2]
        if phi_roll_numb == int(phi_roll_numb) and self.rotation[1] == 0.0:
            # do not need to interpolate, use numpy roll operation
            data[:] = np.roll(data, int(phi_roll_numb), axis=2)
            return data

        # interpolate (rotate) with skylibs
        return self.rotate_image(data)

    def rotate_image(self, data):
        """
        rotation images with skylibs
        :param data: image data will be rotated, dimension is 4 [ x , width, height, 3]
        :return : weather rotated the images
        """
        if [0.0, 0.0] == self.rotation:
            return data

        rotation_matrix = Rotation.from_euler(
            "xyz", [self.rotation[1], self.rotation[0], 0], degrees=True
        ).as_matrix()
        for i in range(0, np.shape(data)[0]):
            if i % self.show_infor_interval == 0:
                self.show_info("Rotating image with 'rotate_image' function, index is {}.".format(i))
            # show_image(image)
            image = data[i]
            envmap = EnvironmentMap(image, format_='latlong')
            new_image = envmap.rotate("DCM", rotation_matrix).data
            new_image = new_image.astype(np.uint8)
            data[i] = new_image
        return data

    def extract_perspective_image(self, frame_data):
        """
        extract perspective images from panoramic images. \
            The aspect ratio of image is the proportional difference between width and height.
        :param frame_data: frame data
        """
        # collect the parameters
        aspect_ratio = float(self.persp_resolution_width) / self.persp_resolution_height
        resolution = (self.persp_resolution_width, self.persp_resolution_height)
        # self.show_info("generate perspective images with rotation, {}".format(persp_forward))

        # get forward rotation matrix
        rotation_matrix = Rotation.from_euler(
            "xyz", [self.persp_forward[1], self.persp_forward[0], 0], degrees=True
        ).as_matrix()

        prespect_frame_data = \
            np.ones((np.shape(frame_data)[0], self.persp_resolution_height, \
                    self.persp_resolution_width, 3), dtype=np.uint8)
        for idx in range(0, np.shape(frame_data)[0]):
            try:
                envmap = EnvironmentMap(frame_data[idx], format_='latlong')
            except AssertionError as error:
                msg = "ERROR! extract perspective image error \" {} \"".format(error)
                print(msg)
                raise

            # get perspective images
            new_image = envmap.project(vfov=self.persp_fov_v, \
                rotation_matrix=rotation_matrix, ar=aspect_ratio, resolution=resolution)
            prespect_frame_data[idx, :, :, :] = new_image.astype(np.uint8)
        return prespect_frame_data

    def preprocess_data(self):
        """Prepare the selected frames with their original frame indices."""
        project_colmap = self.image_type == "panoramic" and self.trajectory_tool in {"colmap", "all"}
        if self.image_type == "panoramic" and self.frame_width != 2 * self.frame_height:
            raise ValueError("Panoramic input must be a 2:1 equirectangular image or video.")

        if any(pathlib.Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}
               for name in self.trajectory_images_list):
            raise ValueError("Use a .jpg, .jpeg, or .png frame filename pattern.")
        if len(set(self.trajectory_images_list)) != len(self.desired_frame_indices):
            raise ValueError("Frame filename pattern must generate unique names.")

        targets = [(self.traj_input_images_dir, "frames")]
        if project_colmap:
            targets.append((self.image_perspective_output_path, "perspective"))
        for target, _ in targets:
            if target.exists() and any(target.iterdir()):
                raise FileExistsError(
                    f"{target} already contains data. Start step 1 in a new project or resume at step 2."
                )

        self.show_info(f"Preparing {len(self.desired_frame_indices)} frames from {self.input_path}.")
        with tempfile.TemporaryDirectory(prefix="frames-", dir=self.temp_dir) as directory:
            staging = pathlib.Path(directory)
            for _, folder in targets:
                (staging / folder).mkdir()
            if self.input_type == "video":
                sources = self.extract_video_frames(staging)
            else:
                sources = [self.input_path / self.original_image_list[index]
                           for index in self.desired_frame_indices]

            for index, (source, filename) in enumerate(zip(sources, self.trajectory_images_list)):
                destination = staging / "frames" / filename
                with Image.open(source) as image:
                    if image.size != (self.frame_width, self.frame_height):
                        raise ValueError(f"Input frame dimensions changed at {source}.")
                    frame = np.array(image.convert("RGB"))[np.newaxis]
                if self.image_type == "panoramic":
                    self.rotate_image_fast(frame)
                if self.input_type == "video" and self.rotation == [0.0, 0.0] and source.suffix == destination.suffix:
                    shutil.move(source, destination)
                else:
                    self.save_images([destination], frame)
                if project_colmap:
                    perspective = self.extract_perspective_image(frame.astype(np.float64))
                    self.save_images([staging / "perspective" / filename], perspective)
                if index % self.show_infor_interval == 0:
                    self.show_info(f"Prepared {filename}.")
                    self.show_image(frame[0])

            for target, folder in targets:
                # rename preserves the fully prepared stage, including when a prior empty directory exists.
                if target.exists():
                    target.rmdir()
                (staging / folder).rename(target)

    def extract_video_frames(self, staging):
        """Decode the video once and write only the selected frames."""
        indices = self.desired_frame_indices
        first, last = indices[0], indices[-1]
        span = last - first
        if len(indices) == span + 1:
            select = f"between(n,{first},{last})"
        else:
            # An index is selected when an evenly spaced sample rounds down to it.
            intervals = len(indices) - 1
            position = f"(n-{first})"
            select = (
                f"between(n,{first},{last})*"
                f"lt(ceil({position}*{intervals}/{span}),ceil(({position}+1)*{intervals}/{span}))"
            )
        suffix = pathlib.Path(self.trajectory_images_list[0]).suffix.lower()
        pattern = staging / f"decoded-%010d{suffix}"
        options = {"start_number": 0, "fps_mode": "passthrough"}
        if suffix in {".jpg", ".jpeg"}:
            options["qscale:v"] = 2
        (ffmpeg.input(str(self.input_path))["v:0"].filter("select", select)
         .output(str(pattern), **options).global_args("-hide_banner", "-loglevel", "error").run())
        sources = sorted(staging.glob(f"decoded-*{suffix}"))
        if len(sources) != len(indices):
            raise RuntimeError(f"FFmpeg extracted {len(sources)} frames, expected {len(indices)}.")
        return sources
