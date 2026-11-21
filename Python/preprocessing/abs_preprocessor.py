# -*- coding: utf-8 -*-
"""Define the OpPreprocessor class.

The module used to define the class OpPreprocessor.

  Typical usage example:

  op_preprocessor = OpPreprocessor()
  op_preprocessor.generating_config_files()
"""

import inspect
import os
import pathlib
import re
from fractions import Fraction

from PIL import Image
import yaml
import numpy as np

import ffmpeg
from abs_ui import AbsUI


class AbsPreprocessor:
    """Class to generate the config files for OmniPhotos.
    """

    abs_ui = AbsUI()  # output interface static variable

    def __init__(self, args):
        """
        load all configuration and set-up runtime environment (directories)
        :param args: parameter from CLI
        """
        self.config = None
        self.show_infor_interval = 10  # the interval of output information & show images
        self.root_dir = None  # set the root directory as the root folder of config_omniphotos.yaml
        self.temp_dir = None
        self.load_config(args)

        self.input_type_list = ["image", "video"]
        self.trajectory_tool_list = ["all", "openvslam", "colmap"]
        self.image_type_list = ["panoramic", "perspective"]

        self.current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))  # current directory

        # the image folder
        # self.original_images_dir = self.root_dir / "original_images"
        # the trajectory reconstruction input images (rotated, selected)
        self.traj_input_images_dir = self.temp_dir / "trajectory_images"

        # OmniPhotos ready images from traj_input_images_dir folder
        self.op_images_dir = self.root_dir / "Input"
        self.original_image_list = []  # storing all original image filenames

        self.input_path = self.root_dir / self.config["preprocessing.input_path"]
        # check input data type
        self.input_type = self.config["preprocessing.input_type"]
        if not self.input_type in self.input_type_list:
            msg = "Input data type is {}, which is not supported.".format(self.input_type)
            self.show_info(msg, "error")

        # get the basic information for input frame info
        self.frame_width = -1
        self.frame_height = -1
        self.frame_number = -1
        self.frame_fps = -1
        self.load_origin_data_info()
        if self.input_type == "image":
            self.frame_fps = 1

        # the input image index range of OmniPhotos
        self.image_start_idx = self.config["preprocessing.frame_index_start"]
        self.image_end_idx = self.config["preprocessing.frame_index_end"]
        self.find_stable_circle = self.config["preprocessing.find_stable_circle"]
        if self.image_start_idx < 0 or self.image_start_idx > self.frame_number:
            self.show_info("preprocessing.frame_index_start set error", "error")
        if self.find_stable_circle:
            self.image_start_idx = 0
            self.image_end_idx = -1

        if self.image_end_idx == -1:
            self.image_end_idx = self.frame_number - 1
        if self.image_end_idx >= self.frame_number:
            self.show_info("preprocessing.frame_index_end set error", "error")

        if self.image_start_idx > self.image_end_idx:
            self.show_info("image_start_id is larger than image_end_idx", "error")

        self.original_filename_expression = self.config["preprocessing.original_filename_expression"]

        # create the images list and keep copy of desired indices for generation of
        # config files for later
        self.set_trajectory_images_list(self.original_filename_expression,
                                        self.config["preprocessing.frame_fixed_number"])


        # NOTE: op_image_list gets overwritten by op_preprocessor if preprocessing.find_stable_cirle is True
        self.op_image_list = self.trajectory_images_list.copy()


        self.image_list = self.op_image_list

        # check the type of input data type, input image & trajectory reconstruction tool
        self.image_type = self.config["preprocessing.colmap.input_camera_type"]
        if not self.image_type in self.image_type_list:
            msg = "Image type is {}, which is not supported.".format(self.image_type)
            self.show_info(msg, "error")

        self.trajectory_tool = self.config["preprocessing.trajectory_tool"].lower()
        if not self.trajectory_tool in self.trajectory_tool_list:
            msg = "Reconstruction tool is {}, which is not supported.".format(self.trajectory_tool)
            self.show_info(msg, "error")

        # set-up trajectory reconstruction run-time environment
        self.image_perspective_output_path = self.root_dir / "colmap_perspective"
        self.traj_output_directory_path = self.root_dir / "Capture"
        self.output_directory_path_colmap = self.traj_output_directory_path / "COLMAP"
        self.output_directory_path_ovslam = self.traj_output_directory_path / "openvslam"
        self.openvslam_config_file = self.root_dir / "config.yaml"

        # COLMAP / openvslam necessary files
        self.openvslam_output_file_list = ["map.msg", "frame_trajectory.txt"]
        self.openvslam_essential_file_list = ["map.msg", "frame_trajectory_with_filename.txt", "cameras.txt"]
        self.colmap_essential_file_list = ["points3D.txt", "images.txt", "cameras.txt"]

        # default settting
        self.cache_folder_name = None

        self.check_config()
    def set_trajectory_images_list(self, filename_expression, frame_fixed_number):
        count = self.image_end_idx - self.image_start_idx + 1
        limit = min(frame_fixed_number, count) if frame_fixed_number > 0 else count
        self.desired_frame_indices = np.linspace(
            self.image_start_idx, self.image_end_idx, num=limit, dtype=int
        ).tolist()
        self.trajectory_images_list = [
            os.path.basename(filename_expression % index)
            for index in self.desired_frame_indices
        ]

    def load_config(self, args):
        """
        Loads configuration from a YAML file and CLI.
        :param args: CLI input options
        """
        # load YAML config file
        config_file = args["config_file"]
        config_file_path = pathlib.Path(config_file).resolve(strict=True)
        if not config_file_path.is_file():
            raise ValueError(f"Expected a YAML file: {config_file_path}")

        # set the root folder
        self.root_dir = config_file_path.parents[0]
        self.temp_dir = self.root_dir / "temp"

        with config_file_path.open(encoding="utf-8-sig") as config_file:
            self.config = yaml.safe_load(config_file)
        if not isinstance(self.config, dict):
            raise ValueError(f"Expected a mapping in {config_file_path}")

        if args.get("input_path") is not None:
            input_path = pathlib.Path(args["input_path"]).resolve(strict=True)
            self.config["preprocessing.input_path"] = str(input_path)
            self.config["preprocessing.input_type"] = "image" if input_path.is_dir() else "video"

        for key, value in self.config.items():
            if value is None:
                raise ValueError(f"Preprocessing option {key} needs a value.")

        self.dir_make(self.temp_dir)

    def check_config(self):
        """
        Check the setting of variables, and set the default value.
        """
        # check the config, if not set, use the default value
        setting_list = {"preprocessing.cache_folder_name": self.cache_folder_name}
        for key in setting_list:
            try:
                item = self.config[key]
            except KeyError:
                self.show_info("{} not set, using default setting {}".format(key, setting_list[key]))

    def load_origin_data_info(self):
        """
        get the base information of input data
        """
        # load data
        if self.input_type == 'video':
            video_stream = next(iter(ffmpeg.probe(str(self.input_path), select_streams="v:0")["streams"]), None)
            if video_stream is None:
                raise ValueError(f"No video stream found in {self.input_path}")
            self.frame_width = int(video_stream["width"])
            self.frame_height = int(video_stream["height"])
            rate = video_stream.get("avg_frame_rate")
            if rate in (None, "0/0", "0/1"):
                rate = video_stream["r_frame_rate"]
            self.frame_fps = float(Fraction(rate))
            count = video_stream.get("nb_frames")
            if count in (None, "N/A"):
                result = ffmpeg.probe(str(self.input_path), select_streams="v:0", count_frames=None)
                count = result["streams"][0].get("nb_read_frames")
            if count in (None, "N/A") or int(count) < 1 or self.frame_fps <= 0:
                raise ValueError(f"Could not determine frame count and frame rate for {self.input_path}")
            self.frame_number = int(count)
        elif self.input_type == 'image':
            if not self.input_path.is_dir():
                msg = "The images input path {} is not a folder.".format(self.input_path)
                raise RuntimeError(msg)
            self.get_image_file_list(self.input_path, self.original_image_list)
            with Image.open(self.input_path / self.original_image_list[0]) as first_image:
                self.frame_width, self.frame_height = first_image.size
            self.frame_number = len(self.original_image_list)
        else:
            msg = 'Input_type error : {}'.format(self.input_type)
            self.show_info(msg, "error")

        msg = 'Input data type is {}. {}: width is {}, height is {}, FPS is {}, Frame number is {}'
        msg = msg.format(self.input_type, self.input_path, self.frame_width, \
                         self.frame_height, self.frame_fps, self.frame_number)
        self.show_info(msg)

    # def get_image_file_list(self, image_directory=None, image_file_list=None):
    def get_image_file_list(self, image_directory, image_file_list):
        """
        get all image file
        :param image_directory: the directory contain images
        :param image_file_list: image file name list, not specify it set to `self.image_list`
        """
        # # the image list is loaded from
        # if image_file_list == None and len(self.image_list) != 0:
        #     return
        # # load data from default directory to
        # if image_directory == None:
        #     image_directory = str(self.image_output_path)

        file_list = [entry.name for entry in pathlib.Path(image_directory).iterdir()
                     if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(file_list) == 0:
            msg = "There are no images in {}".format(image_directory)
            self.show_info(msg, "error")
        file_list.sort()

        # if image_file_list == None and len(self.image_list) == 0:
        #    self.image_list = file_list
        # elif image_file_list != None:
        image_file_list[:] = file_list

    def dir_make(self, directory):
        """
        check the existence of directory, if not mkdir
        :param directory: the directory path
        :type directory: str
        """
        if isinstance(directory, str):
            directory_path = pathlib.Path(directory)
        elif isinstance(directory, pathlib.Path):
            directory_path = directory
        else:
            msg = "Directory is neither str nor pathlib.Path {}".format(directory)
            self.show_info(msg, "error")
            return
        if not directory_path.exists():
            msg = "Directory {} does not exist, making a new directory" \
                .format(directory)
            directory_path.mkdir()
            self.show_info(msg)

    def instance_template(self, input_template_path, output_path, replace_dict):
        """
        Generating the *.conf file for OpenVSLAM. It read config template file and replace key words.
        :param input_template_path: the path of config file template
        :param output_path: the path of output config file
        :param replace_dict: [key-word, value], \
            NOTE: the special character in key word should be escaped, e.g. $
        """
        # read config template file
        with open(input_template_path, 'r') as file_handle:
            content = file_handle.read()

        # replace content
        for key, value in replace_dict.items():
            content = re.sub(key, value, content, flags=re.MULTILINE)

        # output config file
        with open(str(output_path), 'w') as file_handle:
            file_handle.write(content)

    def show_info(self, info_string, level="info"):
        """
        show info in console & UI
        :param info_string: information string
        :param level: the level of information, should be ["info", "error"]
        """
        self.abs_ui.show_info(info_string, level)

    def show_image(self, image_data):
        """
        show image in a windows
        :param image_data: image data in numpy array
        :type image_data: numpy.array
        """
        self.abs_ui.show_image(image_data)
