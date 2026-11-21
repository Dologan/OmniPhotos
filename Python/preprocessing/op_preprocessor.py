# -*- coding: utf-8 -*-

"""Define the OpPreprocessor class.

The module use to define the class OpPreprocessor.

  Typical usage example:

  op_preprocessor = OpPreprocessor()
  op_preprocessor.generating_config_files()
"""

import os
import copy
import pathlib
import hashlib
import shutil
import re
import csv
import subprocess
import concurrent.futures
import json

import yaml
import msgpack

import numpy as np
from PIL import Image
from skimage import transform, io

from abs_preprocessor import AbsPreprocessor
import circleselector
import time

class OpPreprocessor(AbsPreprocessor):
    """Class to generate the config files for OmniPhotos.
    """

    def __init__(self, args):
        """
        load all configuration and set-up runtime environment (directories)
        :param args: parameter from CLI
        """
        super().__init__(args)
        # OmniPhotos configuration file template
        self.omniphotos_config_template_path = \
            os.path.dirname(__file__) + '/template/config.template.yaml'

        self.dataset_name = os.path.basename(str(self.root_dir))

        # the configuration files folder
        self.output_config_file_path = self.root_dir / "Config"
        self.dir_make(self.output_config_file_path)

        # the path & parameters of OmniPhotos executable file
        self.omniphotos_path = os.path.join(os.path.dirname(__file__) , self.config["preprocessing.omniphotos.execution_file_path"])
        self.circle_radius = self.config["preprocessing.omniphotos.circle_radius"]
        if self.circle_radius is None or self.circle_radius <= 0:
            self.show_info("The circle radius {} are wrong.".format(self.circle_radius),"error")

        self.op_exe_config_file = self.config["preprocessing.omniphotos.yaml_config_path"]

        self.op_input_downsample_scalar = self.config["preprocessing.omniphotos.downsample_scalar"]
        scale_x, scale_y = self.op_input_downsample_scalar
        if (scale_x, scale_y) == (-1.0, -1.0):
            scale_x = scale_y = 1.0
        if scale_x != scale_y or not 0 < scale_x <= 1:
            raise ValueError("Input downsampling requires an equal scale in (0, 1] or [-1, -1].")
        self.op_input_frame_height = max(1, int(self.frame_height * scale_y))
        self.op_input_frame_width = max(1, int(self.frame_width * scale_x))
        if self.image_type == "panoramic":
            self.op_input_frame_width = 2 * self.op_input_frame_height
        self.show_info(f"Working images: {self.op_input_frame_width}x{self.op_input_frame_height}.")

        self.of_downsample_scalar = self.config["preprocessing.of.downsample_scalar"]
        if self.of_downsample_scalar not in ([-1.0, -1.0], [1.0, 1.0], [0.5, 0.5]):
            raise ValueError("Optical flow supports full or half resolution.")

        self.cache_dir = self.root_dir / "Cache"
        self.dir_make(self.cache_dir)
        self.capture_data_dir = self.root_dir / "Capture"
        self.cache_folder_name = self.config.get("preprocessing.cache_folder_name")
        self.cache_folder_path_flownet2 = None
        self.op_preprocessing_cache_dir = None
        self.circle_selector_start_idx = 0
        self.circle_selector_end_idx = -1

    def generating_cache_filename(self):
        """Preserve historical names for known resolutions and support new camera sizes."""
        prefix = self.cache_folder_name
        if prefix is None:
            known = {
                (5760, 2880): "5_7K", (3840, 1920): "4k", (3008, 1504): "3k",
                (2880, 1440): "2_9K", (1920, 960): "2k", (1504, 752): "1_5k",
                (960, 480): "1k", (480, 240): "0_5k",
            }
            width, height = self.op_input_frame_width, self.op_input_frame_height
            flow_scale = abs(self.of_downsample_scalar[0])
            flow_width, flow_height = int(width * flow_scale), int(height * flow_scale)
            images = known.get((width, height), f"{width}x{height}")
            flow = known.get((flow_width, flow_height), f"{flow_width}x{flow_height}")
            prefix = f"{len(self.op_image_list)}-{images}-{flow}"
        if pathlib.Path(prefix).name != prefix or prefix in {"", ".", ".."}:
            raise ValueError("Cache folder prefix must be a single directory name.")
        self.op_preprocessing_cache_dir = f"{prefix}-DIS"
        self.cache_folder_path_flownet2 = f"{prefix}-flownet2"


    def generating_config_files(self):
        """
        generate the necessary files for OmniPhotos
        """

        # 1) select the best circle
        # do this here so we copy only the images we need to the Input image directory.
        if self.find_stable_circle:
            if self.trajectory_tool == "colmap":
                raise ValueError("Automatic circle selection requires OpenVSLAM. Trim the COLMAP input to one loop.")
            self.openvslam_select_stable_circle()
        self.generating_cache_filename()

        # create folder for downsampled input images
        self.dir_make(self.op_images_dir)

        expected_images = set(self.op_image_list)
        existing_images = {path.name for path in self.op_images_dir.iterdir() if path.is_file()}
        if not expected_images.issubset(existing_images):
            self.show_info(f"Generating working images in {self.op_images_dir}.")
            self.omniphotos_generate_input_images()

        # 3) convert and generate the trajectory files as OmniPhotos request format
        self.show_info("Checking for necessary files.")
        #  check the OpenVSLAM necessary file
        if self.trajectory_tool == "openvslam" or self.trajectory_tool == "all":
            self.dir_make(self.output_directory_path_ovslam)

            # check whether need the format conversion, if files exist do not convert 
            openvslam_need_convertion = False
            for term in self.openvslam_essential_file_list:
                openvslam_need_convertion = openvslam_need_convertion or (not (self.output_directory_path_ovslam / term).exists())
            # generate the configuration files for OpenVSLAM
            if openvslam_need_convertion:
                # check the openvslam output files
                for term in self.openvslam_output_file_list:
                    if not (self.output_directory_path_ovslam / term).exists():
                        self.show_info("The OpenVSLAM output file {} does not exist.".format(term), "error")

                # convert the openvslam output "frame_trajectory.txt" format
                self.show_info("Converting the OpenVSLAM YAML file to an OmniPhotos compatible data file.")

                self.openvslam_convert_traj_file(
                    str(self.output_directory_path_ovslam), str(self.openvslam_config_file))
                # generate the cameras.txt used by OmniPhotos
                self.openvslam_create_camera_file(
                    self.output_directory_path_ovslam / "cameras.txt")
                # convert the *.msg file to json, and output point cloud to obj file
                self.openvslam_parser_msg(
                    str(self.output_directory_path_ovslam / "map.msg"))
                # backup the unnecessary files, save reconstruction config file
                self.show_info("save reconstruction config file")
                shutil.copyfile(str(self.openvslam_config_file), str(
                    self.output_directory_path_ovslam / "config.yaml"))
            else:
                self.show_info("The OpenVSLAM essential file already exists. Skipping OpenVSLAM file conversion.")

        if self.trajectory_tool in {"colmap", "all"}:
            for filename in self.colmap_essential_file_list:
                source = self.output_directory_path_colmap / "full-model-ba-txt" / filename
                destination = self.output_directory_path_colmap / filename
                if not destination.is_file():
                    if not source.is_file():
                        raise FileNotFoundError(f"Missing COLMAP reconstruction file: {source}")
                    shutil.copyfile(source, destination)

        # 5) generate the config file and images for OmniPhotos
        self.show_info("Generating the config files for OmniPhotos viewer & preprocessing.", "info")
        self.omniphotos_generate_viewer_config()

        cache = self.cache_dir / self.op_preprocessing_cache_dir
        if not cache.exists():
            self.show_info("Running native OmniPhotos preprocessing.")
            self.omniphotos_preprocess()
        if not (cache / "Cameras.csv").is_file() or not any(cache.glob("PreprocessingSetup-*.json")):
            raise RuntimeError(f"Cache {cache} is incomplete. Use a new cache folder name or move it before retrying.")

        # 7) initial flownet2 run-time environment
        if self.config["preprocessing.step_end"] in {-1, 4}:
            self.show_info("Preparing the optional FlowNet2 cache.")
            self.omniphotos_flownet2_init()

    def openvslam_select_stable_circle(self):

        self.show_info("Finding stable circle. This may take a few minutes.")
        self.show_info("Time started was " + time.ctime())
        cached_res = os.path.join(self.capture_data_dir, "best_intervals.json")
        if os.path.exists(cached_res):
            self.show_info("Found cached results at " + cached_res)
            intervals = circleselector.datatypes.PointDict()
            intervals.fromJSON(cached_res)

        else:
            points = circleselector.loader.load_file(
                os.path.join(self.output_directory_path_ovslam, "frame_trajectory.txt"))
            metrics = circleselector.metrics.Metrics(points,errors=["endpoint_error", "perimeter_error", "flatness_error",
                                                                "pairwise_distribution"])
            metrics.run()
            self.show_info("Time of completion was " + time.ctime())
            data = circleselector.datatypes.PointDict(metrics.point_dcts)
            intervals = data.find_local_minima(len(points))
            circleselector.plotting_utils.plot_heatmap(data,len(points),
                                                       dot_coords=np.array(intervals.get("interval")),
                                                       show=False,
                                                       save_to=os.path.join(self.capture_data_dir,"best_intervals.png"))

            self.show_info(str(len(intervals)) + " valid intervals found.")
            intervals = circleselector.metrics.calc(points,
                                                    point_dcts=intervals,
                                                    errors=["ssim","psnr"],
                                                    mp=False,
                                                    dataset_path=self.root_dir,
                                                    rel_input_image_path="temp/trajectory_images")
            intervals.toJSON(cached_res)
            intervals.split_interval().toCSV(os.path.join(self.capture_data_dir,"best_intervals.csv"))
        if len(intervals) == 0:
            self.show_info("No intervals found.","error")

        stable_circle = intervals.find_best_interval()["interval"]
        self.circle_selector_start_idx, self.circle_selector_end_idx= stable_circle[0], stable_circle[1]
        self.op_image_list = self.trajectory_images_list[self.circle_selector_start_idx:self.circle_selector_end_idx]

    def omniphotos_generate_input_images(self):
        """
        generate the images for OmniPhotos input.
        copy & downsample the image from traj_input_images_dir to op_images_dir
        """

        if self.op_input_frame_height == self.frame_height and self.op_input_frame_width == self.frame_width:
            for idx, image_filename in enumerate(self.op_image_list):
                src_image_path = self.traj_input_images_dir / image_filename
                dest_image_path = self.op_images_dir / image_filename
                if idx % self.show_infor_interval == 0:
                    self.show_info("Copy image from {} to {}.".format(str(src_image_path), str(dest_image_path)))
                shutil.copyfile(src_image_path, dest_image_path)
        else:
            self.show_info(\
                "Downsampling the input images to {} x {}".format(self.op_input_frame_height, self.op_input_frame_width))
            self.omniphotos_downsample_input_images()

    def omniphotos_downsample_input_images(self):
        """Downsample only the source frames selected for the output dataset."""
        output_shape = (self.op_input_frame_height, self.op_input_frame_width, 3)

        def downsample(filename):
            pixels = io.imread(self.traj_input_images_dir / filename)
            pixels = transform.resize(pixels, output_shape, preserve_range=True, anti_aliasing=True)
            pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
            path = self.op_images_dir / filename
            image = Image.fromarray(pixels)
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                image.save(path, quality=95)
            else:
                image.save(path)

        # Bound the number of simultaneously decoded source panoramas and float work buffers.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            for _ in executor.map(downsample, self.op_image_list):
                pass

    def omniphotos_logging(self):
        """
        Output preprocessing log file to the cache directory.
        The log info including at binary HASH code, commit info, etc.
        """
        logging_str = ""

        # 0) OmniPhotos bin path
        logging_str = logging_str + "The OmniPhotos bin file path: {} \n".format(self.omniphotos_path)

        # 1) HASH code
        BUF_SIZE = 1048576  # read 1MB chunks
        md5 = hashlib.md5()
        sha1 = hashlib.sha1()

        if not os.path.exists(self.omniphotos_path):
            raise RuntimeError("{} do not exist".format(self.omniphotos_path))

        with open(self.omniphotos_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
                sha1.update(data)

        logging_str = logging_str + "MD5:" +  md5.hexdigest() + "\n"
        logging_str = logging_str + "SHA1:" + sha1.hexdigest() + "\n"
        return logging_str

    def omniphotos_preprocess(self):
        """
        OmniPhotos preprocessing program composing with following steps:
        """
        # 1) generate the configuration file for OmniPhotos binary preprocess
        if self.trajectory_tool == "colmap" or self.trajectory_tool == "all":
            modelfiles = "modelFiles.txt"
            self.show_info("OmniPhotos preprocessing with COLMAP trajectory.")
        elif self.trajectory_tool == "openvslam":
            modelfiles = "modelFiles.openvslam"
            self.show_info(
                "OmniPhotos preprocessing with OpenVSLAM trajectory.")
        else:
            self.show_info("Trajectory reconstruction have error.")

        if self.op_exe_config_file == "None":
            parameters = {
                r'\$circle_radius\$': str(self.circle_radius),
                r'\$dataset_name\$': self.dataset_name,
                r'\$cameras_number\$': str(len(self.op_image_list)),
                r'\$cache_folder_dis\$': self.op_preprocessing_cache_dir,
                r'\$cache_folder_flownet2\$': self.cache_folder_path_flownet2,
                r'\$sfm_file_name\$': modelfiles,
                r'\$load_3d_points\$': str(1),
                r'\$image_filename\$': self.original_filename_expression,
                r'\$load_sparse_point_cloud\$': str(1),
                r'\$image_width\$': str(self.op_input_frame_width),
                r'\$image_height\$': str(self.op_input_frame_height),
                r'\$first_frame\$': str(0),
                r'\$last_frame\$': str(len(self.op_image_list) - 1),
                r'\$image_fps\$': str(self.frame_fps),
                r'\$change_basis\$': str(0),
                r'\$equirectangular\$': str(int(self.image_type == 'panoramic')),
                r'\$intrinsic_scale\$': str(self.op_input_frame_width / self.frame_width if self.image_type == 'perspective' else 1.0),
                r'\$downsample_flow\$': str(0) if self.of_downsample_scalar[0] == -1 or self.of_downsample_scalar[0] == 1 else str(1),
                r'\$compute_optical_flow\$': str(1),
                r'\$use_point_cloud\$': str(1),
            }
            self.op_exe_config_file = str(
                self.output_config_file_path / "config-preprocessing.yaml")
            self.instance_template(self.omniphotos_config_template_path, self.op_exe_config_file,
                                   parameters)

        # 2) call OmniPhotos to do preprocess
        if not os.path.exists(self.omniphotos_path):
            self.show_info("OmniPhotos Preprocessing binary {} not found.".format(self.omniphotos_path), "error")

        subprocess.run([self.omniphotos_path, "-f", self.op_exe_config_file], check=True)

        # 3) output log to cache file
        log_str = self.omniphotos_logging()
        log_file_path = self.cache_dir / "preprocessing.log"
        with open(str(log_file_path), "w") as file_handle:
            file_handle.write(log_str)

        self.show_info("output preprocessing log to {}".format(log_file_path))

    def omniphotos_generate_viewer_config(self):
        """
        generate the necessary config file (*.yaml etc) for OmniPhotos viewer
        """
        # 1) generate OmniPhotos config files
        # assign configuration file variables
        omniphotos_config_info_common = {
            r'\$circle_radius\$': str(self.circle_radius),
            r'\$dataset_name\$': self.dataset_name,
            r'\$image_width\$': str(self.op_input_frame_width),
            r'\$image_height\$': str(self.op_input_frame_height),
            r'\$first_frame\$': str(0),
            r'\$last_frame\$': str(len(self.op_image_list) - 1),
            r'\$image_fps\$': str(self.frame_fps),
            r'\$cameras_number\$': str(len(self.op_image_list)),
            r'\$image_filename\$': self.original_filename_expression,
            r'\$cache_folder_dis\$': self.op_preprocessing_cache_dir,
            r'\$cache_folder_flownet2\$': self.cache_folder_path_flownet2,
            r'\$load_3d_points\$': str(1),
            r'\$load_sparse_point_cloud\$': str(1),
            r'\$change_basis\$': str(0),
            r'\$equirectangular\$': str(int(self.image_type == 'panoramic')),
                r'\$intrinsic_scale\$': str(self.op_input_frame_width / self.frame_width if self.image_type == 'perspective' else 1.0),
            r'\$downsample_flow\$': str(0) if self.of_downsample_scalar[0] == -1 or self.of_downsample_scalar[0] == 1 else str(1),
            r'\$compute_optical_flow\$': str(0),
            r'\$use_point_cloud\$': str(1),
        }

        # read OmniPhotos config file template
        with open(self.omniphotos_config_template_path, 'r') as file_handle:
            content = file_handle.read()

        # 2) output OmniPhotos config file for openvslam
        if self.trajectory_tool == "all" or self.trajectory_tool == 'openvslam':
            omniphotos_config_info = copy.copy(omniphotos_config_info_common)
            content_output = copy.copy(content)
            # output the config.yaml file for openvslam
            output_config_file_name = "config-viewer-openvslam.yaml"
            output_model_file_name = "modelFiles.openvslam"
            omniphotos_config_info[r'\$trajectory_tool\$'] = self.trajectory_tool
            omniphotos_config_info[r'\$sfm_file_name\$'] = output_model_file_name

            for key, value in omniphotos_config_info.items():
                content_output = re.sub(
                    key, value, content_output, flags=re.MULTILINE)
            with open(str(self.output_config_file_path / output_config_file_name), 'w') as file_handle:
                file_handle.write(content_output)

            # output model file modelFiles.txt content & check the file
            openvslam_model_file_list = ["../Capture/openvslam/cameras.txt",\
                                                   "../Capture/openvslam/frame_trajectory_with_filename.txt",\
                                                   "../Capture/openvslam/map.msg"]
            for filename in openvslam_model_file_list:
                if not (self.output_config_file_path / filename).is_file():
                    raise FileNotFoundError(f"Missing OpenVSLAM input: {filename}")
            model_file_content_openvslam = "\n".join(openvslam_model_file_list)
            with open(str(self.output_config_file_path / output_model_file_name), 'w') as file_handle:
                file_handle.write(model_file_content_openvslam)

        # 3) output OmniPhotos config file for colmap
        if self.trajectory_tool == "all" or self.trajectory_tool == 'colmap':
            omniphotos_config_info = copy.copy(omniphotos_config_info_common)
            content_output = copy.copy(content)

            # output the config file for colmap
            output_config_file_name = "config-viewer-colmap.yaml"
            output_model_file_name = "modelFiles.txt"
            omniphotos_config_info[r'\$trajectory_tool\$'] = self.trajectory_tool
            omniphotos_config_info[r'\$sfm_file_name\$'] = output_model_file_name

            for key, value in omniphotos_config_info.items():
                content_output = re.sub(
                    key, value, content_output, flags=re.MULTILINE)
            with open(str(self.output_config_file_path / output_config_file_name), 'w') as file_handle:
                file_handle.write(content_output)

            # output model file modelFiles.txt content
            colmap_model_file_list = ["../Capture/COLMAP/cameras.txt",\
                                  "../Capture/COLMAP/images.txt",\
                                  "../Capture/COLMAP/points3D.txt"]
            for filename in colmap_model_file_list:
                if not (self.output_config_file_path / filename).is_file():
                    raise FileNotFoundError(f"Missing COLMAP input: {filename}")
            model_file_content_colmap = "\n".join(colmap_model_file_list)
            with open(str(self.output_config_file_path / output_model_file_name), 'w') as file_handle:
                file_handle.write(model_file_content_colmap)

    def omniphotos_flownet2_init(self):
        """
        init the folder for the flownet2 output fitting to OmniPhotos request
        """
        # 1) create folder of flownet2
        self.dir_make(self.cache_dir / self.cache_folder_path_flownet2)

        # 2) copy the essential files
        src_folder = self.cache_dir / self.op_preprocessing_cache_dir
        dest_folder = self.cache_dir / self.cache_folder_path_flownet2
        for term in src_folder.iterdir():
            if term.suffix in [".flo", ".floss"]:
                continue
            shutil.copyfile(str(term), str(dest_folder / term.name))

        # 3) rename the optical flow path in Cameras.csv to the flownet2 flo path
        cameras_csv_all_text = None
        with open(str(src_folder / "Cameras.csv"), "r") as file_handle:
            cameras_csv_all_text = file_handle.read()
        cameras_csv_all_text = cameras_csv_all_text.replace(str(self.op_preprocessing_cache_dir), str(self.cache_folder_path_flownet2))
        #cameras_csv_all_text = cameras_csv_all_text.replace(".floss", ".flo")
        with open(str(dest_folder / "Cameras.csv"), "w") as file_handle:
            file_handle.write(cameras_csv_all_text)

    def openvslam_convert_traj_file(self,
                                    openvslam_output_directory,
                                    openvslam_config_file_path):
        """
        transform the original trajectory file format to OmniPhotos format
        :param openvslam_output_directory: *.msg *.txt output directory
        :param openvslam_config_file_path: *.yaml config file for openvslam
        """
        if not os.path.exists(openvslam_output_directory):
            self.show_info("The folder {} does not exist.".format(openvslam_output_directory), "error")
        if not os.path.exists(openvslam_config_file_path):
            self.show_info("The openvslam config file {} does not exist, please put it in the root folder of the "
                           "dataset."\
                .format(openvslam_config_file_path), "error")

        openvslam_output_directory_path = pathlib.Path(
            openvslam_output_directory)
        openvslam_traj_file_path = openvslam_output_directory_path / "frame_trajectory.txt"
        openvslam_traj_file_path_output = openvslam_output_directory_path / "frame_trajectory_with_filename.txt"

        output_csv_header = ["#index", "filename", "trans_wc.x", "trans_wc.y", "trans_wc.z", "quat_wc.x", "quat_wc.y",
                             "quat_wc.z", "quat_wc.w"]

        # load openvslam YAML config file get FPS
        with open(openvslam_config_file_path, "r") as yaml_config_file_handle:
            ovslam_config_str = yaml_config_file_handle.read()
        ovslam_config_data = yaml.safe_load(ovslam_config_str)
        ovslam_fps = ovslam_config_data["Camera.fps"]

        # load frame_trajectory.txt
        yaml_traj_file_handle = open(openvslam_traj_file_path)
        yaml_traj_csv_file_handle = csv.reader(
            yaml_traj_file_handle, delimiter=' ', quotechar='|')

        # map timestamp to filename & output newfile with filename
        output_counter= 0
        with open(openvslam_traj_file_path_output, 'w', newline='') as yaml_traj_file_handle_output:
            yaml_traj_csv_file_handle_output = csv.writer(yaml_traj_file_handle_output, delimiter=' ',
                                                          quoting=csv.QUOTE_MINIMAL)
            yaml_traj_csv_file_handle_output.writerow(output_csv_header)
            if self.config["preprocessing.find_stable_circle"]:
                # since image start_idx and end_idx will have been updated by circleselector by this point.
                start_idx = self.circle_selector_start_idx
                end_idx = self.circle_selector_end_idx
                csv_rows = list(yaml_traj_csv_file_handle)[start_idx:end_idx]
                desired_row_indiced = self.desired_frame_indices[start_idx:end_idx]
                for enum,row in enumerate(csv_rows):
                    idx = desired_row_indiced[enum]
                    row[0] = self.original_filename_expression % idx
                    yaml_traj_csv_file_handle_output.writerow([str(idx)] + row)

            else:
                csv_rows = list(yaml_traj_csv_file_handle)
                for enum,idx in enumerate(self.desired_frame_indices):
                    if self.image_start_idx <= idx <= self.image_end_idx:
                        row = csv_rows[enum]
                        row[0] = self.original_filename_expression % idx
                        yaml_traj_csv_file_handle_output.writerow([str(idx)] + row)



    def openvslam_create_camera_file(self, output_path):
        """
        generate the openvslam cameras.txt file for OmniPhotos
        :param output_path: the path of output camera.txt file for openvslam
        :type output_path:
        """
        with open(output_path, 'w', newline='') as output_file_handle:
            output_file_handle.write(
                "# Camera list with one line of data per camera:\r\n")
            output_file_handle.write(
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\r\n")
            output_file_handle.write("# Number of cameras: 1\r\n")
            camera_info = "1 EQUIRECTANGULAR {} {} {} {} {}\r\n"
            if self.input_type == "video":
                camera_info = camera_info.format(self.frame_width, self.frame_height, 500, self.frame_width / 2,
                                                 self.frame_height / 2)
            elif self.input_type == "image":
                camera_info = camera_info.format(self.frame_width, self.frame_height, 500, self.frame_width / 2,
                                                 self.frame_height / 2)
            output_file_handle.write(camera_info)

    def openvslam_parser_msg(self, msg_file_path):
        """
        parser the msg output the corresponding json and landmark (point cloud).
        output to the same folder of input msg file
        :param msg_file_path: the path of msg file
        """
        json_file_path = msg_file_path + ".json"
        obj_file_path = msg_file_path + ".obj"
        with open(msg_file_path, 'rb') as data:
            data_text = msgpack.Unpacker(data, raw=False, strict_map_key=False).unpack()

        # 1) convert the msg to json
        self.show_info("Output OpenVSLAM msg file to json : {}".format(json_file_path))
        with open(json_file_path, 'w') as json_file:
            json_file.write(json.dumps(data_text))

        # 2) generate the obj file for point cloud
        self.show_info("Output OpenVSLAM point cloud to: {}".format(obj_file_path))
        landmarks = data_text["landmarks"]
        with open(obj_file_path, 'w') as obj_file:
            for key in landmarks:
                landmark = landmarks[key]["pos_w"]
                point_txt = "v {} {} {} {}".\
                    format(str(landmark[0]), str(
                        landmark[1]), str(landmark[2]), os.linesep)
                obj_file.write(point_txt)
