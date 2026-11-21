# Modern OmniPhotos workflow

The current path accepts stitched panoramas, reconstructs camera poses with PyCOLMAP 4.2.1, builds the native cache with OpenCV DIS, and opens the prepared dataset in Viewer. Existing COLMAP and OpenVSLAM pose data can still drive native preprocessing. The desktop launcher runs the same Python pipeline as the CLI and shows the child process log and exit status.

## Set up and process a capture

1. Export a 2:1 equirectangular video from Insta360 Studio. Keep the original INSV files and record the export settings. The [research review](modernization-research.md) covers capture and quality checks.
2. Build Viewer and Preprocessing with a C++17 compiler and CMake 3.16 or later. See the dependency table and [CI build definitions](../.github/workflows/cpp_build.yaml). Initialize the ImGui submodule with `git submodule update --init src/3rdParty/imgui`. Use a current platform package manager for OpenCV 4, GLFW, Eigen and glog. On Windows, the CI build uses the vcpkg toolchain.
3. Install Python 3.12, FFmpeg with FFprobe on PATH, and Tk for the desktop launcher. Create a virtual environment and run `python -m pip install -r Python/preprocessing/requirements.txt -r Python/preprocessing/requirements-colmap.txt`. Building mathutils from source requires a C compiler and the Python development headers.
4. Run `python Python/preprocessing/main.py`. Choose **New project**, a dedicated folder, the stitched video and the native Preprocessing executable. You can select **Image folder** before creating a project to use a sequence of stitched panoramas. The launcher saves the project configuration and remembers the input for later runs. You can also start from [the sample YAML](../Python/preprocessing/config_omniphotos.sample.yaml).
5. Start processing. The default profile expects one capture loop and runs frame preparation, PyCOLMAP pose estimation and native DIS cache generation. The launcher selects the generated dataset YAML when there is exactly one. Choose the Viewer executable if it was built separately and click **Open dataset**.
6. For unattended runs, use `python Python/preprocessing/main.py --headless -c /path/to/pipeline.yaml`. Set both step values to `1` for a preparation-only check. Start at step 2 to reuse matching staged frames or step 3 to reuse a matching reconstruction. Use a new project when changing source media or the set of sampled frames.

The default profile reconstructs from one perspective projection per panorama using known crop intrinsics and writes the standard COLMAP text model. Exhaustive matching includes the start/end pair and scales quadratically with the frame count. The pipeline requires a single connected model covering at least 90% of selected frames by default. Sparse registration coverage indicates whether the model is usable by this gate. Check loop geometry and rendered quality separately. The legacy executable adapters remain available for existing projects.

The PyCOLMAP 4.2.1 CUDA wheel for Python 3.12 is distributed for Linux x86-64. Windows has a CPU wheel. On Linux with CUDA support, install `pycolmap-cuda12==4.2.1` in a fresh environment in place of the CPU PyCOLMAP package. Keep the NVIDIA driver current. Check `python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"`. Set `preprocessing.colmap.device: cuda` when GPU use is required, or keep `auto` for the installed backend. This accelerates feature extraction and matching when CUDA is available. DIS remains CPU-based. This task validated the CPU wheel. [PyCOLMAP installation and CUDA builds](https://colmap.github.io/pycolmap/index.html)

### Quest 3 with a Windows PC and RTX 4080

Build the Viewer with `-DWITH_OPENVR=ON -DOPENVR_ROOT_DIR=/path/to/openvr-sdk`. OpenVR SDK 2.15.6 is the tested SDK generation. On Windows, CMake copies the SDK DLL beside the Viewer when it is in the standard SDK layout. Launch Quest Link and open SteamVR. Select **Use SteamVR (PC VR)** in the launcher, or run `Viewer.exe --vr path/to/Config/config-viewer-openvslam.yaml`. Start seated, confirm views in both eyes, then check head translation, recentering and controller bindings. SteamVR can supply a custom binding if the generic binding misses a controller action. This route renders on the PC. A standalone Quest player would require a separate OpenXR Android port and a mobile dataset format. [OpenVR SDK](https://github.com/ValveSoftware/openvr) · [OpenXR](https://www.khronos.org/openxr/)

For a source build in Windows PowerShell with Visual Studio and vcpkg configured:

```powershell
git submodule update --init src/3rdParty/imgui
vcpkg install opencv4 glfw3 eigen3 glog --triplet x64-windows
git clone --depth 1 --branch v2.15.6 https://github.com/ValveSoftware/openvr.git ../openvr-sdk
cmake -S . -B build -A x64 "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake" -DWITH_OPENVR=ON "-DOPENVR_ROOT_DIR=$((Resolve-Path ../openvr-sdk).Path)"
cmake --build build --config Release
```

At 5888×2944, 200 uncompressed RGB panoramas alone occupy about 9.7 GiB before any flow, staging or GPU overhead. Keep the full master and benchmark a smaller working set first. On the 16 GiB RTX 4080, profile host memory and GPU uploads alongside steady-state rendering. Select a resolution and camera count after comparing the output against the full-resolution master. [RTX 4080 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/)

## Dependency inventory

| Area | Current status |
| --- | --- |
| Native build and tests | C++17, CMake 3.16+, GoogleTest 1.14+ from the system or 1.17.0 pinned as the fallback. Cloud build used GCC 14, CMake 3.31 and Ninja. |
| Native UI and rendering | Dear ImGui 1.92.9b with GLFW and OpenGL backends. Cloud used GLFW 3.4 and Mesa software OpenGL. OpenGL 4.1 shaders, GL3W and the existing GLSL renderer remain. |
| Vision, math and logging | Cloud verified native OpenCV 4.10, Eigen 3.4 and glog 0.7.1. The standalone Python environment used OpenCV 4.14, NumPy 2.5, SciPy 1.18 and scikit-image 0.26. Current Python requirements also cover MsgPack 1.x, Pillow, PyYAML, ImageIO, Matplotlib and tqdm. |
| Projection | SciPy rotations and the pinned skylibs fork handle equirectangular transformations. Blender's mathutils 3.3 remains for the trajectory metrics. Keep projection convention tests when replacing either implementation. |
| Video | FFmpeg/FFprobe with the ffmpeg-python wrapper. FFmpeg decodes the source stream and writes just the selected frames. |
| Pose reconstruction | PyCOLMAP 4.2.1 is the default for new projects. Existing configs without the backend setting still use the historical COLMAP CLI adapter. OpenVSLAM still expects the modified historical binary, vocabulary and MsgPack layout. A stella_vslam adapter remains a candidate for native spherical poses. |
| Flow | OpenCV DIS remains the exercised baseline. Brox CUDA requires an OpenCV CUDA build. FlowNet2 retains its historical Python/CUDA environment, checkpoint and compiled PyTorch extensions. Its legacy code still imports SciPy APIs removed from current releases. RAFT and SEA-RAFT remain benchmark candidates. |
| Optional native components | Ceres/SuiteSparse for proxy fitting, OpenVR for PC VR, and auxiliary comparison/benchmark tools are optional. The vendored JsonCpp, nlohmann/json, cxxopts, tinyfiledialogs, tinyobjloader, stb_image and filesystem shim still need a coordinated update or removal. LPIPS and notebooks sit outside the runtime path. |

## Code and pipeline findings

| Finding | Status and impact |
| --- | --- |
| Windows path and compiler assumptions blocked Linux | Fixed executable path discovery, COLMAP/OpenVSLAM CRLF handling, a legacy COLMAP filename casing mismatch, path separators, glog callback types and OpenGL linking. Both shipped native fixture preparations now run on Linux. |
| Frame extraction produced inconsistent endpoints, redundant files and silent failures | Source frame numbers and inclusive ranges now match the staged panorama and perspective images. FFmpeg skips writing discarded frames. Output is staged until all frames are processed. A nonempty earlier frame directory prompts an explicit resume or new project. |
| Headless and native failures could be reported as success | Headless processing propagates worker and native subprocess failures to its exit status. Pipeline config and video metadata parsing replace the previous ad hoc loaders. The launcher surfaces the new status and log. |
| COLMAP had hard-coded crop intrinsics and launched a blocking viewer | Crop intrinsics now use the configured projection. The normal pipeline avoids launching COLMAP's GUI and reports a missing sparse model. The stage 3 reader preserves intermediate reconstruction files. The PyCOLMAP adapter verifies connected-model coverage and exports the standard text format consumed by the native loader. |
| Viewer dataset changes used a detached thread writing shared state | The render thread now owns dataset transitions and waits for the asynchronous CPU load to finish. A failed CPU load leaves the previous dataset active. Repeated large dataset switching and GPU upload failure still need dedicated stress tests. |
| Cache checks often use directory presence or counts | Stages beyond frame preparation still need a provenance manifest. Record the input hash, frame IDs, pose/flow backend version and settings, and stage completion. Publish cache artifacts atomically and validate them before resuming. Enforce one writer per project before adding batch jobs. |
| OpenVSLAM trajectory conversion can associate poses by row position | A missing or reordered pose can shift the mapping. A maintained backend adapter should emit an explicit frame ID, timestamp and registration state. Gate stage 3 on registration coverage and pose consistency. |
| Viewer eagerly stages textures and geometry | Add byte accounting and profile uploads before selecting compression. Lower working resolution, streamed texture residency and bounded decoding should be evaluated with the headset quality gates. The desktop Linux smoke used a small fixture. |
| VR includes legacy synchronization and generic controller mappings | OpenVR input/submission errors are now reported. Benchmark the current synchronization and whether it is needed on Quest Link before changing behavior. Port pose, action, swapchain and frame-timing responsibilities together for OpenXR. |

See the [research and ablation plan](modernization-research.md) for the quality-preserving sequence. Keep DIS and the existing renderer as the control while comparing pose estimation methods and experimenting with flow. Validate the wired PC VR path before moving to wireless or standalone.

## Validation in this workspace

- The Linux build passed all 26 native tests and all 8 Python tests. The native suite processes prepared COLMAP and OpenVSLAM fixtures. The Python tests include a fractional-rate video with exact source-frame verification.
- The GUI created a project for 32 stitched Beijing fixture panoramas, registered all 32 with PyCOLMAP on CPU, generated 64 DIS flow fields and selected the resulting Viewer configuration. The desktop Viewer rendered the new dataset under Mesa software OpenGL.
- The Linux Viewer built and linked against OpenVR 2.15.6. Windows CI execution, CUDA performance, Insta360 capture quality and Quest 3 controller and headset behavior still need their target environment.
