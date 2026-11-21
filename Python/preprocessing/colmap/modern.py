"""Headless COLMAP 4 reconstruction for a single prepared capture loop."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory


def reconstruct(images, image_names, output, camera_params, device, min_coverage):
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "Install pycolmap from Python/preprocessing/requirements-colmap.txt."
        ) from exc

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("COLMAP device must be auto, cpu, or cuda.")
    if device == "cuda" and not pycolmap.has_cuda:
        raise RuntimeError("The installed PyCOLMAP build has no CUDA support.")
    if not 0 < min_coverage <= 1:
        raise ValueError("COLMAP minimum registration fraction must be greater than zero and at most one.")
    if len(image_names) < 3 or any(not (images / name).is_file() for name in image_names):
        raise ValueError("COLMAP reconstruction requires at least three prepared frames and all selected files.")
    if any(output.iterdir()):
        raise FileExistsError(f"{output} already contains a reconstruction. Resume from step 3 or use a new project.")

    threads = min(os.cpu_count() or 1, 8)
    compute_device = getattr(pycolmap.Device, device)
    use_gpu = device != "cpu" and pycolmap.has_cuda
    reader = pycolmap.ImageReaderOptions(camera_model="SIMPLE_PINHOLE", camera_params=camera_params)
    extraction = pycolmap.FeatureExtractionOptions(num_threads=threads, use_gpu=use_gpu)
    matching = pycolmap.FeatureMatchingOptions(num_threads=threads, use_gpu=use_gpu, guided_matching=True)
    mapping = pycolmap.IncrementalPipelineOptions(
        num_threads=threads, random_seed=0, ba_refine_focal_length=not bool(camera_params),
    )

    with TemporaryDirectory(prefix="pycolmap-", dir=output.parent) as directory:
        staging = Path(directory)
        database = staging / "database.db"
        pycolmap.extract_features(
            database, images, image_names=image_names, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader, extraction_options=extraction, device=compute_device,
        )
        # Exhaustive matching includes the start/end pair needed for loop closure.
        pycolmap.match_exhaustive(database, matching_options=matching, device=compute_device)
        models_dir = staging / "models"
        models_dir.mkdir()
        models = pycolmap.incremental_mapping(database, images, models_dir, options=mapping)
        if not models:
            raise RuntimeError("COLMAP did not reconstruct a camera model.")
        best = max(models.values(), key=lambda model: model.num_reg_images())
        registered = best.num_reg_images()
        if registered < 3 or registered / len(image_names) < min_coverage:
            raise RuntimeError(f"COLMAP registered {registered}/{len(image_names)} prepared frames. "
                               f"The best connected model must cover at least {min_coverage:.0%}.")
        print(f"COLMAP registered {registered}/{len(image_names)} prepared frames.")
        result = staging / "text-model"
        result.mkdir()
        best.write_text(result)
        # Replacing an empty directory keeps the previous output intact if another run populated it.
        output.rmdir()
        result.rename(output)
