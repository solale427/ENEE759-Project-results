from __future__ import annotations

import io
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def ensure_numpy_pickle_compat() -> None:
    if getattr(pickle, "_tailrisk_numpy_compat_installed", False):
        return

    original_load = pickle.load
    original_loads = pickle.loads

    class _NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    def compat_load(file, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None):
        return _NumpyCompatUnpickler(
            file,
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        ).load()

    def compat_loads(data, /, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None):
        return compat_load(
            io.BytesIO(data),
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        )

    pickle.load = compat_load
    pickle.loads = compat_loads
    pickle._tailrisk_numpy_compat_installed = True
    pickle._tailrisk_original_load = original_load
    pickle._tailrisk_original_loads = original_loads


def ensure_repo_pythonpath(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    src_root = repo_root / "src"
    unitraj_root = repo_root / "third_party" / "UniTraj"
    for path in [src_root, unitraj_root]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_unitraj_config(
    repo_root: Path,
    train_path: Path,
    val_path: Path,
    cache_root: Path,
    *,
    object_types: tuple[str, ...] = ("VEHICLE",),
    debug: bool = True,
    max_data_num: int | None = None,
):
    from omegaconf import OmegaConf

    base_cfg = OmegaConf.load(repo_root / "third_party" / "UniTraj" / "unitraj" / "configs" / "config.yaml")
    method_cfg = OmegaConf.load(repo_root / "third_party" / "UniTraj" / "unitraj" / "configs" / "method" / "MTR.yaml")

    OmegaConf.set_struct(base_cfg, False)
    cfg = OmegaConf.merge(base_cfg, method_cfg)
    cfg.method = method_cfg
    cfg.debug = debug
    cfg.devices = [0]
    cfg.load_num_workers = 0
    cfg.train_data_path = [str(train_path)]
    cfg.val_data_path = [str(val_path)]
    cfg.cache_path = str(cache_root)
    cfg.max_data_num = [max_data_num]
    cfg.starting_frame = [0]
    cfg.use_cache = False
    cfg.overwrite_cache = False
    cfg.store_data_in_memory = False
    cfg.object_type = list(object_types)
    cfg.method.model_name = "MTR"
    cfg.eval_waymo = False
    cfg.eval_argoverse2 = False
    cfg.eval_nuscenes = False
    return cfg


def move_to_device(value: Any, device) -> Any:
    try:
        import torch
    except ImportError:
        return value

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _sanitize_unitraj_cache(cfg, *, val: bool) -> None:
    """Delete stale/partial UniTraj cache directories before dataset load.

    UniTraj's base_dataset.load_data() treats *any* existing cache directory
    as usable when use_cache=False and overwrite_cache=False; it then tries
    to read ``file_list.pkl`` which may not exist if a prior run was killed
    mid-build, resulting in ``ValueError: file_list.pkl not found``. To make
    the pipeline self-healing we pre-check each configured data path and
    remove its corresponding cache directory if ``file_list.pkl`` is missing.

    This only runs when we're building a cache (use_cache=False and not
    overwrite_cache) so healthy caches are never touched.
    """
    import os
    import shutil

    if bool(cfg.get("use_cache", False)) or bool(cfg.get("overwrite_cache", False)):
        return

    cache_root = cfg.get("cache_path")
    if not cache_root:
        return

    data_paths = cfg.val_data_path if val else cfg.train_data_path
    for data_path in data_paths:
        data_path_str = str(data_path)
        parts = data_path_str.rstrip("/").split("/")
        if len(parts) < 2:
            continue
        phase, dataset_name = parts[-2], parts[-1]
        cache_dir = os.path.join(str(cache_root), dataset_name, phase)
        if not os.path.isdir(cache_dir):
            continue
        file_list_path = os.path.join(cache_dir, "file_list.pkl")
        if os.path.exists(file_list_path):
            continue
        print(
            f"[tailrisk_mp.runtime] Detected incomplete UniTraj cache "
            f"at {cache_dir} (missing file_list.pkl); removing so it "
            "can be rebuilt cleanly."
        )
        shutil.rmtree(cache_dir, ignore_errors=True)


def make_dataloader(
    cfg,
    *,
    val: bool,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int | None = None,
):
    from torch.utils.data import DataLoader
    from unitraj.datasets import build_dataset

    if seed is not None:
        set_random_seed(seed)
    _sanitize_unitraj_cache(cfg, val=val)
    dataset = build_dataset(cfg, val=val)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if not val else False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


def build_mtr_model(cfg):
    # Import the MTR path directly so checkpoint validation is not coupled to
    # unrelated model families under unitraj.models.__init__.
    from unitraj.models.mtr.MTR import MotionTransformer

    return MotionTransformer(config=cfg)
