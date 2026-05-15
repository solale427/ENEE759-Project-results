#!/usr/bin/env python3
import argparse
import io
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import h5py


def ensure_numpy_pickle_compat() -> None:
    if getattr(pickle, "_tailrisk_numpy_compat_installed", False):
        return

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


def check_imports() -> dict:
    results: dict[str, object] = {}

    try:
        import torch

        results["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            results["torch"]["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        results["torch_error"] = repr(exc)

    for mod_name in ["scenarionet", "metadrive", "h5py", "hydra", "pytorch_lightning", "omegaconf"]:
        try:
            __import__(mod_name)
            results[f"{mod_name}_import"] = "ok"
        except Exception as exc:
            results[f"{mod_name}_import"] = f"error: {exc!r}"

    return results


def configure_pythonpath(repo_root: Path) -> None:
    unitraj_root = repo_root / "third_party" / "UniTraj"
    if str(unitraj_root) not in sys.path:
        sys.path.insert(0, str(unitraj_root))


def check_unitraj_imports(repo_root: Path) -> dict:
    configure_pythonpath(repo_root)
    results: dict[str, object] = {}

    try:
        import unitraj

        results["unitraj_import"] = "ok"
        unitraj_file = getattr(unitraj, "__file__", None)
        if unitraj_file is not None:
            results["unitraj_path"] = str(Path(unitraj_file).resolve())
        else:
            unitraj_path = list(getattr(unitraj, "__path__", []))
            results["unitraj_path"] = str(Path(unitraj_path[0]).resolve()) if unitraj_path else "namespace-package"
    except Exception as exc:
        results["unitraj_import"] = f"error: {exc!r}"
        results["unitraj_traceback"] = traceback.format_exc()
        return results

    try:
        from unitraj.datasets import build_dataset  # noqa: F401

        results["unitraj_dataset_import"] = "ok"
    except Exception as exc:
        results["unitraj_dataset_import"] = f"error: {exc!r}"

    try:
        from unitraj.models import build_model  # noqa: F401

        results["unitraj_model_import"] = "ok"
    except Exception as exc:
        results["unitraj_model_import"] = f"error: {exc!r}"

    try:
        from unitraj.models.mtr.ops.knn import knn_cuda  # noqa: F401

        results["mtr_knn_cuda_import"] = "ok"
    except Exception as exc:
        results["mtr_knn_cuda_import"] = f"error: {exc!r}"

    try:
        from unitraj.models.mtr.ops.attention import attention_cuda  # noqa: F401

        results["mtr_attention_cuda_import"] = "ok"
    except Exception as exc:
        results["mtr_attention_cuda_import"] = f"error: {exc!r}"

    return results


def check_cache(cache_dir: Path) -> dict:
    result: dict[str, object] = {"path": str(cache_dir)}
    file_list_path = cache_dir / "file_list.pkl"
    if not file_list_path.exists():
        result["status"] = "rebuild needed"
        result["reason"] = "missing file_list.pkl"
        return result

    try:
        with open(file_list_path, "rb") as f:
            file_list = pickle.load(f)
        if not file_list:
            result["status"] = "rebuild needed"
            result["reason"] = "empty file_list.pkl"
            return result

        first_key, first_info = next(iter(file_list.items()))
        h5_path = Path(first_info["h5_path"])
        if not h5_path.exists():
            result["status"] = "rebuild needed"
            result["reason"] = f"missing h5 shard {h5_path}"
            return result

        with h5py.File(h5_path, "r") as h5f:
            group = h5f[first_key]
            group_keys = list(group.keys())

        result["status"] = "usable"
        result["entries"] = len(file_list)
        result["sample_key"] = first_key
        result["sample_group_keys"] = group_keys[:10]
        return result
    except Exception as exc:
        result["status"] = "rebuild needed"
        result["reason"] = repr(exc)
        return result


def combine_cache_status(name: str, train_cache_dir: Path, val_cache_dir: Path) -> dict:
    train_result = check_cache(train_cache_dir)
    val_result = check_cache(val_cache_dir)
    combined = {
        "dataset": name,
        "train": train_result,
        "val": val_result,
    }
    combined["status"] = (
        "usable"
        if train_result["status"] == "usable" and val_result["status"] == "usable"
        else "rebuild needed"
    )
    return combined


def build_cfg(repo_root: Path, train_path: Path, val_path: Path, cache_root: Path):
    from omegaconf import OmegaConf

    base_cfg = OmegaConf.load(repo_root / "third_party" / "UniTraj" / "unitraj" / "configs" / "config.yaml")
    method_cfg = OmegaConf.load(repo_root / "third_party" / "UniTraj" / "unitraj" / "configs" / "method" / "MTR.yaml")
    OmegaConf.set_struct(base_cfg, False)
    base_cfg.method = method_cfg
    cfg = OmegaConf.merge(base_cfg, method_cfg)
    cfg.method = method_cfg
    cfg.debug = True
    cfg.devices = [0]
    cfg.load_num_workers = 0
    cfg.train_data_path = [str(train_path)]
    cfg.val_data_path = [str(val_path)]
    cfg.cache_path = str(cache_root)
    cfg.max_data_num = [None]
    cfg.starting_frame = [0]
    cfg.use_cache = False
    cfg.overwrite_cache = True
    cfg.store_data_in_memory = False
    return cfg


def try_dataset_mode(repo_root: Path, cfg, mode_name: str, is_validation: bool) -> dict:
    from torch.utils.data import DataLoader
    from unitraj.datasets import build_dataset

    result: dict[str, object] = {"mode": mode_name}

    dataset = build_dataset(cfg, val=is_validation)
    result["dataset_len"] = len(dataset)
    if len(dataset) == 0:
        result["status"] = "empty"
        return result

    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(loader))
    result["status"] = "usable"
    result["batch_keys"] = list(batch.keys())
    input_dict = batch.get("input_dict", {})
    result["input_dict_keys"] = list(input_dict.keys())[:15]
    return result


def smoke_dataset(repo_root: Path, train_path: Path, val_path: Path, cache_root: Path) -> dict:
    configure_pythonpath(repo_root)
    os.chdir(repo_root)

    result: dict[str, object] = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "cache_root": str(cache_root),
    }

    try:
        import h5py  # noqa: F401
        from unitraj.datasets import build_dataset  # noqa: F401

        ensure_numpy_pickle_compat()
        cfg = build_cfg(repo_root, train_path, val_path, cache_root)
        attempts = []

        try:
            train_attempt = try_dataset_mode(repo_root, cfg, "train", is_validation=False)
            attempts.append(train_attempt)
            if train_attempt["status"] == "usable":
                result["status"] = "usable"
                result["successful_mode"] = "train"
                result["attempts"] = attempts
                return result
        except Exception as exc:
            attempts.append(
                {
                    "mode": "train",
                    "status": "broken",
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        try:
            val_attempt = try_dataset_mode(repo_root, cfg, "validation", is_validation=True)
            attempts.append(val_attempt)
            if val_attempt["status"] == "usable":
                result["status"] = "usable"
                result["successful_mode"] = "validation"
                result["attempts"] = attempts
                return result
        except Exception as exc:
            attempts.append(
                {
                    "mode": "validation",
                    "status": "broken",
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        result["status"] = "broken"
        result["attempts"] = attempts
        return result
    except Exception as exc:
        result["status"] = "broken"
        result["reason"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run environment, UniTraj, and processed-dataset smoke tests.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--waymo-train", required=True)
    parser.add_argument("--waymo-val", required=True)
    parser.add_argument("--waymo-cache-train", required=True)
    parser.add_argument("--waymo-cache-val", required=True)
    parser.add_argument("--av2-train", required=True)
    parser.add_argument("--av2-val", required=True)
    parser.add_argument("--av2-cache-train", required=True)
    parser.add_argument("--av2-cache-val", required=True)
    parser.add_argument("--waymo-subset-cache", required=True)
    parser.add_argument("--av2-subset-cache", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "env_imports": check_imports(),
        "unitraj_imports": check_unitraj_imports(repo_root),
        "waymo_cache": combine_cache_status(
            "waymo",
            Path(args.waymo_cache_train),
            Path(args.waymo_cache_val),
        ),
        "av2_cache": combine_cache_status(
            "av2",
            Path(args.av2_cache_train),
            Path(args.av2_cache_val),
        ),
        "waymo_processed_path": smoke_dataset(
            repo_root,
            Path(args.waymo_train),
            Path(args.waymo_val),
            Path(args.waymo_subset_cache),
        ),
        "av2_processed_path": smoke_dataset(
            repo_root,
            Path(args.av2_train),
            Path(args.av2_val),
            Path(args.av2_subset_cache),
        ),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
