from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def json_default(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    if isinstance(value, Path):
        return str(value)

    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, default=json_default, ensure_ascii=False)


def dump_json(payload: Any, fp) -> None:
    json.dump(payload, fp, default=json_default, ensure_ascii=False)
