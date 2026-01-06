# perf_utils.py
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version()
    info["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = prop.name
        info["gpu_sm"] = f"{prop.major}.{prop.minor}"
        info["gpu_total_mem_gb"] = round(prop.total_memory / (1024**3), 3)

        # driver version via nvidia-smi if present
        drv = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if drv:
            info["nvidia_driver_version"] = drv.splitlines()[0].strip()

        # runtime visible devices
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # helpful knobs (captured, not forced)
    info["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    info["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    info["float32_matmul_precision"] = getattr(torch, "get_float32_matmul_precision", lambda: "n/a")()

    return info


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def percentile(xs: List[float], q: float) -> float:
    if xs is None or len(xs) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


@dataclass
class BenchSummary:
    focus: str
    timesteps_cap: int
    warmup_blocks: int
    measured_blocks: int
    end_to_end_s: float
    peak_vram_gb: float

    total_p50_ms: float
    total_p95_ms: float
    denoise_p50_ms: float
    denoise_p95_ms: float
    ctx_p50_ms: float
    ctx_p95_ms: float
    decode_p50_ms: float
    decode_p95_ms: float

    fps_p50: float
    fps_p95: float


def summarize_metrics(
    *,
    focus: str,
    timesteps_cap: int,
    warmup_blocks: int,
    wall_s: float,
    peak_vram_gb: float,
    metrics: Dict[str, List[float]],
) -> BenchSummary:
    total_ms = metrics.get("total_ms", [])
    denoise_ms = metrics.get("denoise_ms", [])
    ctx_ms = metrics.get("ctx_ms", [])
    decode_ms = metrics.get("decode_ms", [])
    fps = metrics.get("fps", [])

    return BenchSummary(
        focus=focus,
        timesteps_cap=timesteps_cap,
        warmup_blocks=warmup_blocks,
        measured_blocks=len(total_ms),
        end_to_end_s=wall_s,
        peak_vram_gb=peak_vram_gb,

        total_p50_ms=percentile(total_ms, 50),
        total_p95_ms=percentile(total_ms, 95),
        denoise_p50_ms=percentile(denoise_ms, 50),
        denoise_p95_ms=percentile(denoise_ms, 95),
        ctx_p50_ms=percentile(ctx_ms, 50),
        ctx_p95_ms=percentile(ctx_ms, 95),
        decode_p50_ms=percentile(decode_ms, 50),
        decode_p95_ms=percentile(decode_ms, 95),

        fps_p50=percentile(fps, 50),
        fps_p95=percentile(fps, 95),
    )
