"""benchmark.py - đo độ trễ suy luận."""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian call fn() và trả về percentiles."""
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(len(arr)),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward với batch_size cố định."""
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size)
    if device.startswith("cuda"):
        x = x.to("cuda")
        model.to("cuda")
        sync = torch.cuda.synchronize
    else:
        model.to("cpu")
        sync = None
        device = "cpu"

    if dtype == "fp16":
        model.half()
        x = x.half()
    elif dtype == "amp":
        x = x.to(torch.float32)

    def fn():
        with torch.inference_mode():
            if dtype == "amp" and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)

    result = bench(fn, warmup=warmup, iters=iters, sync=sync)
    result.update({
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "images_per_s": batch_size / (result["p50"] / 1000.0),
        "torch": torch.__version__,
    })
    return result


def tta_latency(model, k_views: int, **kw) -> dict:
    """Đo độ trễ TTA khi chạy K view."""
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    device = kw.get("device", "cuda")
    dtype = kw.get("dtype", "fp32")
    x = torch.randn(batch_size, 3, img_size, img_size)
    if device.startswith("cuda"):
        x = x.to("cuda")
        model.to("cuda")
        sync = torch.cuda.synchronize
    else:
        model.to("cpu")
        sync = None

    def fn():
        with torch.inference_mode():
            if device.startswith("cuda") and dtype == "amp":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    for _ in range(k_views):
                        _ = model(x)
            else:
                for _ in range(k_views):
                    _ = model(x)
    res = bench(fn, warmup=10, iters=50 if k_views <= 5 else 20, sync=sync)
    return {"k_views": k_views, **res}
