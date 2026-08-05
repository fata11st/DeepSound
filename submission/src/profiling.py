"""
Замер нагрузки на устройство инференса (пункт (e) задания).

Меряет то, что покадровые счётчики в vo_core не видят:
  * RSS процесса -- реальная оперативная память, а не прирост за вызов;
  * VRAM через torch и через nvidia-smi (torch видит только свои аллокации,
    вне их остаётся контекст CUDA ~300-500 МБ);
  * латентность с прогревом и перцентилями, а не только медианой.

Требует: pip install psutil pynvml
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field

import numpy as np


def _rss_mb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20


def _nvml_used_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20
    except Exception:
        return float("nan")


def bench_frontend(frontend, img0, img1, warmup: int = 10, runs: int = 50) -> dict:
    """
    Латентность и память одного фронтенда на фиксированной паре кадров.

    Прогрев обязателен: первый вызов GPU-модели включает загрузку ядер CUDA
    и autotune, он медленнее установившегося в разы. Для CPU-детекторов
    прогрев съедает эффект холодного кэша.
    """
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False

    for _ in range(warmup):
        frontend(img0, img1)

    gc.collect()
    if cuda:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rss0, nvml0 = _rss_mb(), _nvml_used_mb()

    t_det, t_match, t_all, n_match = [], [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        m = frontend(img0, img1)
        if cuda:
            torch.cuda.synchronize()
        t_all.append(time.perf_counter() - t0)
        t_det.append(m.t_detect); t_match.append(m.t_match); n_match.append(m.n)

    q = lambda a, p: float(np.percentile(np.asarray(a) * 1000, p))
    out = dict(
        имя=getattr(frontend, "name", type(frontend).__name__),
        мс_медиана=round(q(t_all, 50), 2), мс_p95=round(q(t_all, 95), 2),
        мс_детект=round(q(t_det, 50), 2), мс_матч=round(q(t_match, 50), 2),
        кадр_в_с=round(1.0 / np.median(t_all), 1),
        матчей=int(np.median(n_match)),
        RSS_МБ=round(_rss_mb(), 1), RSS_прирост_МБ=round(_rss_mb() - rss0, 1),
    )
    if cuda:
        out["VRAM_torch_МБ"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        out["VRAM_nvml_МБ"] = round(_nvml_used_mb(), 1)
        out["VRAM_прирост_МБ"] = round(_nvml_used_mb() - nvml0, 1)
    return out


def bench_backend(solver, matches, warmup: int = 5, runs: int = 50) -> dict:
    """Латентность геометрического решателя на фиксированном наборе соответствий.

    RANSAC недетерминирован по времени: число итераций зависит от доли
    выбросов, поэтому p95 здесь информативнее медианы.
    """
    for _ in range(warmup):
        solver(matches)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); solver(matches); ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts) * 1000
    return dict(мс_медиана=round(float(np.median(ts)), 3),
                мс_p95=round(float(np.percentile(ts, 95)), 3),
                мс_max=round(float(ts.max()), 3),
                n_точек=matches.n)


def baseline_memory() -> dict:
    """Память до загрузки моделей -- вычитается из замеров, чтобы получить чистый вклад."""
    gc.collect()
    return dict(RSS_МБ=round(_rss_mb(), 1), VRAM_nvml_МБ=round(_nvml_used_mb(), 1))
