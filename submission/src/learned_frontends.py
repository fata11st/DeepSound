"""
Обучаемые фронтенды для VO: SuperPoint+LightGlue и DISK+LightGlue.

Две независимые реализации, потому что пакеты ставятся из разных мест:

  * `SuperPointLightGlue`  -- пакет cvg/LightGlue (ставится с github);
  * `DISKLightGlue`        -- kornia (ставится с pypi).

Второй нужен как запасной путь: в kornia есть матчер LightGlue и детектор DISK,
но **экстрактора SuperPoint нет** (`kornia.feature.SuperPoint` не существует
даже в 0.8.3). Поэтому запасной вариант -- это честно DISK+LightGlue,
а не SuperPoint, и в отчёте его надо называть своим именем.

Оба класса возвращают vo_core.Matches и совместимы с общим каркасом run().

Установка:
    pip install git+https://github.com/cvg/LightGlue.git      # основной путь
    pip install kornia kornia_rs                              # запасной
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from vo_core import Matches


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    except Exception:
        return 0.0


class _TorchFrontEnd:
    """Общая часть: перенос на устройство, замер времени, VRAM и RSS."""

    name = "torch-frontend"

    def __init__(self, device: str = "cuda", fp16: bool = False):
        import torch

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            print("[learned] CUDA недоступна, работаю на CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"

    def _gray_tensor(self, img: np.ndarray):
        t = self.torch.from_numpy(np.ascontiguousarray(img)).float()[None, None] / 255.0
        return (t.half() if self.fp16 else t).to(self.device)

    def _rgb_tensor(self, img: np.ndarray):
        t = self._gray_tensor(img)
        return t.repeat(1, 3, 1, 1)

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def _reset_mem(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def _peak_vram(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return self.torch.cuda.max_memory_allocated() / 2 ** 20


class SuperPointLightGlue(_TorchFrontEnd):
    """SuperPoint + LightGlue из пакета cvg/LightGlue."""

    def __init__(self, max_keypoints: int = 2048, device: str = "cuda",
                 fp16: bool = False, detection_threshold: float = 0.0005):
        super().__init__(device, fp16)
        from lightglue import LightGlue, SuperPoint

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints,
                                    detection_threshold=detection_threshold).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        if self.fp16:
            self.extractor = self.extractor.half()
            self.matcher = self.matcher.half()
        self.name = "SuperPoint+LightGlue"

    def __call__(self, img0: np.ndarray, img1: np.ndarray) -> Matches:
        torch = self.torch
        self._reset_mem()
        rss0 = _rss_mb()
        with torch.inference_mode():
            t0 = time.perf_counter()
            f0 = self.extractor.extract(self._gray_tensor(img0))
            f1 = self.extractor.extract(self._gray_tensor(img1))
            self._sync()
            t_det = time.perf_counter() - t0

            t0 = time.perf_counter()
            out = self.matcher({"image0": f0, "image1": f1})
            self._sync()
            t_match = time.perf_counter() - t0

        m = out["matches"][0].cpu().numpy()
        k0 = f0["keypoints"][0].float().cpu().numpy()
        k1 = f1["keypoints"][0].float().cpu().numpy()
        if len(m) == 0:
            empty = np.zeros((0, 2), np.float32)
            return Matches(empty, empty.copy(), len(k0), len(k1), t_det, t_match,
                           self._peak_vram(), _rss_mb() - rss0)
        return Matches(k0[m[:, 0]].astype(np.float32), k1[m[:, 1]].astype(np.float32),
                       len(k0), len(k1), t_det, t_match, self._peak_vram(), _rss_mb() - rss0)


class DISKLightGlue(_TorchFrontEnd):
    """
    DISK + LightGlue из kornia -- запасной путь, если github недоступен.

    Это НЕ SuperPoint: kornia не содержит его экстрактора. DISK -- другой
    обучаемый детектор-дескриптор (128-мерный против 256 у SuperPoint),
    матчер тот же LightGlue со своими весами под DISK.
    """

    def __init__(self, max_keypoints: int = 2048, device: str = "cuda",
                 fp16: bool = False, checkpoint: str = "depth", window_size: int = 5):
        super().__init__(device, fp16)
        from kornia.feature import DISK, LightGlue

        self.extractor = DISK.from_pretrained(checkpoint).eval().to(self.device)
        self.matcher = LightGlue("disk").eval().to(self.device)
        if self.fp16:
            self.extractor = self.extractor.half()
            self.matcher = self.matcher.half()
        self.max_kp = max_keypoints
        self.window_size = window_size
        self.name = "DISK+LightGlue"

    def _extract(self, img: np.ndarray):
        # DISK ждёт трёхканальный вход
        feats = self.extractor(self._rgb_tensor(img), n=self.max_kp,
                               window_size=self.window_size, score_threshold=0.0,
                               pad_if_not_divisible=True)[0]
        h, w = img.shape[:2]
        size = self.torch.tensor([[float(w), float(h)]], device=self.device)
        return dict(keypoints=feats.keypoints[None].float(),
                    descriptors=feats.descriptors[None].float(),
                    image_size=size)

    def __call__(self, img0: np.ndarray, img1: np.ndarray) -> Matches:
        torch = self.torch
        self._reset_mem()
        rss0 = _rss_mb()
        with torch.inference_mode():
            t0 = time.perf_counter()
            f0, f1 = self._extract(img0), self._extract(img1)
            self._sync()
            t_det = time.perf_counter() - t0

            t0 = time.perf_counter()
            out = self.matcher({"image0": f0, "image1": f1})
            self._sync()
            t_match = time.perf_counter() - t0

        m = out["matches"][0].cpu().numpy()
        k0 = f0["keypoints"][0].cpu().numpy()
        k1 = f1["keypoints"][0].cpu().numpy()
        if len(m) == 0:
            empty = np.zeros((0, 2), np.float32)
            return Matches(empty, empty.copy(), len(k0), len(k1), t_det, t_match,
                           self._peak_vram(), _rss_mb() - rss0)
        return Matches(k0[m[:, 0]].astype(np.float32), k1[m[:, 1]].astype(np.float32),
                       len(k0), len(k1), t_det, t_match, self._peak_vram(), _rss_mb() - rss0)


def build_learned(max_keypoints: int = 2048, device: str = "cuda",
                  prefer: str = "superpoint") -> Optional[object]:
    """
    Возвращает доступный обучаемый фронтенд или None.

    prefer='superpoint' -- сначала пробует cvg/LightGlue, потом kornia.
    Печатает, что именно получилось: подмена SuperPoint на DISK должна быть
    видна в логе, иначе легко описать в отчёте не тот метод.
    """
    order = ([SuperPointLightGlue, DISKLightGlue] if prefer == "superpoint"
             else [DISKLightGlue, SuperPointLightGlue])
    for cls in order:
        try:
            fe = cls(max_keypoints=max_keypoints, device=device)
            print(f"[learned] используется {fe.name}")
            return fe
        except Exception as e:
            print(f"[learned] {cls.__name__} недоступен: {type(e).__name__}: {e}")
    return None
