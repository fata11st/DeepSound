"""
ALTO (GPR-Competition 2022 subset) -- загрузка данных и гео-утилиты для Visual Odometry.

Особенности этой раздачи, важные для VO:
  * изображения 500x500 RGB png, получены ресайзом/кропом из исходных 1600x1200 ->
    настоящая матрица камеры НЕИЗВЕСТНА, её надо оценивать по данным (см. 00_diagnostics);
  * позиции даны сразу в UTM 17N (метры) -> ECEF-конверсия для позиций не нужна;
  * altitude -- высота над эллипсоидом WGS84, НЕ высота над землёй (AGL);
  * ориентация -- кватернион scalar-last относительно ECEF;
  * split Test перемешан и без телеметрии -> для VO непригоден.

Использование:
    from alto_data import AltoSplit, DEFAULT_ROOT
    val = AltoSplit(DEFAULT_ROOT, "Val")
    img = val.gray(0)
    xy  = val.xy          # (N,2) easting/northing в метрах
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

# WSL видит диск D: как /mnt/d
DEFAULT_ROOT = Path("/mnt/d/deepsound")

QUAT_COLS = ["orient_x", "orient_y", "orient_z", "orient_w"]  # scipy-совместимый порядок
UTM17N_EPSG = 32617


# ----------------------------------------------------------------------------- split
class AltoSplit:
    """Одна последовательность query-кадров (Train или Val) с телеметрией."""

    def __init__(self, root: Path | str, name: str, cache_dir: Path | str | None = None):
        self.name = name
        self.dir = Path(root) / name
        if not self.dir.exists():
            raise FileNotFoundError(f"Нет каталога {self.dir}")

        csv_path = self.dir / "query.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} не найден. У split '{name}' нет телеметрии "
                f"(это ожидаемо для Test) -- для VO он непригоден."
            )
        df = pd.read_csv(csv_path)

        # 'name' в csv может быть как '000000.png', так и '000000'
        def _to_path(n: str) -> str:
            n = str(n)
            if not n.lower().endswith(".png"):
                n = n + ".png"
            return str(self.dir / "query_images" / n)

        df["path"] = df["name"].map(_to_path)
        self.df = df.reset_index(drop=True)

        self._cache = None
        self._cache_path = None
        if cache_dir is not None:
            self._cache_path = Path(cache_dir) / f"{name}_gray.npy"

    # ---- базовое
    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"<AltoSplit {self.name}: {len(self)} кадров, {self.path_length_m()/1000:.1f} км>"

    # ---- телеметрия
    @property
    def xy(self) -> np.ndarray:
        """(N,2) easting, northing в метрах (UTM 17N)."""
        return self.df[["easting", "northing"]].to_numpy(float)

    @property
    def alt(self) -> np.ndarray:
        """(N,) высота над эллипсоидом WGS84, метры. Это НЕ AGL."""
        return self.df["altitude"].to_numpy(float)

    @property
    def quat(self) -> np.ndarray:
        """(N,4) кватернион scalar-last (x,y,z,w) относительно ECEF."""
        return self.df[QUAT_COLS].to_numpy(float)

    def step_m(self) -> np.ndarray:
        """(N-1,) расстояние между соседними кадрами."""
        return np.linalg.norm(np.diff(self.xy, axis=0), axis=1)

    def path_length_m(self) -> float:
        return float(self.step_m().sum())

    def arc_length_m(self) -> np.ndarray:
        """(N,) накопленная длина пути от первого кадра."""
        return np.concatenate([[0.0], np.cumsum(self.step_m())])

    # ---- изображения
    def gray(self, i: int) -> np.ndarray:
        if self._cache is not None:
            return self._cache[i]
        img = cv2.imread(self.df.at[i, "path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.df.at[i, "path"])
        return img

    def rgb(self, i: int) -> np.ndarray:
        img = cv2.imread(self.df.at[i, "path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.df.at[i, "path"])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def build_gray_cache(self, force: bool = False) -> None:
        """
        Кладёт весь split в один .npy (uint8, N x H x W) и открывает его как memmap.

        Зачем: чтение тысяч мелких png через /mnt/d (drvfs) в WSL катастрофически
        медленное -- порядка 100-300 файлов/с против 5000+ на нативной ext4.
        Val (~4000 кадров) -> ~1 ГБ, Train (~24700) -> ~6 ГБ.
        cache_dir обязательно должен быть на ext4 (например ~/alto_cache).
        """
        if self._cache_path is None:
            raise ValueError("cache_dir не задан в конструкторе")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self._cache_path.exists() and not force:
            self._cache = np.load(self._cache_path, mmap_mode="r")
            return

        probe = cv2.imread(self.df.at[0, "path"], cv2.IMREAD_GRAYSCALE)
        h, w = probe.shape
        arr = np.lib.format.open_memmap(
            self._cache_path, mode="w+", dtype=np.uint8, shape=(len(self), h, w)
        )
        for i in range(len(self)):
            img = cv2.imread(self.df.at[i, "path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(self.df.at[i, "path"])
            arr[i] = img
            if i % 500 == 0:
                print(f"  {self.name}: {i}/{len(self)}", end="\r")
        arr.flush()
        del arr
        self._cache = np.load(self._cache_path, mmap_mode="r")
        print(f"\n  кэш готов: {self._cache_path}")

    # ---- сегментация на непрерывные куски
    def segments(self, max_gap_m: float = 10.0, min_len: int = 200) -> list[tuple[int, int]]:
        """
        Режет split на непрерывные подпоследовательности по разрывам в GT-позициях.
        Возвращает список полуинтервалов [start, end).
        Разрывы бывают: пропуски кадров, зависания, развороты.
        """
        breaks = np.flatnonzero(self.step_m() > max_gap_m) + 1
        bounds = np.concatenate([[0], breaks, [len(self)]])
        return [
            (int(a), int(b))
            for a, b in zip(bounds[:-1], bounds[1:])
            if b - a >= min_len
        ]


# ------------------------------------------------------------------------- геометрия
def utm_to_lonlat(easting: np.ndarray, northing: np.ndarray, epsg: int = UTM17N_EPSG):
    from pyproj import Transformer

    tf = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(np.asarray(easting), np.asarray(northing))
    return np.asarray(lon), np.asarray(lat)


def R_ecef_to_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Матрица поворота из ECEF в локальный ENU (East-North-Up)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sla, cla = np.sin(lat), np.cos(lat)
    slo, clo = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-slo, clo, 0.0],
            [-sla * clo, -sla * slo, cla],
            [cla * clo, cla * slo, sla],
        ]
    )


def cam_rotations_enu(split: AltoSplit, invert: bool = False) -> np.ndarray:
    """
    (N,3,3) ориентация камеры в локальном ENU-фрейме.

    invert=False трактует кватернион как R_ecef_cam (камера -> ECEF),
    invert=True  -- как R_cam_ecef. Какая из трактовок верна, определяется
    эмпирически надир-тестом в 00_diagnostics (ось z камеры должна смотреть вниз).
    """
    from scipy.spatial.transform import Rotation as Rot

    R_q = Rot.from_quat(split.quat).as_matrix()
    if invert:
        R_q = np.transpose(R_q, (0, 2, 1))
    lon, lat = utm_to_lonlat(split.xy[:, 0], split.xy[:, 1])
    out = np.empty_like(R_q)
    for i in range(len(R_q)):
        out[i] = R_ecef_to_enu(lat[i], lon[i]) @ R_q[i]
    return out


def yaw_from_R_enu(R_enu: np.ndarray) -> np.ndarray:
    """
    (N,) курс камеры в радианах: азимут проекции оси x камеры на горизонт,
    отсчитывается от севера по часовой стрелке.
    """
    x_axis = R_enu[:, :, 0]          # ось x камеры в координатах ENU
    return np.arctan2(x_axis[:, 0], x_axis[:, 1])   # atan2(East, North)


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------ оценка масштаба (GSD)
def similarity_between(img0: np.ndarray, img1: np.ndarray, nfeat: int = 3000):
    """
    Быстрая оценка 2D-подобия между двумя кадрами через ORB + estimateAffinePartial2D.
    Возвращает (dx_px, dy_px, theta_rad, scale, n_inliers) либо None.

    Используется как измерительный инструмент в диагностике (оценка GSD,
    проверка north-alignment), а не как VO-фронтенд.
    """
    orb = cv2.ORB_create(nfeatures=nfeat, fastThreshold=7)
    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)
    if d0 is None or d1 is None or len(k0) < 20 or len(k1) < 20:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 12:
        return None

    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000
    )
    if M is None or inl is None or int(inl.sum()) < 10:
        return None

    a, b = M[0, 0], M[1, 0]
    scale = float(np.hypot(a, b))
    theta = float(np.arctan2(b, a))
    return float(M[0, 2]), float(M[1, 2]), theta, scale, int(inl.sum())


def K_from_gsd(gsd_m_per_px: float, agl_m: float, w: int, h: int) -> np.ndarray:
    """
    Матрица камеры, восстановленная из эмпирической GSD и высоты над землёй.
    Для надирной камеры: GSD = AGL / f_px  =>  f_px = AGL / GSD.
    """
    f = agl_m / gsd_m_per_px
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def rotation_scale_fourier(img0: np.ndarray, img1: np.ndarray):
    """
    Независимая оценка поворота и масштаба: фазовая корреляция в лог-полярных
    координатах амплитудного спектра. Инвариантна к сдвигу, работает без
    ключевых точек вообще -- поэтому служит перекрёстной проверкой для
    оценок, полученных через детекторы и RANSAC.

    Возвращает (theta_rad, scale, response). Знаки проверены синтетически.
    """
    h, w = img0.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def spec(x):
        F = np.fft.fftshift(np.abs(np.fft.fft2(x.astype(np.float32) * win)))
        return np.log1p(F).astype(np.float32)

    M = w / np.log(w / 2.0)
    center = (w / 2.0, h / 2.0)
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    la = cv2.logPolar(spec(img0), center, M, flags) * win
    lb = cv2.logPolar(spec(img1), center, M, flags) * win
    (dx, dy), resp = cv2.phaseCorrelate(la, lb)
    return float(np.radians(dy * 360.0 / h)), float(np.exp(-dx / M)), float(resp)


def yaw_from_translation(dxy_world: np.ndarray, dxy_img: np.ndarray) -> np.ndarray:
    """
    Абсолютный курс камеры из направления смещения -- без оценки поворота.

    Для надирной камеры с осью x под азимутом psi мировое смещение с азимутом
    alpha даёт сдвиг сцены в кадре в направлении (alpha - psi) + 180°.
    Отсюда psi = alpha - atan2(dy_px, dx_px) + pi.

    Ключевое свойство: оценка получается покадрово и не накапливается,
    поэтому систематическая ошибка оценщика поворота на неё не влияет.
    Постоянное слагаемое (крепление камеры относительно носа) остаётся,
    но при сравнении с GT-курсом оно видно как ровное смещение графика.

    dxy_world -- (N,2) приращения easting/northing, метры
    dxy_img   -- (N,2) сдвиг сцены в кадре, пиксели (x вправо, y вниз)
    """
    alpha = np.arctan2(dxy_world[:, 0], dxy_world[:, 1])          # азимут от севера
    beta = np.arctan2(dxy_img[:, 1], dxy_img[:, 0])
    return wrap_pi(alpha - beta + np.pi)
