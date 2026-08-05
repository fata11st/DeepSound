"""
vo_core -- движок Visual Odometry для ALTO (GPR-Competition subset).

Архитектура:

    FrontEnd(img0, img1) -> Matches        сопоставление точек
    Backend(Matches, K)  -> RelPose        относительная поза
    Integrator           -> Trajectory     накопление
    metrics              -> ATE / RPE      оценка

Соглашения (проверены синтетическим тестом в test_vo_core.py):
  * Камера: x вправо, y вниз, z вперёд (в сцену). Для надира z смотрит к земле.
  * RelPose.R, RelPose.t задают преобразование точки из кадра 0 в кадр 1:
        P1 = R @ P0 + t
    Это то же соглашение, что у cv2.recoverPose.
  * Поза камеры в мире накапливается как  T_w1 = T_w0 @ inv([R|t]).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- типы
@dataclass
class Matches:
    pts0: np.ndarray                 # (N,2) float32
    pts1: np.ndarray                 # (N,2) float32
    n_kpts0: int = 0
    n_kpts1: int = 0
    t_detect: float = 0.0
    t_match: float = 0.0
    peak_vram_mb: float = 0.0
    rss_mb: float = 0.0            # прирост resident set size за вызов

    @property
    def n(self) -> int:
        return len(self.pts0)


@dataclass
class RelPose:
    R: np.ndarray                    # (3,3)
    t: np.ndarray                    # (3,) — единичной длины для 'essential', метры для 'homography'
    inliers: np.ndarray              # (N,) bool
    ok: bool
    t_solve: float = 0.0
    scale_known: bool = False        # True -> t уже в метрах
    extra: dict = field(default_factory=dict)

    @property
    def inlier_ratio(self) -> float:
        return float(self.inliers.mean()) if len(self.inliers) else 0.0


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


# ---------------------------------------------------------------------- фронтенды
class DescriptorFrontEnd:
    """SIFT или ORB + BFMatcher с ratio-test и взаимной проверкой."""

    def __init__(self, kind: Literal["sift", "orb"] = "sift", n_features: int = 2000,
                 ratio: float = 0.8, cross_check: bool = True):
        self.kind, self.ratio, self.cross_check = kind, ratio, cross_check
        if kind == "sift":
            self.det = cv2.SIFT_create(nfeatures=n_features)
            self.norm = cv2.NORM_L2
        elif kind == "orb":
            self.det = cv2.ORB_create(nfeatures=n_features, fastThreshold=7,
                                      scaleFactor=1.2, nlevels=8)
            self.norm = cv2.NORM_HAMMING
        else:
            raise ValueError(kind)
        self.name = f"{kind.upper()}+BF"

    def _knn_ratio(self, dA, dB):
        bf = cv2.BFMatcher(self.norm)
        pairs = bf.knnMatch(dA, dB, k=2)
        return {m.queryIdx: m.trainIdx
                for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self.ratio * n.distance}

    def __call__(self, img0, img1) -> Matches:
        t0 = time.perf_counter()
        k0, d0 = self.det.detectAndCompute(img0, None)
        k1, d1 = self.det.detectAndCompute(img1, None)
        t_det = time.perf_counter() - t0

        empty = Matches(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                        len(k0 or []), len(k1 or []), t_det, 0.0)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return empty

        t0 = time.perf_counter()
        fwd = self._knn_ratio(d0, d1)
        if self.cross_check:
            bwd = self._knn_ratio(d1, d0)
            fwd = {i: j for i, j in fwd.items() if bwd.get(j, -1) == i}
        t_match = time.perf_counter() - t0

        if not fwd:
            return empty
        idx = np.array(list(fwd.items()))
        p0 = np.float32([k0[i].pt for i in idx[:, 0]])
        p1 = np.float32([k1[j].pt for j in idx[:, 1]])
        return Matches(p0, p1, len(k0), len(k1), t_det, t_match)


class LKFrontEnd:
    """
    Детектор (SIFT/ORB/goodFeaturesToTrack) + пирамидальный Lucas-Kanade
    с forward-backward проверкой.

    Важно: LK по определению трекер малых смещений. При stride, дающем сдвиг
    в 40-90 px, нужна пирамида глубиной >= 4 и/или инициализация сдвига.
    Параметр `init_shift` позволяет подать априорную оценку сдвига
    (например, от предыдущего шага) -- без неё LK на больших stride разваливается,
    и это само по себе результат для отчёта.
    """

    def __init__(self, detector: Literal["sift", "orb", "gftt"] = "gftt",
                 n_features: int = 2000, win: int = 21, levels: int = 4,
                 fb_thresh: float = 1.0):
        self.detector, self.n_features = detector, n_features
        self.fb_thresh = fb_thresh
        self.lk = dict(winSize=(win, win), maxLevel=levels,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if detector == "sift":
            self.det = cv2.SIFT_create(nfeatures=n_features)
        elif detector == "orb":
            self.det = cv2.ORB_create(nfeatures=n_features, fastThreshold=7)
        else:
            self.det = None
        self.name = f"{detector.upper()}+LK"

    def _detect(self, img):
        if self.det is None:
            p = cv2.goodFeaturesToTrack(img, self.n_features, 0.01, 7)
            return np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2)
        kp = self.det.detect(img, None)
        return np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)

    def __call__(self, img0, img1, init_shift: Optional[np.ndarray] = None) -> Matches:
        t0 = time.perf_counter()
        p0 = self._detect(img0)
        t_det = time.perf_counter() - t0
        empty = Matches(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                        len(p0), 0, t_det, 0.0)
        if len(p0) < 8:
            return empty

        t0 = time.perf_counter()
        guess = (p0 + np.asarray(init_shift, np.float32)) if init_shift is not None else p0.copy()
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW if init_shift is not None else 0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0.reshape(-1, 1, 2),
                                             guess.reshape(-1, 1, 2), flags=flags, **self.lk)
        p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, p0.reshape(-1, 1, 2).copy(),
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        t_match = time.perf_counter() - t0

        p1 = p1.reshape(-1, 2); p0b = p0b.reshape(-1, 2)
        good = (st.ravel() == 1) & (st_b.ravel() == 1)
        good &= np.linalg.norm(p0 - p0b, axis=1) < self.fb_thresh
        return Matches(p0[good], p1[good], len(p0), int(good.sum()), t_det, t_match)


# SuperPoint+LightGlue и DISK+LightGlue вынесены в learned_frontends.py:
# они тянут torch и веса из сети, и их не должно быть в импорте базового модуля.
#     from learned_frontends import build_learned
#     fe = build_learned(max_keypoints=2048, device="cuda")


# ------------------------------------------------------------------------ бэкенды
RANSAC_METHODS = {
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
    "magsac": cv2.USAC_MAGSAC,
    "usac_default": cv2.USAC_DEFAULT,
}


def solve_essential(m: Matches, K: np.ndarray, thresh: float = 1.0,
                    method: str = "magsac", conf: float = 0.9999) -> RelPose:
    """5-точечный алгоритм. Возвращает t единичной длины (масштаб неизвестен)."""
    t0 = time.perf_counter()
    bad = RelPose(np.eye(3), np.zeros(3), np.zeros(m.n, bool), False)
    if m.n < 8:
        return bad
    E, mask = cv2.findEssentialMat(m.pts0, m.pts1, K, method=RANSAC_METHODS[method],
                                   prob=conf, threshold=thresh)
    if E is None or E.shape != (3, 3):
        return bad
    n_in, R, t, mask_pose = cv2.recoverPose(E, m.pts0, m.pts1, K, mask=mask)
    inl = (mask_pose.ravel() > 0)
    return RelPose(R, t.ravel(), inl, n_in >= 8, time.perf_counter() - t0,
                   scale_known=False, extra=dict(n_pose_inliers=int(n_in)))


def solve_homography(m: Matches, K: np.ndarray, thresh: float = 3.0,
                     method: str = "magsac", conf: float = 0.9999,
                     plane_dist: Optional[float] = None,
                     n_prior: np.ndarray = np.array([0.0, 0.0, 1.0])) -> RelPose:
    """
    Гомография + разложение. Плоскость задана в системе кадра 0.

    plane_dist -- расстояние до плоскости в метрах (для надира ≈ AGL).
    Если задано, t возвращается в метрах и scale_known=True.
    """
    t0 = time.perf_counter()
    bad = RelPose(np.eye(3), np.zeros(3), np.zeros(m.n, bool), False)
    if m.n < 8:
        return bad
    H, mask = cv2.findHomography(m.pts0, m.pts1, RANSAC_METHODS[method], thresh,
                                 maxIters=10000, confidence=conf)
    if H is None:
        return bad
    inl = mask.ravel() > 0
    if inl.sum() < 8:
        return bad

    n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    # ВАЖНО: filterHomographyDecompByVisibleRefpoints здесь НЕ используется --
    # синтетический тест показывает, что при слабом параллаксе он выбрасывает
    # физически верное решение. Четыре решения образуют две пары (n, t) / (-n, -t);
    # для надирной камеры земля лежит в +z, значит верная нормаль имеет n_z > 0.
    # Отбор по максимуму n·n_prior разделяет и пары, и знаки одновременно.
    cand = list(range(n_sol))

    best, best_score = None, -np.inf
    for i in cand:
        score = float(np.asarray(ns[i]).ravel() @ n_prior)
        if score > best_score:
            best, best_score = i, score
    R = np.asarray(Rs[best])
    t = np.asarray(ts[best]).ravel()                 # это t/d
    scale_known = plane_dist is not None
    if scale_known:
        t = t * float(plane_dist)
    return RelPose(R, t, inl, True, time.perf_counter() - t0, scale_known,
                   extra=dict(H=H, normal=np.asarray(ns[best]).ravel(),
                              n_candidates=len(cand), normal_score=best_score))


def solve_similarity(m: Matches, gsd: float, thresh: float = 3.0,
                     agl: Optional[float] = None,
                     principal_point=(250.0, 250.0)) -> RelPose:
    """
    Базовый 2D-метод: подобие в плоскости изображения (сдвиг + поворот + масштаб).

    Для строго надирной камеры над плоскостью это почти полная модель движения,
    поэтому метод служит нижней границей, которую «умные» пайплайны обязаны обойти.
    Сдвиг переводится в метры через GSD, поворот интерпретируется как рыскание,
    изменение масштаба -- как изменение высоты.
    """
    t0 = time.perf_counter()
    bad = RelPose(np.eye(3), np.zeros(3), np.zeros(m.n, bool), False)
    if m.n < 6:
        return bad
    M, inl = cv2.estimateAffinePartial2D(m.pts0, m.pts1, method=cv2.RANSAC,
                                         ransacReprojThreshold=thresh, maxIters=10000,
                                         confidence=0.9999)
    if M is None or inl is None or inl.sum() < 6:
        return bad
    a, b = M[0, 0], M[1, 0]
    s = float(np.hypot(a, b))
    theta = float(np.arctan2(b, a))                  # поворот сцены в кадре
    R = cv2.Rodrigues(np.array([0.0, 0.0, theta]))[0]

    # Смещение надо мерить относительно главной точки, а не относительно
    # пикселя (0,0): в аффинной матрице член M[:,2] содержит вклад поворота
    # вокруг угла кадра. Для 500x500 и поворота в 1 градус это ошибка ~4 px
    # (~2 м на местности) на каждом шаге -- систематический дрейф.
    A = M[:2, :2]
    d_px = M[:2, 2] - (np.eye(2) - A) @ np.asarray(principal_point, float)

    # s > 1 -> сцена крупнее -> камера снизилась. z камеры направлена к земле,
    # поэтому смещение начала координат по z отрицательно.
    tz = 0.0 if agl is None else -float(agl) * (1.0 - 1.0 / max(s, 1e-6))
    t = np.array([d_px[0] * gsd, d_px[1] * gsd, tz])
    return RelPose(R, t, inl.ravel() > 0, True, time.perf_counter() - t0,
                   scale_known=True, extra=dict(scale=s, theta=theta))


# --------------------------------------------------------------------- интегратор
class Integrator:
    """
    Накапливает относительные позы в траекторию камеры.

    scale_policy:
      'native' -- доверять |t| из бэкенда (гомография с plane_dist, подобие);
      'gt'     -- брать длину шага из ground truth (стандартная практика оценки
                  монокулярной одометрии: убирает неопределённость масштаба и
                  показывает качество направления и вращения отдельно);
      'const'  -- фиксированная длина шага.
    """

    def __init__(self, scale_policy: Literal["native", "gt", "const"] = "native",
                 const_step: float = 1.0,
                 on_fail: Literal["hold", "repeat"] = "repeat"):
        """
        on_fail определяет поведение при отказе бэкенда:
          'hold'   -- поза замирает. Каждый отказ теряет целый шаг движения:
                      на Train при 4 % отказов это больше километра пути;
          'repeat' -- повторяется последнее удачное относительное движение
                      (модель постоянной скорости). На гладкой траектории
                      вертолёта это заметно ближе к истине.
        """
        self.scale_policy = scale_policy
        self.const_step = const_step
        self.on_fail = on_fail
        self.T = np.eye(4)
        self.poses = [self.T.copy()]
        self.last_rel = None
        self.log: list[dict] = []

    def step(self, rp: RelPose, gt_step: Optional[float] = None, meta: Optional[dict] = None):
        if not rp.ok:
            if self.on_fail == "repeat" and self.last_rel is not None:
                self.T = self.T @ se3_inv(self.last_rel)
            self.poses.append(self.T.copy())
            self.log.append(dict(ok=False, inlier_ratio=0.0, **(meta or {})))
            return

        t = rp.t.astype(float).copy()
        if self.scale_policy == "gt":
            nt = np.linalg.norm(t)
            t = t / nt * float(gt_step) if nt > 1e-9 else t
        elif self.scale_policy == "const":
            nt = np.linalg.norm(t)
            t = t / nt * self.const_step if nt > 1e-9 else t

        rel = se3(rp.R, t)
        self.last_rel = rel
        self.T = self.T @ se3_inv(rel)
        self.poses.append(self.T.copy())
        self.log.append(dict(ok=True, inlier_ratio=rp.inlier_ratio,
                             t_norm=float(np.linalg.norm(t)), **(meta or {})))

    @property
    def positions(self) -> np.ndarray:
        return np.array([T[:3, 3] for T in self.poses])

    @property
    def rotations(self) -> np.ndarray:
        return np.array([T[:3, :3] for T in self.poses])


# ------------------------------------------------------------------------ метрики
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Оптимальное выравнивание src -> dst. Возвращает (R, t, s)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[-1, -1] = -1
    R = U @ W @ Vt
    var = float(S.var(0).sum())
    s = float((sig * np.diag(W)).sum() / var) if (with_scale and var > 1e-12) else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(est: np.ndarray, gt: np.ndarray, with_scale: bool = True):
    """Absolute Trajectory Error после выравнивания Умеямы."""
    R, t, s = umeyama(est, gt, with_scale)
    aligned = (s * (R @ est.T).T) + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return dict(rmse=float(np.sqrt((err**2).mean())), mean=float(err.mean()),
                median=float(np.median(err)), max=float(err.max()),
                scale=s, aligned=aligned, err=err)


def rpe(est: np.ndarray, gt: np.ndarray, delta: int = 10,
        R_est: Optional[np.ndarray] = None, R_gt: Optional[np.ndarray] = None):
    """
    Relative Pose Error: ошибка перемещения на окне delta кадров.

    Траектории живут в разных системах координат, поэтому вычитать смещения
    напрямую нельзя. Если переданы ориентации, смещения переводятся в локальный
    фрейм каждого кадра (корректная инвариантная форма). Иначе выполняется
    глобальное выравнивание Умеямы -- этого достаточно для сравнения методов
    между собой, но абсолютные числа слегка занижены.
    """
    if R_est is not None and R_gt is not None:
        de = np.einsum("nij,nj->ni", np.transpose(R_est[:-delta], (0, 2, 1)),
                       est[delta:] - est[:-delta])
        dg = np.einsum("nij,nj->ni", np.transpose(R_gt[:-delta], (0, 2, 1)),
                       gt[delta:] - gt[:-delta])
    else:
        R, t, s = umeyama(est, gt, with_scale=True)
        a = (s * (R @ est.T).T) + t
        de = a[delta:] - a[:-delta]
        dg = gt[delta:] - gt[:-delta]
    err = np.linalg.norm(de - dg, axis=1)
    dist = np.linalg.norm(dg, axis=1)
    return dict(rmse=float(np.sqrt((err**2).mean())),
                median=float(np.median(err)),
                pct=float(100 * np.median(err / np.maximum(dist, 1e-6))))


def drift_vs_length(est: np.ndarray, gt: np.ndarray, lengths_m=(250, 500, 1000, 2000, 4000),
                    step: int = 25):
    """
    Дрейф в процентах как функция длины подтраектории (стиль KITTI).
    Для каждой длины берём все подотрезки, выравниваем по началу, меряем
    конечное расхождение, нормируем на пройденный путь.
    """
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    out = []
    for L in lengths_m:
        errs = []
        for i in range(0, len(gt) - 1, step):
            j = int(np.searchsorted(arc, arc[i] + L))
            if j >= len(gt):
                break
            R, t, s = umeyama(est[i:j + 1], gt[i:j + 1], with_scale=True)
            a = (s * (R @ est[i:j + 1].T).T) + t
            e = float(np.linalg.norm(a[-1] - gt[j]))
            if np.isfinite(e):
                errs.append(e)
        if errs:
            med = float(np.median(errs))
            out.append(dict(length_m=L, n=len(errs), median_err_m=med,
                            drift_pct=float(100 * med / L)))
    return out


def predicted_rotation_bias(R_enu_cam: np.ndarray, dxy_world: np.ndarray,
                            plane_dist: float) -> np.ndarray:
    """
    Прогноз ложного поворота, который выдаёт модель подобия на наклонной сцене.

    Истинное преобразование между кадрами для плоскости -- гомография
    H_e = R + t n^T / d. Аффинная часть t n^T / d содержит антисимметричную
    компоненту (t_x n_y - t_y n_x) / (2d), то есть сдвиговую деформацию.
    Модель подобия сдвига не имеет и вынуждена списать половину его на вращение.

    Ключевое: n -- нормаль земли в системе КАМЕРЫ, поэтому величина зависит от
    крена относительно направления полёта. Чистый тангаж вклада не даёт.
    При постоянном крене в крейсерском полёте смещение систематическое.

    R_enu_cam -- (N,3,3) ориентация камеры в ENU
    dxy_world -- (N,2) приращение easting/northing, метры
    plane_dist -- расстояние до земли, метры

    Возвращает (N,) прогноз в радианах, в том же знаке, что и theta из
    similarity_between -- то есть как поворот СЦЕНЫ в кадре. Поворот камеры
    противоположен по знаку.
    """
    up = np.array([0.0, 0.0, 1.0])
    n_cam = -np.einsum("nji,j->ni", R_enu_cam, up)          # нормаль земли в камере
    t_w = np.c_[dxy_world[:, 0], dxy_world[:, 1], np.zeros(len(dxy_world))]
    d_cam = np.einsum("nji,nj->ni", R_enu_cam, t_w)          # смещение камеры в её системе
    return (d_cam[:, 0] * n_cam[:, 1] - d_cam[:, 1] * n_cam[:, 0]) / (2.0 * plane_dist)


def undistort_matches(m: Matches, K: np.ndarray, dist) -> Matches:
    """
    Приводит точки к идеальной pinhole-камере. Изображения ALTO не исправлены,
    а объектив 3.5 мм при FOV 75° имеет заметную дисторсию, которая протекает
    в оценку вращения. Точки исправлять дешевле, чем изображения.
    """
    dist = np.asarray(dist, float).ravel() if dist is not None else None
    if dist is None or not np.any(dist) or m.n == 0:
        return m          # cv2.undistortPoints падает на пустом массиве
    f = lambda p: cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64),
                                      K, dist, P=K).reshape(-1, 2).astype(np.float32)
    return Matches(f(m.pts0), f(m.pts1), m.n_kpts0, m.n_kpts1,
                   m.t_detect, m.t_match, m.peak_vram_mb)


def solve_hybrid(m: Matches, K: np.ndarray, gsd: float, thresh: float = 1.0,
                 method: str = "ransac", agl: Optional[float] = None,
                 principal_point=(250.0, 250.0)) -> RelPose:
    """
    Вращение и направление трансляции -- из эссенциальной матрицы,
    длина шага -- из сдвига в кадре через GSD.

    Смысл: на этом датасете эссенциальная матрица даёт лучшее вращение
    (рельефа хватает для параллакса, и она не страдает от сдвиговой
    деформации, которая портит подобие), но масштаб ей взять неоткуда.
    Подобие масштаб знает точно, а вращение портит. Комбинация берёт
    от каждой сильную сторону и, в отличие от 'essential' с политикой 'gt',
    не подглядывает в ground truth.
    """
    t0 = time.perf_counter()
    re_ = solve_essential(m, K, thresh=thresh, method=method)
    if not re_.ok:
        return re_
    rs = solve_similarity(m, gsd, thresh=3.0, agl=agl, principal_point=principal_point)
    if not rs.ok:
        return re_
    mag = float(np.linalg.norm(rs.t[:2]))
    t = re_.t / (np.linalg.norm(re_.t) + 1e-12) * mag
    return RelPose(re_.R, t, re_.inliers, True, time.perf_counter() - t0,
                   scale_known=True, extra=dict(mag_from_similarity=mag))
