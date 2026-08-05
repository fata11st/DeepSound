"""
pose_auc -- воспроизведение метрики "AUC of pose error @ 5/10/20 deg" на ALTO.

Метрика взята из статьи
    H. Zhang, H. He, Y. Zhou, F. Lei.
    "Efficient image matching for UAV visual navigation via DALGlue".
    Scientific Reports 15, 37684 (2025). doi:10.1038/s41598-025-21602-5

Статья репортит AUC@5/10/20 на MegaDepth-1500 и ссылается на протокол
SuperGlue/LoFTR. Само определение в тексте статьи неполно (не сказано, как
считается площадь под кривой и что делать с отказами оценщика), поэтому
здесь воспроизведена каноническая реализация из репозитория SuperGlue,
на которую опираются и LightGlue, и LoFTR, и сама статья:

    err_R  = arccos( (tr(R_est^T R_gt) - 1) / 2 )                  [градусы]
    err_t  = arccos( <t_est, t_gt> / (|t_est| |t_gt|) )            [градусы]
    err_t <- min(err_t, 180 - err_t)          # знак t из E неоднозначен
    err    = max(err_R, err_t)
    AUC@X  = (1/X) * integral_0^X  F(e) de,
             где F(e) -- доля пар с err <= e (эмпирическая функция распределения)

Все допущения, сделанные при воспроизведении, перечислены в ASSUMPTIONS ниже
и попадают в раздел Discussions отчёта.

Что здесь есть:
  * pose_auc / rel_pose_error   -- сама метрика;
  * gt_rel_pose                 -- GT-относительная поза из телеметрии ALTO;
  * calibrate_extrinsic         -- поиск постоянного разворота между системой
                                   координат кватерниона и системой OpenCV;
  * evaluate_pairs              -- прогон одного фронтенда по набору пар;
  * epipolar_precision          -- аналог MMA/Precision из статьи, но по
                                   эпиполярной невязке (глубин у нас нет).
"""
from __future__ import annotations

import itertools
import time
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

# numpy 2.x переименовал trapz -> trapezoid, а 1.x знает только trapz
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

# Решатели эссенциальной матрицы. USAC_ACCURATE -- ближайший доступный в OpenCV
# аналог LO-RANSAC, которым пользуется протокол SURE: тот же цикл RANSAC плюс
# локальная оптимизация на инлаерах. Точного LO-RANSAC из PoseLib в OpenCV нет,
# и это расхождение с протоколом статьи зафиксировано в допущениях.
RANSAC_METHODS = {
    "ransac": cv2.RANSAC,
    "magsac": cv2.USAC_MAGSAC,
    "lo": cv2.USAC_ACCURATE,
    "usac_default": cv2.USAC_DEFAULT,
}


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    """Матрица камеры после масштабирования кадра в s раз."""
    Ks = K.astype(float).copy()
    Ks[:2] *= s
    return Ks


def resized_gray(gray, size: int, interp=cv2.INTER_CUBIC):
    """
    Обёртка над источником кадров, отдающая их в другом разрешении.

    Нужна для проверки гипотезы о влиянии разрешения на метрику: публикации
    считают AUC на 832-1200 px, у нас 500. Апскейл не добавляет информации,
    поэтому любое изменение метрики -- это чувствительность самих детекторов
    и порога RANSAC к масштабу, а не улучшение данных.
    """
    return lambda i: cv2.resize(gray(i), (size, size), interpolation=interp)


ASSUMPTIONS = """
Допущения при воспроизведении метрики (для раздела Discussions):
 1. Площадь под кривой считается по канонической реализации SuperGlue
    (кусочно-линейная интерполяция ЭФР, нормировка на порог). В статье
    формула AUC не выписана.
 2. Отказ оценщика (E не найдена, < 8 инлаеров) кодируется ошибкой 180 deg,
    то есть пара учитывается как полностью проваленная, а не выбрасывается.
    Это соглашение SuperGlue; в статье поведение при отказе не описано.
 3. err_t сворачивается как min(err_t, 180 - err_t): знак трансляции из
    эссенциальной матрицы определён с точностью до направления. Функция
    умеет считать и без свёртки (fold_translation=False) -- разница
    показывает, сколько пар развёрнуты на 180 deg.
 4. GT-поза берётся из телеметрии (UTM + кватернион), а не из SfM-реконструкции,
    как в MegaDepth. Точность GT ограничена GNSS/INS, а не BA.
 5. Пары с базой меньше min_baseline_m исключаются: при |t| -> 0 направление
    трансляции не определено и err_t вырождается в шум.
 6. Протокол SURE использует LO-RANSAC из PoseLib; в OpenCV его нет, ближайший
    аналог -- USAC_ACCURATE (RANSAC с локальной оптимизацией). Расхождение
    измерено отдельно: см. развёртку по решателю и порогу в ноутбуке.
 7. Кадры ALTO имеют разрешение 500x500, тогда как протоколы публикаций
    работают на 832-1200 px. Влияние этого фактора измеряется апскейлом,
    а не берётся на веру.
"""


# ============================================================== 1. сама метрика
def pose_auc(errors: Sequence[float],
             thresholds: Sequence[float] = (5.0, 10.0, 20.0)) -> list[float]:
    """
    AUC кумулятивной кривой ошибок -- реализация из SuperGlue (Sarlin et al.).

    errors -- ошибки позы в градусах, по одной на пару кадров;
              nan/inf трактуются как полный отказ (180 deg).
    Возвращает список значений AUC в долях единицы (умножить на 100 для %).
    """
    e = np.asarray(errors, dtype=float)
    e = np.where(np.isfinite(e), e, 180.0)
    if len(e) == 0:
        return [float("nan")] * len(thresholds)

    e = np.sort(e)
    recall = (np.arange(len(e)) + 1) / len(e)
    e = np.r_[0.0, e]                      # кривая обязана начинаться в нуле
    recall = np.r_[0.0, recall]

    out = []
    for t in thresholds:
        last = int(np.searchsorted(e, t))          # >= 1, так как e[0] == 0
        r = np.r_[recall[:last], recall[last - 1]]
        x = np.r_[e[:last], t]
        out.append(float(_TRAPZ(r, x=x) / t))
    return out


def rel_pose_error(R_est: np.ndarray, t_est: np.ndarray,
                   R_gt: np.ndarray, t_gt: np.ndarray,
                   fold_translation: bool = True) -> tuple[float, float]:
    """
    Угловые ошибки вращения и направления трансляции в градусах.

    Длина t не участвует нигде: в монокулярной постановке масштаб неизвестен,
    и метрика статьи специально устроена так, чтобы от него не зависеть.
    """
    n = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if n < 1e-12:
        err_t = 180.0                      # нулевая база -- направления нет
    else:
        cos_t = float(np.dot(np.ravel(t_est), np.ravel(t_gt)) / n)
        err_t = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
        if fold_translation:
            err_t = min(err_t, 180.0 - err_t)

    cos_R = (float(np.trace(R_est.T @ R_gt)) - 1.0) / 2.0
    err_R = float(np.degrees(np.arccos(np.clip(cos_R, -1.0, 1.0))))
    return err_R, err_t


def auc_row(err_R, err_t, thresholds=(5.0, 10.0, 20.0)) -> dict:
    """Сводка по массиву покадровых ошибок: AUC по max, по R и по t отдельно."""
    err_R = np.asarray(err_R, float)
    err_t = np.asarray(err_t, float)
    err = np.maximum(err_R, err_t)
    a = pose_auc(err, thresholds)
    aR = pose_auc(err_R, thresholds)
    aT = pose_auc(err_t, thresholds)
    out = {"пар": int(len(err))}
    for i, th in enumerate(thresholds):
        out[f"AUC@{th:g}"] = round(100 * a[i], 2)
    for i, th in enumerate(thresholds):
        out[f"AUC_R@{th:g}"] = round(100 * aR[i], 2)
    for i, th in enumerate(thresholds):
        out[f"AUC_t@{th:g}"] = round(100 * aT[i], 2)
    out["медиана_err_R"] = round(float(np.median(err_R)), 3)
    out["медиана_err_t"] = round(float(np.median(err_t)), 3)
    return out


# ================================================= 2. GT-поза из телеметрии ALTO
def signed_permutations() -> list[np.ndarray]:
    """24 матрицы поворота вида "переставить оси и поменять знаки" (det = +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[p, i] = signs[i]
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gt_rel_pose(c0: np.ndarray, c1: np.ndarray,
                Q0: np.ndarray, Q1: np.ndarray,
                P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Относительная поза кадр0 -> кадр1 в системе OpenCV.

    Модель: точка мира p_enu связана с точкой камеры p_cv как
        p_enu = Q_i @ P @ p_cv + c_i,
    где Q_i -- матрица кватерниона (телеметрийная система камеры -> ENU),
    P -- постоянный разворот между телеметрийной системой и системой OpenCV
    (x вправо, y вниз, z вперёд). Отсюда

        R = P^T Q1^T Q0 P,      t = P^T Q1^T (c0 - c1),

    что совпадает с соглашением vo_core: P1 = R @ P0 + t.
    """
    R = P.T @ Q1.T @ Q0 @ P
    t = P.T @ Q1.T @ (np.asarray(c0, float) - np.asarray(c1, float))
    return R, t


def _circmean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def calibrate_extrinsic(centers_enu: np.ndarray, Q: np.ndarray,
                        idx0: np.ndarray, idx1: np.ndarray,
                        shift_px: np.ndarray,
                        verbose: bool = True) -> dict:
    """
    Оценка постоянного разворота P между системой кватерниона и системой OpenCV.

    Зачем это нужно. Ошибка позы считается в системе камеры, поэтому GT-поза
    должна быть выражена в тех же осях, что и оценка из cv2.recoverPose.
    Постоянный разворот между "камерой телеметрии" и "камерой OpenCV" в
    относительную позу НЕ сокращается: R_gt переходит в P^T R P, и ошибка
    вращения меняется. Ошибка направления трансляции меняется тем более.

    Как оценивается, без подглядывания в оцениваемый пайплайн:
      1. перебираются 24 разворота "перестановка осей + знаки"; для каждого
         предсказывается направление сдвига сцены в кадре: при почти надирной
         камере смещение точки в пикселях сонаправлено с (t_x, t_y);
      2. предсказание сравнивается с ИЗМЕРЕННЫМ сдвигом shift_px, который
         берётся из 2D-подобия (ORB + estimateAffinePartial2D) -- это
         измерительный инструмент, а не один из сравниваемых методов;
      3. остаточный поворот вокруг оптической оси доводится непрерывной
         поправкой psi0 (крепление камеры относительно носа не обязано быть
         кратно 90 deg).

    centers_enu -- (N,3) центры камеры в локальном ENU, метры;
    Q           -- (N,3,3) матрицы кватернионов (телеметрийная камера -> ENU);
    idx0, idx1  -- индексы кадров в парах;
    shift_px    -- (M,2) измеренный сдвиг сцены в кадре, пиксели.

    Возвращает dict с P, psi0_deg, resultant (0..1 -- согласованность) и
    таблицей качества всех кандидатов.
    """
    centers_enu = np.asarray(centers_enu, float)
    shift_px = np.asarray(shift_px, float)
    beta = np.arctan2(shift_px[:, 1], shift_px[:, 0])       # измеренное направление

    rows = []
    for k, P in enumerate(signed_permutations()):
        # ось z камеры в ENU должна смотреть вниз: третий столбец Q@P ~ (0,0,-1)
        zdown = float(np.median((Q @ P)[:, 2, 2]))
        alpha = np.empty(len(idx0))
        for j, (i0, i1) in enumerate(zip(idx0, idx1)):
            _, t = gt_rel_pose(centers_enu[i0], centers_enu[i1], Q[i0], Q[i1], P)
            alpha[j] = np.arctan2(t[1], t[0])               # предсказанное направление
        d = beta - alpha
        R_len = float(np.abs(np.mean(np.exp(1j * d))))      # длина результирующего вектора
        rows.append(dict(k=k, z_down=round(zdown, 3), resultant=round(R_len, 4),
                         psi0_deg=round(float(np.degrees(_circmean(d))), 2)))

    cand = [r for r in rows if r["z_down"] < -0.9] or rows
    best = max(cand, key=lambda r: r["resultant"])
    P0 = signed_permutations()[best["k"]]
    psi0 = np.radians(best["psi0_deg"])
    P = P0 @ Rz(-psi0)

    if verbose:
        print(f"[калибровка] кандидат #{best['k']}: z_down={best['z_down']:+.3f}, "
              f"согласованность={best['resultant']:.3f}, psi0={best['psi0_deg']:+.2f} deg")
        if best["resultant"] < 0.9:
            print("[калибровка] ВНИМАНИЕ: согласованность низкая, GT-позы под вопросом")
    return dict(P=P, P_axis=P0, psi0_deg=float(np.degrees(psi0)),
                resultant=best["resultant"], z_down=best["z_down"], table=rows)


# ================================================ 3. вспомогательное для прогона
def sample_pairs(n_frames: int, stride: int, n_pairs: int,
                 start: int = 0, seed: int = 0) -> np.ndarray:
    """Равномерная выборка индексов первых кадров пар (i, i+stride)."""
    hi = n_frames - stride - 1
    if hi <= start:
        return np.zeros(0, int)
    if n_pairs >= hi - start:
        return np.arange(start, hi)
    return np.unique(np.linspace(start, hi, n_pairs).astype(int))


def epipolar_precision(pts0: np.ndarray, pts1: np.ndarray,
                       R_gt: np.ndarray, t_gt: np.ndarray, K: np.ndarray,
                       px_thresh: float = 1.0) -> float:
    """
    Доля соответствий, удовлетворяющих ИСТИННОЙ эпиполярной геометрии
    (симметричная невязка Сампсона < px_thresh пикселей).

    Это аналог Precision/MMA из статьи. В статье корректность матча
    проверяется по GT-гомографии (для HPatches) или по GT-глубине; у нас
    ни того, ни другого нет, поэтому проверка идёт по GT-эпиполярной линии.
    Метрика слабее (точка может лежать на линии и не быть верным матчем),
    что честно оговорено в отчёте.
    """
    if len(pts0) == 0:
        return float("nan")
    t = np.ravel(t_gt)
    if np.linalg.norm(t) < 1e-9:
        return float("nan")
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R_gt
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

    p0 = np.c_[pts0, np.ones(len(pts0))]
    p1 = np.c_[pts1, np.ones(len(pts1))]
    l1 = (F @ p0.T).T                       # эпиполярные линии в кадре 1
    l0 = (F.T @ p1.T).T
    num = np.einsum("ij,ij->i", p1, (F @ p0.T).T) ** 2
    den = l1[:, 0] ** 2 + l1[:, 1] ** 2 + l0[:, 0] ** 2 + l0[:, 1] ** 2
    sampson = num / np.maximum(den, 1e-12)
    return float(np.mean(sampson < px_thresh ** 2))


def evaluate_pairs(gray: Callable[[int], np.ndarray],
                   centers_enu: np.ndarray, Q: np.ndarray, P: np.ndarray,
                   pairs: Iterable[int], stride: int,
                   frontend, K: np.ndarray,
                   ransac_thresh_px: float = 1.0,
                   ransac_method: str = "magsac",
                   min_baseline_m: float = 1.0,
                   fold_translation: bool = True,
                   use_lk_init: bool = False,
                   precision_thresh_px: float = 1.0,
                   verbose_every: int = 0) -> "pd.DataFrame":
    """
    Прогон одного фронтенда по набору пар и покадровый расчёт ошибок позы.

    Оценщик у всех подходов ОДИН И ТОТ ЖЕ (эссенциальная матрица + RANSAC),
    как в статье: сравниваются фронтенды, а не геометрия. Возвращает
    pandas.DataFrame с покадровыми ошибками -- из него считается AUC.
    """
    import pandas as pd

    rows = []
    last_shift = None
    for n, i0 in enumerate(pairs):
        i1 = i0 + stride
        dc = centers_enu[i0] - centers_enu[i1]
        base = float(np.linalg.norm(dc))
        if base < min_baseline_m:
            continue

        R_gt, t_gt = gt_rel_pose(centers_enu[i0], centers_enu[i1], Q[i0], Q[i1], P)

        img0, img1 = gray(i0), gray(i1)
        t_start = time.perf_counter()
        if use_lk_init and last_shift is not None:
            m = frontend(img0, img1, init_shift=last_shift)
        else:
            m = frontend(img0, img1)
        t_front = time.perf_counter() - t_start

        if m.n >= 8:
            E, mask = cv2.findEssentialMat(
                m.pts0, m.pts1, K, prob=0.9999, threshold=ransac_thresh_px,
                method=RANSAC_METHODS[ransac_method])
        else:
            E, mask = None, None

        if E is None or E.shape != (3, 3):
            ok, err_R, err_t, inl = False, 180.0, 180.0, 0.0
            R_est = np.eye(3)
        else:
            n_in, R_est, t_est, mask_pose = cv2.recoverPose(E, m.pts0, m.pts1, K, mask=mask)
            ok = n_in >= 8
            inl = float(np.mean(mask_pose.ravel() > 0))
            if ok:
                err_R, err_t = rel_pose_error(R_est, t_est.ravel(), R_gt, t_gt,
                                              fold_translation)
            else:
                err_R, err_t = 180.0, 180.0

        if m.n > 0:
            med = np.median(m.pts1 - m.pts0, axis=0)
            last_shift = med.astype(np.float32)

        rows.append(dict(
            i0=int(i0), i1=int(i1), база_м=round(base, 2),
            матчей=int(m.n), inl=round(inl, 3), ok=bool(ok),
            err_R=round(err_R, 4), err_t=round(err_t, 4),
            err=round(max(err_R, err_t), 4),
            prec=round(epipolar_precision(m.pts0, m.pts1, R_gt, t_gt, K,
                                          precision_thresh_px), 4) if m.n else np.nan,
            мс_фронтенд=round(1000 * t_front, 2),
        ))
        if verbose_every and (n + 1) % verbose_every == 0:
            print(f"  {n + 1} пар...", end="\r")
    return pd.DataFrame(rows)


# ================================================== 4. синтетическая самопроверка
def synthetic_check(seed: int = 0, verbose: bool = True) -> dict:
    """
    Проверка метрики на данных с известным ответом (без ALTO).

    Три теста:
      A. аналитические свойства AUC (нулевые ошибки -> 1.0, равномерные -> 0.5);
      B. rel_pose_error на паре с точно известным относительным поворотом;
      C. полный путь "надирная сцена -> матчи -> E -> ошибка позы": ошибки
         должны быть малыми, AUC@5 -- близким к единице.
    """
    rng = np.random.default_rng(seed)
    out = {}

    # --- A
    out["auc_zero"] = pose_auc(np.zeros(100), (5, 10, 20))
    out["auc_uniform20"] = pose_auc(np.linspace(0, 20, 20001), (5, 10, 20))
    out["auc_all_fail"] = pose_auc(np.full(100, np.inf), (5, 10, 20))

    # --- B: поворот на 3 градуса вокруг оси z и трансляция под 30 градусов
    ang = np.radians(3.0)
    R_gt = Rz(ang)
    t_gt = np.array([np.cos(np.radians(30.0)), np.sin(np.radians(30.0)), 0.0])
    eR, eT = rel_pose_error(np.eye(3), np.array([1.0, 0.0, 0.0]), R_gt, t_gt)
    out["err_R_should_be_3"] = round(eR, 6)
    out["err_t_should_be_30"] = round(eT, 6)

    # --- C: надирная камера над рельефом, известные позы
    F, C, AGL = 324.07, 250.0, 194.8
    K = np.array([[F, 0, C], [0, F, C], [0, 0, 1.0]])
    R_NADIR = np.diag([1.0, -1.0, -1.0])

    def pose_wc(cx, cy, h, yaw):
        T = np.eye(4)
        T[:3, :3] = Rz(yaw) @ R_NADIR
        T[:3, 3] = [cx, cy, h]
        return T

    errs_R, errs_t = [], []
    for _ in range(40):
        dx = rng.uniform(40, 70)
        dyaw = np.radians(rng.uniform(-2, 2))
        T0, T1 = pose_wc(0, 0, AGL, 0.0), pose_wc(dx, 0, AGL, dyaw)
        span = AGL / F * 250 * 1.6
        Pw = np.c_[rng.uniform(-span, span, 900), rng.uniform(-span, span, 900),
                   rng.uniform(-15, 15, 900)]

        def proj(T):
            Ti = np.eye(4)
            Ti[:3, :3] = T[:3, :3].T
            Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
            Pc = (Ti @ np.c_[Pw, np.ones(len(Pw))].T)[:3].T
            uv = (K @ Pc.T).T
            return uv[:, :2] / uv[:, 2:3], Pc[:, 2]

        uv0, z0 = proj(T0)
        uv1, z1 = proj(T1)
        ok = (z0 > 1) & (z1 > 1) & (uv0 > 0).all(1) & (uv0 < 500).all(1) \
             & (uv1 > 0).all(1) & (uv1 < 500).all(1)
        p0 = uv0[ok] + rng.normal(0, 0.5, (ok.sum(), 2))
        p1 = uv1[ok] + rng.normal(0, 0.5, (ok.sum(), 2))
        # 20 % выбросов -- как в реальном фронтенде
        n_out = int(0.2 * len(p0))
        p1[:n_out] = rng.uniform(0, 500, (n_out, 2))

        R_gt = T1[:3, :3].T @ T0[:3, :3]
        t_gt = T1[:3, :3].T @ (T0[:3, 3] - T1[:3, 3])

        E, mask = cv2.findEssentialMat(p0.astype(np.float32), p1.astype(np.float32),
                                       K, method=cv2.USAC_MAGSAC, prob=0.9999,
                                       threshold=1.0)
        _, R_est, t_est, _ = cv2.recoverPose(E, p0.astype(np.float32),
                                             p1.astype(np.float32), K, mask=mask)
        eR, eT = rel_pose_error(R_est, t_est.ravel(), R_gt, t_gt)
        errs_R.append(eR)
        errs_t.append(eT)

    out["synth_median_err_R"] = round(float(np.median(errs_R)), 3)
    out["synth_median_err_t"] = round(float(np.median(errs_t)), 3)
    out["synth_auc"] = [round(100 * v, 2)
                        for v in pose_auc(np.maximum(errs_R, errs_t), (5, 10, 20))]

    if verbose:
        print("A. AUC(нулевые ошибки)       =", [round(v, 4) for v in out["auc_zero"]],
              "  ожидается [1, 1, 1]")
        print("A. AUC(равномерные на [0,20])=", [round(v, 4) for v in out["auc_uniform20"]],
              "  ожидается [0.125, 0.25, 0.5]")
        print("A. AUC(все отказы)           =", [round(v, 4) for v in out["auc_all_fail"]],
              "  ожидается [0, 0, 0]")
        print(f"B. err_R = {out['err_R_should_be_3']} deg (ожидается 3), "
              f"err_t = {out['err_t_should_be_30']} deg (ожидается 30)")
        print(f"C. синтетика с 20 % выбросов: медиана err_R = "
              f"{out['synth_median_err_R']} deg, err_t = {out['synth_median_err_t']} deg, "
              f"AUC@5/10/20 = {out['synth_auc']}")
    return out


if __name__ == "__main__":
    print(ASSUMPTIONS)
    synthetic_check()
