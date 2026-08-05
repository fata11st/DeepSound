"""
Синтетическая проверка pose_auc: метрика + калибровка внешних параметров.

Запуск:  python test_pose_auc.py
Данные ALTO не нужны -- всё генерируется с заранее известным ответом.
"""
import numpy as np

import pose_auc as pa

F, C, AGL = 324.07, 250.0, 194.8
K = np.array([[F, 0, C], [0, F, C], [0, 0, 1.0]])
R_NADIR = np.diag([1.0, -1.0, -1.0])          # камера смотрит вниз, x на восток при psi=0


def make_flight(n=120, step=2.74, seed=3):
    """Полёт по дуге с плавным изменением курса и лёгкими креном/тангажом."""
    rng = np.random.default_rng(seed)
    psi = np.cumsum(np.full(n, np.radians(0.06))) + np.radians(20.0)
    c = np.zeros((n, 3))
    c[:, 2] = AGL + np.cumsum(rng.normal(0, 0.05, n))
    for i in range(1, n):
        c[i, :2] = c[i - 1, :2] + step * np.array([np.sin(psi[i]), np.cos(psi[i])])
    R_enu_cv = np.empty((n, 3, 3))
    for i in range(n):
        roll = np.radians(1.5 * np.sin(i / 17.0))     # крен и тангаж крейсерского полёта
        pitch = np.radians(1.0 * np.cos(i / 23.0))
        Rr = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Rp = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_enu_cv[i] = pa.Rz(-psi[i]) @ R_NADIR @ Rr @ Rp
    return c, R_enu_cv


def measured_shift(c, R_enu_cv, i0, i1, rng, npts=400, noise=0.4):
    """Медианный поток между кадрами: имитация того, что меряет ORB + подобие."""
    Pw = np.c_[rng.uniform(-160, 160, npts) + c[i0, 0],
               rng.uniform(-160, 160, npts) + c[i0, 1],
               rng.uniform(-8, 8, npts)]

    def proj(i):
        Rc = R_enu_cv[i]
        Pc = (Pw - c[i]) @ Rc                       # = Rc^T (Pw - c)
        uv = (K @ Pc.T).T
        return uv[:, :2] / uv[:, 2:3], Pc[:, 2]

    uv0, z0 = proj(i0)
    uv1, z1 = proj(i1)
    ok = (z0 > 1) & (z1 > 1) & (uv0 > 0).all(1) & (uv0 < 500).all(1) \
        & (uv1 > 0).all(1) & (uv1 < 500).all(1)
    d = (uv1[ok] - uv0[ok]) + rng.normal(0, noise, (ok.sum(), 2))
    return np.median(d, axis=0)


def main():
    print("=" * 78)
    print("1. Метрика на данных с известным ответом")
    print("=" * 78)
    pa.synthetic_check()

    print()
    print("=" * 78)
    print("2. Калибровка внешних параметров (постоянный разворот P)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    c, R_enu_cv = make_flight()

    # истинный разворот: перестановка осей + разворот крепления на 17 градусов
    P_true = pa.signed_permutations()[7] @ pa.Rz(np.radians(17.0))
    Q = np.array([R_enu_cv[i] @ P_true.T for i in range(len(c))])   # то, что "лежит в телеметрии"

    stride = 20
    i0 = np.arange(0, len(c) - stride - 1, 3)
    i1 = i0 + stride
    shift = np.array([measured_shift(c, R_enu_cv, a, b, rng) for a, b in zip(i0, i1)])

    cal = pa.calibrate_extrinsic(c, Q, i0, i1, shift)
    P = cal["P"]

    errs = []
    for a, b in zip(i0, i1):
        R_true, t_true = pa.gt_rel_pose(c[a], c[b], Q[a], Q[b], P_true)
        R_hat, t_hat = pa.gt_rel_pose(c[a], c[b], Q[a], Q[b], P)
        errs.append(pa.rel_pose_error(R_hat, t_hat, R_true, t_true))
    errs = np.array(errs)
    print(f"  расхождение GT-поз при восстановленном P: "
          f"вращение {np.median(errs[:, 0]):.3f} deg, направление t {np.median(errs[:, 1]):.3f} deg")
    ok = np.median(errs[:, 0]) < 1.0 and np.median(errs[:, 1]) < 2.0
    print("  РЕЗУЛЬТАТ:", "калибровка восстанавливает P" if ok else "ПРОВАЛ")

    print()
    print("=" * 78)
    print("3. Влияние некалиброванного P на метрику (почему шаг 2 обязателен)")
    print("=" * 78)
    for name, P_bad in [("без поправки psi0", cal["P_axis"]),
                        ("чужая перестановка", pa.signed_permutations()[0])]:
        e = []
        for a, b in zip(i0, i1):
            R_true, t_true = pa.gt_rel_pose(c[a], c[b], Q[a], Q[b], P_true)
            R_hat, t_hat = pa.gt_rel_pose(c[a], c[b], Q[a], Q[b], P_bad)
            e.append(pa.rel_pose_error(R_hat, t_hat, R_true, t_true))
        e = np.array(e)
        auc = pa.pose_auc(np.maximum(e[:, 0], e[:, 1]), (5, 10, 20))
        print(f"  {name:22s}: err_R={np.median(e[:, 0]):6.2f} deg, "
              f"err_t={np.median(e[:, 1]):6.2f} deg, "
              f"AUC@5/10/20={[round(100 * v, 1) for v in auc]}")


if __name__ == "__main__":
    main()
