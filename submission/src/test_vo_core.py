"""Синтетическая проверка соглашений vo_core: надирная камера над плоскостью."""
import numpy as np, cv2, sys
sys.path.insert(0, "/home/claude/alto_vo")
import vo_core as vc

rng = np.random.default_rng(0)
F, C = 324.0, 250.0
K = np.array([[F, 0, C], [0, F, C], [0, 0, 1.0]])
AGL = 172.0
GSD = AGL / F

R_NADIR = np.diag([1.0, -1.0, -1.0])


def pose_wc(cx, cy, h, yaw):
    Rz = cv2.Rodrigues(np.array([0, 0, yaw]))[0]
    return vc.se3(Rz @ R_NADIR, [cx, cy, h])


def project(Pw, Twc):
    Pc = (vc.se3_inv(Twc) @ np.c_[Pw, np.ones(len(Pw))].T)[:3].T
    uv = (K @ Pc.T).T
    return uv[:, :2] / uv[:, 2:3], Pc[:, 2]


def make_pair(dx, dy, dh, dyaw, relief=0.0, npts=800, noise=0.3):
    T0 = pose_wc(0, 0, AGL, 0.0)
    T1 = pose_wc(dx, dy, AGL + dh, dyaw)
    span = AGL / F * 250 * 1.6
    Pw = np.c_[rng.uniform(-span, span, npts), rng.uniform(-span, span, npts),
               rng.uniform(-relief, relief, npts)]
    uv0, z0 = project(Pw, T0)
    uv1, z1 = project(Pw, T1)
    ok = (z0 > 1) & (z1 > 1)
    ok &= (uv0 > 0).all(1) & (uv0 < 500).all(1) & (uv1 > 0).all(1) & (uv1 < 500).all(1)
    m = vc.Matches((uv0[ok] + rng.normal(0, noise, (ok.sum(), 2))).astype(np.float32),
                   (uv1[ok] + rng.normal(0, noise, (ok.sum(), 2))).astype(np.float32))
    T10 = vc.se3_inv(T1) @ T0
    return m, T10, T0, T1


def report(name, rp, T10_true, metric=True):
    if not rp.ok:
        print(f"  {name:12s} ОТКАЗ"); return
    Rt, tt = T10_true[:3, :3], T10_true[:3, 3]
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(rp.R.T @ Rt)[0]))
    if metric:
        te = np.linalg.norm(rp.t - tt)
        extra = f"|t| ош = {te:6.3f} м   t={np.round(rp.t,2)} vs {np.round(tt,2)}"
    else:
        cos = rp.t @ tt / (np.linalg.norm(rp.t) * np.linalg.norm(tt) + 1e-12)
        extra = f"угол напр. t = {np.degrees(np.arccos(np.clip(cos,-1,1))):6.2f}°"
    print(f"  {name:12s} ош. вращения = {ang:5.2f}°   {extra}   inl={rp.inlier_ratio:.2f}")


print("=" * 78)
print("A. Рельеф 15 м (есть параллакс), stride 20: dx=55 м, dyaw=1°")
m, T10, *_ = make_pair(55, 0, 0, np.radians(1.0), relief=15.0)
report("essential", vc.solve_essential(m, K, 1.0), T10, metric=False)
report("homography", vc.solve_homography(m, K, 3.0, plane_dist=AGL), T10)
report("similarity", vc.solve_similarity(m, GSD, 3.0, agl=AGL, principal_point=(C,C)), T10)

print()
print("B. Идеальная плоскость (вырождение Essential), stride 20")
m, T10, *_ = make_pair(55, 0, 0, np.radians(1.0), relief=0.0)
report("essential", vc.solve_essential(m, K, 1.0), T10, metric=False)
report("homography", vc.solve_homography(m, K, 3.0, plane_dist=AGL), T10)
report("similarity", vc.solve_similarity(m, GSD, 3.0, agl=AGL, principal_point=(C,C)), T10)

print()
print("C. Плоскость, малая база (stride 1: dx=2.7 м)")
m, T10, *_ = make_pair(2.7, 0, 0, np.radians(0.05), relief=0.0)
report("essential", vc.solve_essential(m, K, 1.0), T10, metric=False)
report("homography", vc.solve_homography(m, K, 3.0, plane_dist=AGL), T10)
report("similarity", vc.solve_similarity(m, GSD, 3.0, agl=AGL, principal_point=(C,C)), T10)

print()
print("D. Со снижением: dx=55, dy=-12, dh=-6 м, dyaw=2°")
m, T10, *_ = make_pair(55, -12, -6, np.radians(2.0), relief=8.0)
report("homography", vc.solve_homography(m, K, 3.0, plane_dist=AGL), T10)
report("similarity", vc.solve_similarity(m, GSD, 3.0, agl=AGL, principal_point=(C,C)), T10)

print()
print("=" * 78)
print("E. Замкнутый цикл интегратора: дуга 60 кадров с поворотом")
N, step, dyaw = 60, 55.0, np.radians(1.5)
T = pose_wc(0, 0, AGL, 0.0)
gt = [T[:3, 3].copy()]
integ_h = vc.Integrator("native")
integ_s = vc.Integrator("native")
integ_e = vc.Integrator("gt")
for k in range(N):
    yaw0 = k * dyaw
    T0 = pose_wc(*gt[-1][:2], AGL, yaw0)
    hdg = np.array([np.cos(yaw0), np.sin(yaw0)]) * step
    c1 = gt[-1][:2] + hdg
    T1 = pose_wc(*c1, AGL, yaw0 + dyaw)
    gt.append(np.array([*c1, AGL]))
    span = AGL / F * 250 * 1.6
    Pw = np.c_[rng.uniform(-span, span, 900) + gt[-2][0],
               rng.uniform(-span, span, 900) + gt[-2][1],
               rng.uniform(-10, 10, 900)]
    uv0, z0 = project(Pw, T0); uv1, z1 = project(Pw, T1)
    ok = (z0 > 1) & (z1 > 1) & (uv0 > 0).all(1) & (uv0 < 500).all(1) & (uv1 > 0).all(1) & (uv1 < 500).all(1)
    m = vc.Matches((uv0[ok] + rng.normal(0, .3, (ok.sum(), 2))).astype(np.float32),
                   (uv1[ok] + rng.normal(0, .3, (ok.sum(), 2))).astype(np.float32))
    integ_h.step(vc.solve_homography(m, K, 3.0, plane_dist=AGL))
    integ_s.step(vc.solve_similarity(m, GSD, 3.0, agl=AGL, principal_point=(C,C)))
    integ_e.step(vc.solve_essential(m, K, 1.0), gt_step=step)

gt = np.array(gt)
for nm, ig in [("homography", integ_h), ("similarity", integ_s), ("essential(gt-scale)", integ_e)]:
    p = ig.positions
    a = vc.ate(p, gt, with_scale=False)
    print(f"  {nm:22s} ATE rmse = {a['rmse']:8.3f} м  (путь {N*step:.0f} м, "
          f"дрейф {100*a['rmse']/(N*step):.3f} %)")
print("\nRPE(10) для homography:", {k: round(v, 3) for k, v in vc.rpe(integ_h.positions, gt, 10).items()})

print("\nRPE(10), исправленный:")
for nm, ig in [("homography", integ_h), ("similarity", integ_s), ("essential(gt)", integ_e)]:
    print(f"  {nm:14s}", {k: round(v, 3) for k, v in vc.rpe(ig.positions, gt, 10).items()})
print("\ndrift_vs_length (homography):")
for r in vc.drift_vs_length(integ_h.positions, gt, lengths_m=(500, 1000, 2000), step=5):
    print("  ", r)
