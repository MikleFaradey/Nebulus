# -*- coding: utf-8 -*-
"""
AUV Survey GUI — табличные параметры + симуляция, переходные с 5% трубкой, Mp и Ts.
Зависимости: PyQt6, matplotlib
pip install pyqt6 matplotlib
"""

import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# ---- Matplotlib (Qt backend) ----
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# ---- PyQt6 ----
from PyQt6 import QtWidgets, QtGui, QtCore

# ---- Симуляция (ядро) ----
import matplotlib.patches as mpatches

try:
    import serial  # type: ignore
except Exception:
    serial = None

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


def wrap_rad(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


# =========================
# КУРС (Yaw) с течением: сила+угол
# =========================
class Kurs:
    def __init__(
        self,
        Jy=90.0, L55=87.0,
        Cwy1=75.0, Cwy2=7.4,
        Tdv=0.15, Kdv=20.0, b=0.3,
        Umax=10.0,
        K1=0.6772091636,   # В/град
        K2=0.6157978918,   # В/(град/с)
        # Возмущения
        Mdist=0.0,
        Fcur_yaw_mag=0.0,
        Fcur_yaw_dir_deg=0.0,
        l_yaw_arm=1.0,
        # Измерения
        use_5pct_error_psi=False,
        use_5pct_error_r=False,
        meas_scale=1.05,
        dt=0.02,
    ):
        self.Jy, self.L55 = Jy, L55
        self.Cwy1, self.Cwy2 = Cwy1, Cwy2
        self.Tdv, self.Kdv, self.b = Tdv, Kdv, b
        self.Km = 2.0 * b * Kdv
        self.Umax = abs(Umax)
        self.K1, self.K2 = K1, K2
        # Возмущения
        self.Mdist = float(Mdist)
        self.Fcur_yaw = float(Fcur_yaw_mag)
        self.th_cur_yaw = float(Fcur_yaw_dir_deg) * DEG2RAD
        self.l_yaw_arm = float(l_yaw_arm)
        # Состояния
        self.dt = dt
        self.psi = 0.0
        self.wy = 0.0
        self.Mdy = 0.0
        # Измерения
        self.use_5pct_error_psi = use_5pct_error_psi
        self.use_5pct_error_r = use_5pct_error_r
        self.meas_scale = float(meas_scale)

    def _meas_psi(self, psi_true):
        return self.meas_scale * psi_true if self.use_5pct_error_psi else psi_true

    def _meas_r(self, r_true):
        return self.meas_scale * r_true if self.use_5pct_error_r else r_true

    def control_voltage(self, psi_rad, wy_rad_s, psi_dist_deg):
        psi_meas = self._meas_psi(psi_rad)
        r_meas = self._meas_r(wy_rad_s)
        e_deg = psi_dist_deg - psi_meas * RAD2DEG
        while e_deg <= -180.0:
            e_deg += 360.0
        while e_deg > 180.0:
            e_deg -= 360.0
        wy_deg_s = r_meas * RAD2DEG
        u = self.K1 * e_deg - self.K2 * wy_deg_s
        if u > self.Umax:
            u = self.Umax
        if u < -self.Umax:
            u = -self.Umax
        return u

    def f(self, psi, wy, Mdy, psi_dist_deg):
        Upsi = self.control_voltage(psi, wy, psi_dist_deg)
        dMdy = (self.Km * Upsi - Mdy) / self.Tdv
        Jtot = self.Jy + self.L55
        Mcur = self.l_yaw_arm * self.Fcur_yaw * math.sin(self.th_cur_yaw - psi)
        Mext = self.Mdist + Mcur
        dwy = (Mdy + Mext - self.Cwy1 * abs(wy) * wy - self.Cwy2 * wy) / Jtot
        dpsi = wy
        return dpsi, dwy, dMdy

    def rk4_step(self, psi_dist_deg):
        y1 = (self.psi, self.wy, self.Mdy)
        k1 = self.f(*y1, psi_dist_deg)
        y2 = tuple(y1[i] + 0.5 * self.dt * k1[i] for i in range(3))
        k2 = self.f(*y2, psi_dist_deg)
        y3 = tuple(y1[i] + 0.5 * self.dt * k2[i] for i in range(3))
        k3 = self.f(*y3, psi_dist_deg)
        y4 = tuple(y1[i] + self.dt * k3[i] for i in range(3))
        k4 = self.f(*y4, psi_dist_deg)
        self.psi = y1[0] + (self.dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.wy = y1[1] + (self.dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.Mdy = y1[2] + (self.dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.psi = wrap_rad(self.psi)

    def step(self, psi_dist_deg):
        self.rk4_step(psi_dist_deg)
        return self.psi * RAD2DEG


# =========================
# МАРШ (Surge) с течением: скорость+угол
# =========================
class MarshevPD:
    def __init__(
        self,
        m=90.0, lam11=78.0,
        Cxu1=37.0, Cxu2=3.7,
        Tdx=0.15, Kdv=20.0,
        Umax=10.0,
        Kp=1.98712, Kd=1.20176,
        at=12.0,
        wn_ref=1.2, zeta_ref=0.7,
        Ucur_mag=0.0, Ucur_dir_deg=0.0,
        use_5pct_error_u=False,
        meas_scale=1.05,
        dt=0.02,
    ):
        self.Meff = m + lam11
        self.C1, self.C2 = Cxu1, Cxu2
        self.Tdx = Tdx
        self.Kdx = 2.0 * Kdv
        self.Umax = abs(Umax)
        self.Kp = max(0.0, Kp)
        self.Kd = max(0.0, Kd)
        self.at = at
        self.wn = max(1e-6, wn_ref)
        self.zr = max(0.2, zeta_ref)
        self.Ucur_mag = float(Ucur_mag)
        self.Ucur_dir = float(Ucur_dir_deg) * DEG2RAD
        self.use_5pct_error_u = use_5pct_error_u
        self.meas_scale = float(meas_scale)
        self.dt = dt
        self.u = 0.0
        self.x = 0.0
        self.T = 0.0
        self.ur = 0.0
        self.urd = 0.0

    def _update_ref_filter(self, u_ref):
        ydd = self.wn * self.wn * (u_ref - self.ur) - 2.0 * self.zr * self.wn * self.urd
        self.urd += self.dt * ydd
        self.ur += self.dt * self.urd
        return self.ur, self.urd

    def _drag(self, u):
        return self.C1 * abs(u) * u + self.C2 * u

    def _u_dot(self, u, T):
        return (T - self._drag(u)) / self.Meff

    def _ucmd_from_Tcmd(self, Tcmd, T):
        u_unsat = (self.Tdx / self.Kdx) * (self.at * (Tcmd - T)) + T / self.Kdx
        if u_unsat > self.Umax:
            u_unsat = self.Umax
        if u_unsat < -self.Umax:
            u_unsat = -self.Umax
        return u_unsat

    def f(self, u, T, x, u_ref_ground, psi):
        Uc_par = self.Ucur_mag * math.cos(self.Ucur_dir - psi)
        u_ref_body = max(0.0, u_ref_ground - Uc_par)
        uref, uref_dot = self._update_ref_filter(u_ref_body)
        udot = self._u_dot(u, T)
        u_meas = self.meas_scale * u if self.use_5pct_error_u else u
        e = uref - u_meas
        edot = uref_dot - udot
        Tff = self.C2 * u_meas + self.C1 * abs(u_meas) * u_meas
        Tcmd = self.Meff * (self.Kp * e + self.Kd * edot) + Tff
        U = self._ucmd_from_Tcmd(Tcmd, T)
        dT = (self.Kdx * U - T) / self.Tdx
        du = udot
        dx = u
        return du, dT, dx

    def rk4_step(self, u_ref_ground, psi):
        u1, T1, x1 = self.u, self.T, self.x
        k1 = self.f(u1, T1, x1, u_ref_ground, psi)
        u2 = u1 + 0.5 * self.dt * k1[0]
        T2 = T1 + 0.5 * self.dt * k1[1]
        x2 = x1 + 0.5 * self.dt * k1[2]
        k2 = self.f(u2, T2, x2, u_ref_ground, psi)
        u3 = u1 + 0.5 * self.dt * k2[0]
        T3 = T1 + 0.5 * self.dt * k2[1]
        x3 = x1 + 0.5 * self.dt * k2[2]
        k3 = self.f(u3, T3, x3, u_ref_ground, psi)
        u4 = u1 + self.dt * k3[0]
        T4 = T1 + self.dt * k3[1]
        x4 = x1 + self.dt * k3[2]
        k4 = self.f(u4, T4, x4, u_ref_ground, psi)
        self.u = u1 + (self.dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.T = T1 + (self.dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.x = x1 + (self.dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

    def step(self, u_ref_ground, psi=0.0):
        self.rk4_step(u_ref_ground, psi)
        return self.u


# =========================
# Утилиты для маршрута/SSS
# =========================
def sss_swath_and_spacing(R, h, beta_deg=4.0, cross_overlap=0.15) -> Tuple[float, float]:
    if R <= h:
        raise ValueError("R_slant должен быть > h_alt.")
    beta = math.radians(beta_deg)
    W = 2.0 * math.sqrt(max(0.0, R * R - h * h)) - 2.0 * h * math.tan(beta / 2.0)
    W = max(0.0, W)
    S_lane = (1.0 - cross_overlap) * W
    return W, S_lane


def build_lawnmower_in_rect_from_corner(rect, lane_spacing, heading_deg, margin=0.0, eps=1e-6):
    xmin, ymin, W, H = rect
    xmin += margin
    ymin += margin
    xmax = xmin + (W - 2 * margin)
    ymax = ymin + (H - 2 * margin)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Слишком большой margin.")
    p0 = (xmin, ymin)
    th = math.radians(heading_deg)
    vx, vy = math.cos(th), math.sin(th)
    nx, ny = -vy, vx
    if nx < 0.0:
        nx, ny = -nx, -ny
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    proj = [(cx - p0[0]) * nx + (cy - p0[1]) * ny for (cx, cy) in corners]
    d_min, d_max = min(proj), max(proj)
    delta = max(1e-9, float(lane_spacing))

    def sort_left_first(A, B, eps=1e-9):
        if A[0] < B[0] - eps:
            return (A, B)
        if A[0] > B[0] + eps:
            return (B, A)
        return (A, B) if A[1] <= B[1] else (B, A)

    def clip_line(d):
        Px = p0[0] + d * nx
        Py = p0[1] + d * ny
        if abs(vx) < eps:
            if Px < xmin - eps or Px > xmax + eps:
                return None
            tx_min, tx_max = -math.inf, math.inf
        else:
            t1 = (xmin - Px) / vx
            t2 = (xmax - Px) / vx
            tx_min, tx_max = (min(t1, t2), max(t1, t2))
        if abs(vy) < eps:
            if Py < ymin - eps or Py > ymax + eps:
                return None
            ty_min, ty_max = -math.inf, math.inf
        else:
            t3 = (ymin - Py) / vy
            t4 = (ymax - Py) / vy
            ty_min, ty_max = (min(t3, t4), max(t3, t4))
        t_enter = max(tx_min, ty_min)
        t_exit = min(tx_max, ty_max)
        if not (t_enter < t_exit):
            return None
        A = (Px + t_enter * vx, Py + t_enter * vy)
        B = (Px + t_exit * vx, Py + t_exit * vy)
        if (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 < (10 * eps) ** 2:
            return None
        A, B = sort_left_first(A, B)
        return (A, B)

    def canonical(seg):
        A, B = seg
        a = (round(A[0], 9), round(A[1], 9))
        b = (round(B[0], 9), round(B[1], 9))
        return a, b

    segments = {}
    k = 0
    while True:
        d_list = [0.0] if k == 0 else [k * delta, -k * delta]
        any_added = False
        for d in d_list:
            if d < d_min - eps or d > d_max + eps:
                continue
            seg = clip_line(d)
            if seg is None:
                continue
            key = canonical(seg)
            if key in segments:
                continue
            segments[key] = seg
            any_added = True
        if not any_added and (k * delta > max(abs(d_min), abs(d_max)) + 5 * delta):
            break
        k += 1
        if k > 10000:
            break

    if not segments:
        return [p0]

    def seg_d(seg):
        A, B = seg
        cx, cy = (0.5 * (A[0] + B[0]), 0.5 * (A[1] + B[1]))
        return (cx - p0[0]) * nx + (cy - p0[1]) * ny

    segs = list(segments.values())
    segs.sort(key=seg_d)

    def dist2(P, Q):
        return (P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2

    waypoints = [p0]
    first_idx = min(range(len(segs)), key=lambda i: dist2(p0, segs[i][0]))
    A0, B0 = segs[first_idx]
    waypoints.extend([A0, B0])
    cur = B0
    for idx, seg in enumerate(segs):
        if idx == first_idx:
            continue
        A, B = seg
        if dist2(cur, A) <= dist2(cur, B):
            waypoints.extend([A, B])
            cur = B
        else:
            waypoints.extend([B, A])
            cur = A
    return waypoints


def los_pure_pursuit(pos, wp_i, wp_j, Ld):
    x, y = pos
    x1, y1 = wp_i
    x2, y2 = wp_j
    vx, vy = (x2 - x1), (y2 - y1)
    seg_len2 = vx * vx + vy * vy
    if seg_len2 < 1e-9:
        return 0.0, (x2, y2), 0.0
    t = ((x - x1) * vx + (y - y1) * vy) / seg_len2
    t_clamp = max(0.0, min(1.0, t))
    xs = x1 + t_clamp * vx
    ys = y1 + t_clamp * vy
    seg_len = math.sqrt(seg_len2)
    s = min(t_clamp * seg_len + Ld, seg_len)
    xt = x1 + (s / seg_len) * vx
    yt = y1 + (s / seg_len) * vy
    psi_ref = math.atan2(yt - y, xt - x)
    ex, ey = x - xs, y - ys
    nx, ny = -vy / seg_len, vx / seg_len
    e_ct = ex * nx + ey * ny
    return psi_ref, (xt, yt), e_ct


def limit_speed_by_turn(epsi, d_target, yaw_sys: Kurs, a_lat_max=0.6, v_cap=None, eps=1e-6):
    epsi_abs = abs(epsi)
    d = max(eps, d_target)
    kappa_req = 2.0 * math.sin(epsi_abs) / d
    omega_max = yaw_sys.b * yaw_sys.Umax
    v_omega = omega_max / max(eps, kappa_req)
    v_a = math.sqrt(max(0.0, a_lat_max / max(eps, kappa_req)))
    v_lim = min(v_omega, v_a)
    if v_cap is not None:
        v_lim = min(v_lim, v_cap)
    return v_lim


def send_camera_cmd(event, port="/dev/ttyUSB0", baudrate=9600, timeout=1.0):
    cmd = bytes([0xFF])
    if serial is None:
        print(f"[CAMERA] {event}: send {cmd.hex().upper()} to {port} (emu)")
        return True
    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as ser:
            ser.write(cmd)
            ser.flush()
        print(f"[CAMERA] {event}: sent {cmd.hex().upper()} to {port}")
        return True
    except Exception as e:
        print(f"[CAMERA][ERROR] {event}: {e}")
        return False


def draw_cov_rectangle(ax, S, E, width, color="#6EC1FF", alpha=0.15):
    dx, dy = E[0] - S[0], E[1] - S[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    w2 = width / 2.0
    p1 = (S[0] + nx * w2, S[1] + ny * w2)
    p2 = (E[0] + nx * w2, E[1] + ny * w2)
    p3 = (E[0] - nx * w2, E[1] - ny * w2)
    p4 = (S[0] - nx * w2, S[1] - ny * w2)
    ax.add_patch(
        mpatches.Polygon([p1, p2, p3, p4], closed=True, facecolor=color, edgecolor="none", alpha=alpha)
    )


# =========================
# Результаты марш-сценария для GUI
# =========================
@dataclass
class SimResult:
    xs: List[float]
    ys: List[float]
    wps: List[Tuple[float, float]]
    rect: Tuple[float, float, float, float]
    acq_spans: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    camera_events: List[Tuple[str, Tuple[float, float], float]]
    t_hist: List[float]
    psi_deg: List[float]
    u_ground: List[float]
    e_ct_hist: List[float]
    meta: Dict[str, Any]


def run_simulation(p: Dict[str, Any]) -> SimResult:
    # Геометрия, SSS
    rect = (p["rect_x"], p["rect_y"], p["rect_w"], p["rect_h"])
    heading_deg = p["heading_deg"]

    h_alt = p["h_alt"]
    R_slant = p["R_slant"]
    beta_deg = p["beta_deg"]
    cross_overlap = p["cross_overlap"]
    along_overlap = p["along_overlap"]

    # Скорости/ограничения
    U_cruise = p["U_cruise"]
    U_acq = p["U_acq"]
    lookahead = p["lookahead"]
    switch_R = p["switch_R"]
    a_lat_max = p["a_lat_max"]
    dt = p["dt"]

    # Ошибки
    use_5pct_error_psi = p["use_5pct_error_psi"]
    use_5pct_error_r = p["use_5pct_error_r"]
    use_5pct_error_u = p["use_5pct_error_u"]
    meas_scale = p["meas_scale"]

    # Параметры yaw (курс)
    yaw_kwargs = dict(
        Jy=p["Jy"], L55=p["L55"],
        Cwy1=p["Cwy1"], Cwy2=p["Cwy2"],
        Tdv=p["Tdv"], Kdv=p["Kdv_yaw"], b=p["b"],
        Umax=p["Umax_yaw"],
        K1=p["K1"], K2=p["K2"],
        Mdist=p["Mdist"],
        Fcur_yaw_mag=p["Fcur_yaw_mag"],
        Fcur_yaw_dir_deg=p["Fcur_yaw_dir_deg"],
        l_yaw_arm=p["l_yaw_arm"],
        use_5pct_error_psi=use_5pct_error_psi,
        use_5pct_error_r=use_5pct_error_r,
        meas_scale=meas_scale,
        dt=dt,
    )

    # Параметры surge (марш)
    surge_kwargs = dict(
        m=p["m"], lam11=p["lam11"],
        Cxu1=p["Cxu1"], Cxu2=p["Cxu2"],
        Tdx=p["Tdx"], Kdv=p["Kdv_surge"],
        Umax=p["Umax_surge"],
        Kp=p["Kp"], Kd=p["Kd"], at=p["at"],
        wn_ref=p["wn_ref"], zeta_ref=p["zeta_ref"],
        Ucur_mag=p["Ucur_mag"], Ucur_dir_deg=p["Ucur_dir_deg"],
        use_5pct_error_u=use_5pct_error_u,
        meas_scale=meas_scale,
        dt=dt,
    )

    # Доминирующее время регулирования (оценка)
    def estimate_ts_yaw(sys: Kurs, psi_step_deg=1.0, tol=0.02, Tmax=8.0):
        tloc = 0.0
        target = psi_step_deg
        reached = False
        ts = Tmax
        while tloc < Tmax:
            tloc += sys.dt
            y = sys.step(target)
            err = abs(target - y)
            if not reached and err <= tol * abs(target):
                reached = True
                ts = tloc
            if reached and err > tol * abs(target):
                reached = False
                ts = Tmax
        return ts

    def estimate_ts_surge(sys: MarshevPD, u_step=1.0, tol=0.02, Tmax=12.0):
        tloc = 0.0
        target = u_step  # ground speed
        reached = False
        ts = Tmax
        psi0 = 0.0
        while tloc < Tmax:
            tloc += sys.dt
            y_rel = sys.step(target, psi=psi0)
            y_ground = max(0.0, y_rel + sys.Ucur_mag * math.cos(sys.Ucur_dir - psi0))
            err = abs(target - y_ground)
            if not reached and err <= tol * abs(target):
                reached = True
                ts = tloc
            if reached and err > tol * abs(target):
                reached = False
                ts = Tmax
        return ts

    ts_yaw = estimate_ts_yaw(Kurs(**yaw_kwargs), psi_step_deg=1.0, Tmax=8.0)
    ts_surge = estimate_ts_surge(MarshevPD(**surge_kwargs), u_step=1.0, Tmax=12.0)
    ts_dom = max(ts_yaw, ts_surge)
    turn_buffer = 1.2 * U_cruise * ts_dom

    # Разнос полос
    W_use, lane_spacing = sss_swath_and_spacing(R_slant, h_alt, beta_deg, cross_overlap)
    s_ping = (1.0 - along_overlap) * W_use
    dt_ping = s_ping / max(1e-6, U_acq)

    # Маршрут
    wps = build_lawnmower_in_rect_from_corner(rect, lane_spacing, heading_deg, margin=0.0)

    # Участки съёмки (прямые минус буфер разворота)
    def build_acq_spans_on_straights(wps_list, buf):
        spans_ = []
        for i in range(1, len(wps_list) - 1, 2):
            A = wps_list[i]; B = wps_list[i + 1]
            L = math.hypot(B[0] - A[0], B[1] - A[1])
            if L <= 2 * buf:
                continue
            dx = (B[0] - A[0]) / L; dy = (B[1] - A[1]) / L
            S = (A[0] + dx * buf, A[1] + dy * buf)
            E = (B[0] - dx * buf, B[1] - dy * buf)
            spans_.append((S, E))
        return spans_

    acq_spans = build_acq_spans_on_straights(wps, turn_buffer)

    # Подсчёты длин/времени
    total_dist = sum(
        math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1]) for i in range(len(wps) - 1)
    )
    span_len_total = sum(math.hypot(E[0] - S[0], E[1] - S[1]) for (S, E) in acq_spans)
    span_len_total = max(0.0, min(span_len_total, total_dist))
    T_nom = (span_len_total / max(1e-6, U_acq)) + ((total_dist - span_len_total) / max(1e-6, U_cruise))
    T_end = 1.25 * T_nom

    # Объекты систем
    yaw = Kurs(**yaw_kwargs)
    surge = MarshevPD(**surge_kwargs)

    # Начальная поза
    x, y = wps[0]
    if len(wps) >= 2:
        dx0, dy0 = wps[1][0] - wps[0][0], wps[1][1] - wps[0][1]
        yaw.psi = math.atan2(dy0, dx0)

    # Истории
    xs, ys = [x], [y]
    psis_deg = [yaw.psi * RAD2DEG]
    us_ground = []
    e_ct_hist = [0.0]
    t_hist = [0.0]

    t = 0.0
    i_seg = 0
    max_steps = int(2.0 * T_end / dt) + 10000
    steps = 0

    Ucx = surge.Ucur_mag * math.cos(surge.Ucur_dir)
    Ucy = surge.Ucur_mag * math.sin(surge.Ucur_dir)

    while t < T_end and i_seg < len(wps) - 1 and steps < max_steps:
        steps += 1
        wp_i = wps[i_seg]; wp_j = wps[i_seg + 1]
        psi_ref, target_pt, e_ct = los_pure_pursuit((x, y), wp_i, wp_j, lookahead)
        xt, yt = target_pt
        d_target = math.hypot(xt - x, yt - y)

        dist_to_end = math.hypot(wp_j[0] - x, wp_j[1] - y)
        u_ref_ground = U_acq if (i_seg % 2 == 1) else U_cruise * max(0.3, min(1.0, dist_to_end / (2.0 * lookahead)))

        epsi = wrap_rad(psi_ref - yaw.psi)
        u_ref_turn_cap = limit_speed_by_turn(epsi, d_target, yaw_sys=yaw, a_lat_max=a_lat_max, v_cap=u_ref_ground)
        u_ref_ground = max(0.0, min(u_ref_ground, u_ref_turn_cap))

        # Шаг регуляторов
        yaw.step(psi_ref * RAD2DEG)
        surge.step(u_ref_ground, psi=yaw.psi)

        psi = yaw.psi
        Vx = surge.u * math.cos(psi) + Ucx
        Vy = surge.u * math.sin(psi) + Ucy
        v_ground = math.hypot(Vx, Vy)

        x += Vx * dt
        y += Vy * dt

        if d_target < 2.0:
            i_seg += 1

        t += dt
        xs.append(x); ys.append(y)
        psis_deg.append(psi * RAD2DEG)
        us_ground.append(v_ground)
        e_ct_hist.append(e_ct)
        t_hist.append(t)

    meta = dict(
        W_use=W_use,
        lane_spacing=lane_spacing,
        s_ping=s_ping,
        dt_ping=dt_ping,
        turn_buffer=turn_buffer,
        U_acq=U_acq,
        U_cruise=U_cruise,
        ts_yaw=ts_yaw,
        ts_surge=ts_surge,
    )

    return SimResult(
        xs=xs, ys=ys, wps=wps, rect=rect, acq_spans=acq_spans, camera_events=[],
        t_hist=t_hist, psi_deg=psis_deg, u_ground=us_ground, e_ct_hist=e_ct_hist, meta=meta
    )


# =========================
# Переходные — расчёт + метрики (Mp, Ts, 5%-трубка)
# =========================
@dataclass
class StepResponses:
    t_yaw: List[float]
    yaw_deg: List[float]
    yaw_ref: List[float]
    yaw_band: Tuple[float, float]
    yaw_Mp_pct: float
    yaw_Ts: float | None

    t_surge: List[float]
    u_ground: List[float]
    u_ref_ground: List[float]
    surge_band: Tuple[float, float]
    surge_Mp_pct: float
    surge_Ts: float | None


def _step_metrics(t: List[float], y: List[float], ref: List[float], tol=0.05) -> Tuple[Tuple[float, float], float, float | None]:
    """
    Возвращает:
      (y_low, y_high), Mp_pct, Ts
    где Ts — первое время, после которого |y-yr| <= tol*|yr| навсегда.
    """
    if not t:
        return (0.0, 0.0), 0.0, None
    yr = ref[-1]
    eps = 1e-9
    A = max(abs(yr), eps)  # для нормировки
    y_low = yr - tol * A
    y_high = yr + tol * A

    # Перерегулирование в сторону шага (знак учитываем)
    sgn = 1.0 if yr >= 0 else -1.0
    peak = max((yi - yr) * sgn for yi in y)
    Mp_pct = max(0.0, 100.0 * peak / A)

    # Время установления (5%): первое t_i, что для всех j>=i выполнено |y-yr|<=0.05*|yr|
    Ts = None
    for i in range(len(t)):
        if all(abs(y[j] - yr) <= tol * A for j in range(i, len(t))):
            Ts = t[i]
            break
    return (y_low, y_high), Mp_pct, Ts


def compute_step_responses(p: Dict[str, Any]) -> StepResponses:
    dt = p["dt"]

    # --- Yaw step ---
    yaw_kwargs = dict(
        Jy=p["Jy"], L55=p["L55"], Cwy1=p["Cwy1"], Cwy2=p["Cwy2"],
        Tdv=p["Tdv"], Kdv=p["Kdv_yaw"], b=p["b"], Umax=p["Umax_yaw"],
        K1=p["K1"], K2=p["K2"],
        Mdist=p["Mdist"],
        Fcur_yaw_mag=p["Fcur_yaw_mag"], Fcur_yaw_dir_deg=p["Fcur_yaw_dir_deg"], l_yaw_arm=p["l_yaw_arm"],
        use_5pct_error_psi=p["use_5pct_error_psi"], use_5pct_error_r=p["use_5pct_error_r"],
        meas_scale=p["meas_scale"], dt=dt
    )
    sys_yaw = Kurs(**yaw_kwargs)
    T_yaw = max(dt, float(p["yaw_step_T"]))
    amp_yaw = float(p["yaw_step_amp_deg"])

    t_yaw = [0.0]; yaw_deg = [sys_yaw.psi * RAD2DEG]; yaw_ref = [amp_yaw]
    tcur = 0.0
    while tcur < T_yaw:
        tcur += dt
        y = sys_yaw.step(amp_yaw)
        t_yaw.append(tcur); yaw_deg.append(y); yaw_ref.append(amp_yaw)

    yaw_band, yaw_Mp, yaw_Ts = _step_metrics(t_yaw, yaw_deg, yaw_ref, tol=0.05)

    # --- Surge (ground speed) step ---
    surge_kwargs = dict(
        m=p["m"], lam11=p["lam11"], Cxu1=p["Cxu1"], Cxu2=p["Cxu2"],
        Tdx=p["Tdx"], Kdv=p["Kdv_surge"], Umax=p["Umax_surge"],
        Kp=p["Kp"], Kd=p["Kd"], at=p["at"],
        wn_ref=p["wn_ref"], zeta_ref=p["zeta_ref"],
        Ucur_mag=p["Ucur_mag"], Ucur_dir_deg=p["Ucur_dir_deg"],
        use_5pct_error_u=p["use_5pct_error_u"], meas_scale=p["meas_scale"], dt=dt
    )
    sys_surge = MarshevPD(**surge_kwargs)
    T_surge = max(dt, float(p["surge_step_T"]))
    amp_u = float(p["surge_step_amp_ground"])
    psi_hold = 0.0  # фиксируем курс

    t_surge = [0.0]
    ug0 = max(0.0, sys_surge.u + sys_surge.Ucur_mag * math.cos(sys_surge.Ucur_dir - psi_hold))
    u_ground = [ug0]; u_ref_ground = [amp_u]
    tcur = 0.0
    while tcur < T_surge:
        tcur += dt
        sys_surge.step(amp_u, psi=psi_hold)  # опорная — наземная скорость
        ug = max(0.0, sys_surge.u + sys_surge.Ucur_mag * math.cos(sys_surge.Ucur_dir - psi_hold))
        t_surge.append(tcur); u_ground.append(ug); u_ref_ground.append(amp_u)

    surge_band, surge_Mp, surge_Ts = _step_metrics(t_surge, u_ground, u_ref_ground, tol=0.05)

    return StepResponses(
        t_yaw=t_yaw, yaw_deg=yaw_deg, yaw_ref=yaw_ref, yaw_band=yaw_band, yaw_Mp_pct=yaw_Mp, yaw_Ts=yaw_Ts,
        t_surge=t_surge, u_ground=u_ground, u_ref_ground=u_ref_ground, surge_band=surge_band, surge_Mp_pct=surge_Mp, surge_Ts=surge_Ts
    )


# =========================
# GUI: Таблица параметров + 6 графиков
# =========================
@dataclass
class ParamRow:
    category: str
    key: str
    label: str
    value: Any
    ptype: str  # 'float'|'int'|'bool'
    unit: str
    desc: str


def default_param_rows() -> List[ParamRow]:
    return [
        # Геометрия
        ParamRow("Геометрия", "rect_x", "rect.x", 0.0, "float", "м", "Начальная x сектора"),
        ParamRow("Геометрия", "rect_y", "rect.y", 0.0, "float", "м", "Начальная y сектора"),
        ParamRow("Геометрия", "rect_w", "rect.W", 200.0, "float", "м", "Ширина сектора"),
        ParamRow("Геометрия", "rect_h", "rect.H", 150.0, "float", "м", "Высота сектора"),
        ParamRow("Геометрия", "heading_deg", "heading", 30.0, "float", "°", "Азимут «галсов»"),

        # SSS
        ParamRow("SSS", "h_alt", "h_alt", 1.5, "float", "м", "Высота над дном"),
        ParamRow("SSS", "R_slant", "R_slant", 15.0, "float", "м", "Наклонная дальность"),
        ParamRow("SSS", "beta_deg", "beta", 4.0, "float", "°", "Угол «тени» по бортам"),
        ParamRow("SSS", "cross_overlap", "cross_overlap", 0.20, "float", "дол.", "Перекрытие между галсами"),
        ParamRow("SSS", "along_overlap", "along_overlap", 0.30, "float", "дол.", "Перекрытие по ходу"),

        # Скорости/ограничения
        ParamRow("Движение", "U_cruise", "U_cruise", 1.2, "float", "м/с", "Крейсерская наземная"),
        ParamRow("Движение", "U_acq", "U_acq", 0.7, "float", "м/с", "Рабочая наземная"),
        ParamRow("Движение", "lookahead", "lookahead", 6.0, "float", "м", "Длина упреждения LOS"),
        ParamRow("Движение", "switch_R", "switch_R", 2.0, "float", "м", "Радиус переключения сегмента"),
        ParamRow("Движение", "a_lat_max", "a_lat_max", 0.6, "float", "м/с²", "Макс. поперечное ускорение"),
        ParamRow("Движение", "dt", "dt", 0.02, "float", "с", "Шаг интегрирования"),

        # Курс (Yaw)
        ParamRow("Yaw", "Jy", "Jy", 90.0, "float", "кг·м²", "Момент инерции по курсу"),
        ParamRow("Yaw", "L55", "L55", 87.0, "float", "кг·м²", "Добавочный момент инерции"),
        ParamRow("Yaw", "Cwy1", "Cwy1", 75.0, "float", "-", "Квадратная часть сопротивления"),
        ParamRow("Yaw", "Cwy2", "Cwy2", 7.4, "float", "-", "Линейная часть сопротивления"),
        ParamRow("Yaw", "Tdv", "Tdv", 0.15, "float", "с", "Постоянная времени привода"),
        ParamRow("Yaw", "Kdv_yaw", "Kdv", 20.0, "float", "-", "Коэф. привода (yaw)"),
        ParamRow("Yaw", "b", "b", 0.3, "float", "-", "Коэф. привода в омегу"),
        ParamRow("Yaw", "Umax_yaw", "Umax", 10.0, "float", "В", "Насыщение по управляющему"),
        ParamRow("Yaw", "K1", "K1", 0.6772091636, "float", "В/°", "P-усиление"),
        ParamRow("Yaw", "K2", "K2", 0.6157978918, "float", "В/(°/с)", "D-усиление"),
        ParamRow("Yaw", "Mdist", "Mdist", 0.0, "float", "Н·м", "Пост. возм. момент (опц.)"),
        ParamRow("Yaw.Течение", "Fcur_yaw_mag", "F_cur (yaw)", 0.0, "float", "Н", "Сила течения(для момента)"),
        ParamRow("Yaw.Течение", "Fcur_yaw_dir_deg", "θ_cur (yaw)", 0.0, "float", "°", "Азимут течения"),
        ParamRow("Yaw.Течение", "l_yaw_arm", "l_arm", 1.0, "float", "м", "Плечо приложения силы"),

        # Surge (марш)
        ParamRow("Surge", "m", "m", 90.0, "float", "кг", "Масса"),
        ParamRow("Surge", "lam11", "λ11", 78.0, "float", "кг", "Добавочная масса"),
        ParamRow("Surge", "Cxu1", "Cxu1", 37.0, "float", "-", "Квадратная часть сопротивления"),
        ParamRow("Surge", "Cxu2", "Cxu2", 3.7, "float", "-", "Линейная часть сопротивления"),
        ParamRow("Surge", "Tdx", "Tdx", 0.15, "float", "с", "Постоянная времени привода"),
        ParamRow("Surge", "Kdv_surge", "Kdv", 20.0, "float", "-", "Коэф. привода (surge)"),
        ParamRow("Surge", "Umax_surge", "Umax", 10.0, "float", "В", "Насыщение по управляющему"),
        ParamRow("Surge", "Kp", "Kp", 1.98712, "float", "-", "P-усиление марша"),
        ParamRow("Surge", "Kd", "Kd", 1.20176, "float", "1/с", "D-усиление марша"),
        ParamRow("Surge", "at", "a_t", 12.0, "float", "1/с", "Скорость привода"),
        ParamRow("Surge", "wn_ref", "ω_n", 1.2, "float", "рад/с", "Собств. частота фильтра ссылки"),
        ParamRow("Surge", "zeta_ref", "ζ", 0.7, "float", "-", "Демпфирование фильтра ссылки"),
        ParamRow("Surge.Течение", "Ucur_mag", "U_cur", 0.0, "float", "м/с", "Скорость течения"),
        ParamRow("Surge.Течение", "Ucur_dir_deg", "θ_cur", 0.0, "float", "°", "Азимут течения"),

        # Ошибки измерений
        ParamRow("Ошибки", "use_5pct_error_psi", "err ψ +5%", False, "bool", "-", "Ошибка измерения курса"),
        ParamRow("Ошибки", "use_5pct_error_r", "err r +5%", False, "bool", "-", "Ошибка измерения ω"),
        ParamRow("Ошибки", "use_5pct_error_u", "err u +5%", False, "bool", "-", "Ошибка измерения скорости"),
        ParamRow("Ошибки", "meas_scale", "meas_scale", 1.05, "float", "-", "Мультипликативный коэффициент"),

        # Переходные процессы
        ParamRow("Переходные", "yaw_step_amp_deg", "Δψ_step", 1.0, "float", "°", "Амплитуда ступеньки по курсу"),
        ParamRow("Переходные", "yaw_step_T", "T_step(yaw)", 8.0, "float", "с", "Длительность переходной по курсу"),
        ParamRow("Переходные", "surge_step_amp_ground", "Δu_ground_step", 1.0, "float", "м/с", "Амплитуда ступеньки по наземной скорости"),
        ParamRow("Переходные", "surge_step_T", "T_step(surge)", 12.0, "float", "с", "Длительность переходной по маршу"),
    ]


class ParamTable(QtWidgets.QTableWidget):
    COLS = ["Категория", "Параметр", "Значение", "Ед.", "Описание"]

    def __init__(self, rows: List[ParamRow], parent=None):
        super().__init__(parent)
        self.setColumnCount(len(self.COLS))
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setShowGrid(True)
        self._rows = rows
        self._key_index: Dict[int, str] = {}
        self.populate(rows)
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def populate(self, rows: List[ParamRow]):
        self.setRowCount(len(rows))
        self._key_index.clear()
        for i, r in enumerate(rows):
            cat_item = QtWidgets.QTableWidgetItem(r.category); cat_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            key_item = QtWidgets.QTableWidgetItem(r.label); key_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)

            # Значение
            if r.ptype == "bool":
                val_item = QtWidgets.QTableWidgetItem()
                val_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                val_item.setCheckState(QtCore.Qt.CheckState.Checked if bool(r.value) else QtCore.Qt.CheckState.Unchecked)
                val_item.setText("True" if bool(r.value) else "False")
            else:
                val_item = QtWidgets.QTableWidgetItem(str(r.value))
                val_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                val_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

            unit_item = QtWidgets.QTableWidgetItem(r.unit); unit_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            desc_item = QtWidgets.QTableWidgetItem(r.desc); desc_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)

            self.setItem(i, 0, cat_item); self.setItem(i, 1, key_item)
            self.setItem(i, 2, val_item); self.setItem(i, 3, unit_item); self.setItem(i, 4, desc_item)

            for c in range(self.columnCount()):
                self.item(i, c).setData(QtCore.Qt.ItemDataRole.UserRole, r.key)
            self._key_index[i] = r.key

        self.setSortingEnabled(True)
        self.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)

    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for r in range(self.rowCount()):
            key = self.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            val_item = self.item(r, 2)
            spec = next((x for x in self._rows if x.key == key), None)
            if spec is None:
                continue
            if spec.ptype == "bool":
                val = val_item.checkState() == QtCore.Qt.CheckState.Checked
            else:
                txt = (val_item.text() or "").strip().replace(",", ".")
                try:
                    if spec.ptype == "int":
                        val = int(float(txt))
                    else:
                        val = float(txt)
                except Exception:
                    val_item.setBackground(QtGui.QBrush(QtGui.QColor("#ffcccc")))
                    raise ValueError(f"Неверное число в параметре: {spec.label}")
            params[key] = val
        return params

    def reset_defaults(self):
        self.populate(default_param_rows())

    def load_from_dict(self, d: Dict[str, Any]):
        for r in range(self.rowCount()):
            key = self.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            if key not in d:
                continue
            spec = next((x for x in self._rows if x.key == key), None)
            if spec is None:
                continue
            val_item = self.item(r, 2)
            if spec.ptype == "bool":
                checked = QtCore.Qt.CheckState.Checked if bool(d[key]) else QtCore.Qt.CheckState.Unchecked
                val_item.setCheckState(checked); val_item.setText("True" if bool(d[key]) else "False")
            else:
                val_item.setText(str(d[key]))


class PlotTab(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.grid(True)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUV Survey — табличные параметры и симуляция")
        self.resize(1300, 850)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        self.param_table = ParamTable(default_param_rows())
        splitter.addWidget(self.param_table)

        self.tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 6)

        self.tab_plan = PlotTab("План (траектория и покрытие)")
        self.tab_yaw = PlotTab("Курс ψ(t)")
        self.tab_speed = PlotTab("Скорость (ground) v(t)")
        self.tab_ect = PlotTab("Поперечная ошибка e_ct(t)")
        self.tab_step_yaw = PlotTab("Переходная по курсу")
        self.tab_step_surge = PlotTab("Переходная по маршу")

        self.tabs.addTab(self.tab_plan, "План")
        self.tabs.addTab(self.tab_yaw, "Курс")
        self.tabs.addTab(self.tab_speed, "Скорость")
        self.tabs.addTab(self.tab_ect, "e_ct")
        self.tabs.addTab(self.tab_step_yaw, "Переходная по курсу")
        self.tabs.addTab(self.tab_step_surge, "Переходная по маршу")

        # Панель кнопок
        btn_run = QtWidgets.QPushButton("Запустить")
        btn_reset = QtWidgets.QPushButton("Сброс")
        btn_save = QtWidgets.QPushButton("Сохранить пресет…")
        btn_load = QtWidgets.QPushButton("Загрузить пресет…")

        btn_run.clicked.connect(self.on_run)
        btn_reset.clicked.connect(self.on_reset)
        btn_save.clicked.connect(self.on_save)
        btn_load.clicked.connect(self.on_load)

        top_bar = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.addWidget(btn_run)
        top_layout.addWidget(btn_reset)
        top_layout.addStretch(1)
        top_layout.addWidget(btn_save)
        top_layout.addWidget(btn_load)
        self.addToolBarBreak()
        tool = QtWidgets.QToolBar("Управление")
        tool.addWidget(top_bar)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tool)

        self.statusBar().showMessage("Готово")

    def on_reset(self):
        self.param_table.reset_defaults()
        self.statusBar().showMessage("Параметры сброшены к значениям по умолчанию.")

    def on_save(self):
        try:
            params = self.param_table.to_params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка параметров", str(e)); return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить пресет", filter="JSON (*.json)")
        if fn:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f"Пресет сохранён: {fn}")

    def on_load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Загрузить пресет", filter="JSON (*.json)")
        if fn:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.param_table.load_from_dict(data)
            self.statusBar().showMessage(f"Пресет загружен: {fn}")

    def on_run(self):
        try:
            params = self.param_table.to_params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка параметров", str(e)); return

        # --- Симуляция по маршруту ---
        try:
            res = run_simulation(params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка симуляции", str(e)); return

        # --- Переходные ---
        try:
            steps = compute_step_responses(params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка переходных", str(e)); return

        # Очистка осей
        for tab in (self.tab_plan, self.tab_yaw, self.tab_speed, self.tab_ect, self.tab_step_yaw, self.tab_step_surge):
            tab.fig.clf(); tab.ax = tab.fig.add_subplot(111); tab.ax.grid(True)

        # ---- План
        ax = self.tab_plan.ax
        xmin, ymin, W, H = res.rect
        rx = [xmin, xmin + W, xmin + W, xmin, xmin]
        ry = [ymin, ymin, ymin + H, ymin + H, ymin]
        ax.plot(rx, ry, linestyle="--", color="k", linewidth=1.4, label="Сектор")
        if len(res.wps) >= 2:
            wx, wy = zip(*res.wps)
            ax.plot(wx, wy, ":", color="gray", alpha=0.7, linewidth=1.2, label="Маршрут (галсы)")
        ax.plot(res.xs, res.ys, label="Траектория ПА (ground)")

        drew_span_label = False
        for (S, E) in res.acq_spans:
            draw_cov_rectangle(ax, S, E, width=res.meta["W_use"], alpha=0.15)
            if not drew_span_label:
                ax.plot([S[0], E[0]], [S[1], E[1]], linewidth=3.0, alpha=0.9,
                        color="limegreen", label=f"Участок съёмки (U={res.meta['U_acq']:.1f} м/с)")
                drew_span_label = True
            else:
                ax.plot([S[0], E[0]], [S[1], E[1]], linewidth=3.0, alpha=0.9, color="limegreen")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
        ax.set_title(
            f"SSS: W≈{res.meta['W_use']:.1f} м | lane≈{res.meta['lane_spacing']:.1f} м | "
            f"U_acq={res.meta['U_acq']:.1f} м/с | buffer≈{res.meta['turn_buffer']:.1f} м | "
            f"t_s(yaw)≈{res.meta['ts_yaw']:.2f}s | t_s(surge)≈{res.meta['ts_surge']:.2f}s"
        )
        ax.legend(loc="best"); self.tab_plan.canvas.draw()

        # ---- Курс
        ax = self.tab_yaw.ax
        ax.plot(res.t_hist, res.psi_deg)
        ax.set_xlabel("t, c"); ax.set_ylabel("ψ, °"); ax.set_title("Курс ψ(t)")
        self.tab_yaw.canvas.draw()

        # ---- Скорость (ground)
        ax = self.tab_speed.ax
        t_sp = res.t_hist[1:] if len(res.t_hist) == len(res.u_ground) + 1 else res.t_hist[: len(res.u_ground)]
        ax.plot(t_sp, res.u_ground)
        ax.set_xlabel("t, c"); ax.set_ylabel("|V_ground|, м/с"); ax.set_title("Скорость (ground) v(t)")
        self.tab_speed.canvas.draw()

        # ---- e_ct
        ax = self.tab_ect.ax
        ax.plot(res.t_hist, res.e_ct_hist)
        ax.set_xlabel("t, c"); ax.set_ylabel("e_ct, м"); ax.set_title("Поперечная ошибка e_ct(t)")
        self.tab_ect.canvas.draw()

        # ---- Переходная по курсу
        ax = self.tab_step_yaw.ax
        ax.plot(steps.t_yaw, steps.yaw_deg, label="ψ(t)")
        ax.plot(steps.t_yaw, steps.yaw_ref, linestyle="--", label="ψ_ref")
        # 5%-трубка
        ax.axhline(steps.yaw_band[0], linestyle="--", linewidth=1)
        ax.axhline(steps.yaw_band[1], linestyle="--", linewidth=1)
        # Ts вертикаль + подпись
        if steps.yaw_Ts is not None:
            ax.axvline(steps.yaw_Ts, linestyle=":", linewidth=1)
        # Заголовок сокращённый
        ax.set_title("Переходная по курсу")
        ax.set_xlabel("t, c"); ax.set_ylabel("ψ, °")
        # Текст с метриками
        txt = f"Mp ≈ {steps.yaw_Mp_pct:.1f}%"
        if steps.yaw_Ts is not None:
            txt += f", Ts ≈ {steps.yaw_Ts:.2f} c"
        ax.legend(loc="best", title=txt)
        self.tab_step_yaw.canvas.draw()

        # ---- Переходная по маршу
        ax = self.tab_step_surge.ax
        ax.plot(steps.t_surge, steps.u_ground, label="|V_ground|(t)")
        ax.plot(steps.t_surge, steps.u_ref_ground, linestyle="--", label="u_ref")
        ax.axhline(steps.surge_band[0], linestyle="--", linewidth=1)
        ax.axhline(steps.surge_band[1], linestyle="--", linewidth=1)
        if steps.surge_Ts is not None:
            ax.axvline(steps.surge_Ts, linestyle=":", linewidth=1)
        ax.set_title("Переходная по маршу")
        ax.set_xlabel("t, c"); ax.set_ylabel("м/с")
        txt = f"Mp ≈ {steps.surge_Mp_pct:.1f}%"
        if steps.surge_Ts is not None:
            txt += f", Ts ≈ {steps.surge_Ts:.2f} c"
        ax.legend(loc="best", title=txt)
        self.tab_step_surge.canvas.draw()

        # Статус
        self.statusBar().showMessage(
            f"L≈{(sum(math.hypot(res.xs[i+1]-res.xs[i], res.ys[i+1]-res.ys[i]) for i in range(len(res.xs)-1))):.1f} м; "
            f"W≈{res.meta['W_use']:.1f} м; lane≈{res.meta['lane_spacing']:.1f} м; "
            f"s_ping≈{res.meta['s_ping']:.2f} м; dt_ping≈{res.meta['dt_ping']:.2f} c"
        )


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(248, 248, 248))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(255, 255, 255))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(242, 242, 242))
    app.setPalette(pal)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
