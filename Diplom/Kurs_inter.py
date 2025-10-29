# -*- coding: utf-8 -*-
"""
AUV Survey GUI — двухдвижительная схема БФС ДРК (полная версия):
- Галсы + прямоугольники покрытия SSS
- "Рабочие" отрезки съёмки, события SSS_START/SSS_END (опциональная COM-отправка)
- Контуры: курс (PD) + марш (PD с фильтром ссылки)
- Микшер: U_L = mix_surge*U_surge - mix_yaw*U_yaw; U_R = ... + ...
  Сатурация с сохранением пропорций для обоих каналов
- Автонастройка под Ts≈2 c (опционально)
- Переходные:
    * ψ(t) (курс)
    * u(тело) (скорость в корпусе)
    * x(t) (по координате X с наружным ПД → u_ref)
- График моментов: M_thr(t), M_net(t)

Новое (защиты начала/конца съёмки):
- Прямоугольник маршрута увеличивается на +10% ОТ ЦЕНТРА (исходный рисуется пунктиром)
- Рабочие участки съёмки: S = A + (turn_buffer + guard_after_turn_m); E = B - (turn_buffer + guard_before_turn_m)
- События SSS_START/SSS_END срабатывают только на прямой с допуском по курсу
- START — только после выхода из поворота и выравнивания на прямой; END — до входа в поворот
Зависимости: PyQt6, matplotlib
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
import matplotlib.patches as mpatches

# ---- PyQt6 ----
from PyQt6 import QtWidgets, QtGui, QtCore

try:
    import serial  # type: ignore
except Exception:
    serial = None

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


def wrap_rad(a: float) -> float:
    while a <= -math.pi: a += 2 * math.pi
    while a >   math.pi: a -= 2 * math.pi
    return a


# =========================
# УТИЛИТЫ SSS/маршрута
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
    xmin += margin; ymin += margin
    xmax = xmin + (W - 2 * margin)
    ymax = ymin + (H - 2 * margin)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Слишком большой margin.")
    p0 = (xmin, ymin)
    th = math.radians(heading_deg)
    vx, vy = math.cos(th), math.sin(th)
    nx, ny = -vy, vx
    if nx < 0.0: nx, ny = -nx, -ny
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    proj = [(cx - p0[0]) * nx + (cy - p0[1]) * ny for (cx, cy) in corners]
    d_min, d_max = min(proj), max(proj)
    delta = max(1e-9, float(lane_spacing))

    def sort_left_first(A, B, eps=1e-9):
        if A[0] < B[0] - eps: return (A, B)
        if A[0] > B[0] + eps: return (B, A)
        return (A, B) if A[1] <= B[1] else (B, A)

    def clip_line(d):
        Px = p0[0] + d * nx; Py = p0[1] + d * ny
        if abs(vx) < eps:
            if Px < xmin - eps or Px > xmax + eps: return None
            tx_min, tx_max = -math.inf, math.inf
        else:
            t1 = (xmin - Px) / vx; t2 = (xmax - Px) / vx
            tx_min, tx_max = (min(t1, t2), max(t1, t2))
        if abs(vy) < eps:
            if Py < ymin - eps or Py > ymax + eps: return None
            ty_min, ty_max = -math.inf, math.inf
        else:
            t3 = (ymin - Py) / vy; t4 = (ymax - Py) / vy
            ty_min, ty_max = (min(t3, t4), max(t3, t4))
        t_enter = max(tx_min, ty_min); t_exit = min(tx_max, ty_max)
        if not (t_enter < t_exit): return None
        A = (Px + t_enter * vx, Py + t_enter * vy)
        B = (Px + t_exit * vx, Py + t_exit * vy)
        if (A[0]-B[0])**2 + (A[1]-B[1])**2 < (10*eps)**2: return None
        A, B = sort_left_first(A, B); return (A, B)

    def canonical(seg):
        A, B = seg
        a = (round(A[0], 9), round(A[1], 9))
        b = (round(B[0], 9), round(B[1], 9))
        return a, b

    segments = {}; k = 0
    while True:
        d_list = [0.0] if k == 0 else [k*delta, -k*delta]
        any_added = False
        for d in d_list:
            if d < d_min - eps or d > d_max + eps: continue
            seg = clip_line(d)
            if seg is None: continue
            key = canonical(seg)
            if key in segments: continue
            segments[key] = seg; any_added = True
        if not any_added and (k*delta > max(abs(d_min), abs(d_max)) + 5*delta): break
        k += 1
        if k > 10000: break

    if not segments: return [p0]

    def seg_d(seg):
        A, B = seg
        cx, cy = (0.5*(A[0]+B[0]), 0.5*(A[1]+B[1]))
        return (cx - p0[0])*nx + (cy - p0[1])*ny

    segs = list(segments.values()); segs.sort(key=seg_d)

    def dist2(P, Q): return (P[0]-Q[0])**2 + (P[1]-Q[1])**2
    waypoints = [p0]
    first_idx = min(range(len(segs)), key=lambda i: dist2(p0, segs[i][0]))
    A0, B0 = segs[first_idx]; waypoints.extend([A0, B0]); cur = B0
    for idx, seg in enumerate(segs):
        if idx == first_idx: continue
        A, B = seg
        if dist2(cur, A) <= dist2(cur, B):
            waypoints.extend([A, B]); cur = B
        else:
            waypoints.extend([B, A]); cur = A
    return waypoints


def los_pure_pursuit(pos, wp_i, wp_j, Ld):
    x, y = pos
    x1, y1 = wp_i; x2, y2 = wp_j
    vx, vy = (x2 - x1), (y2 - y1)
    seg_len2 = vx*vx + vy*vy
    if seg_len2 < 1e-9: return 0.0, (x2, y2), 0.0
    t = ((x - x1)*vx + (y - y1)*vy) / seg_len2
    t_clamp = max(0.0, min(1.0, t))
    xs = x1 + t_clamp*vx; ys = y1 + t_clamp*vy
    seg_len = math.sqrt(seg_len2)
    s = min(t_clamp*seg_len + Ld, seg_len)
    xt = x1 + (s/seg_len)*vx; yt = y1 + (s/seg_len)*vy
    psi_ref = math.atan2(yt - y, xt - x)
    ex, ey = x - xs, y - ys
    nx, ny = -vy/seg_len, vx/seg_len
    e_ct = ex*nx + ey*ny
    return psi_ref, (xt, yt), e_ct


def limit_speed_by_turn(epsi, d_target, omega_max, a_lat_max=0.6, v_cap=None, eps=1e-6):
    epsi_abs = abs(epsi)
    d = max(eps, d_target)
    kappa_req = 2.0 * math.sin(epsi_abs) / d
    v_omega = omega_max / max(eps, kappa_req)
    v_a = math.sqrt(max(0.0, a_lat_max / max(eps, kappa_req)))
    v_lim = min(v_omega, v_a)
    if v_cap is not None: v_lim = min(v_lim, v_cap)
    return v_lim


# ---- Команда на гидролокатор (SSS) ----
def send_sonar_cmd(event: str, enable: bool, port: str, baudrate: int, timeout: float = 1.0) -> bool:
    """
    Отправка простого сигнала START/END на SSS.
    По умолчанию шлём один байт 0xFF — при необходимости замените на протокол вашего SSS.
    """
    if not enable:
        print(f"[SSS:{event}] disabled")
        return True

    cmd = bytes([0xFF])
    if serial is None:
        print(f"[SSS:{event}] EMU send {cmd.hex().upper()} -> {port}@{baudrate}")
        return True
    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as ser:
            ser.write(cmd); ser.flush()
        print(f"[SSS:{event}] sent {cmd.hex().upper()} -> {port}@{baudrate}")
        return True
    except Exception as e:
        print(f"[SSS:{event}][ERROR] {e}")
        return False


def draw_cov_rectangle(ax, S, E, width, color="#6EC1FF", alpha=0.15, label=None):
    dx, dy = E[0] - S[0], E[1] - S[1]
    L = math.hypot(dx, dy)
    if L < 1e-6: return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    w2 = width / 2.0
    p1 = (S[0] + nx * w2, S[1] + ny * w2)
    p2 = (E[0] + nx * w2, E[1] + ny * w2)
    p3 = (E[0] - nx * w2, E[1] - ny * w2)
    p4 = (S[0] - nx * w2, S[1] - ny * w2)
    ax.add_patch(mpatches.Polygon([p1, p2, p3, p4], closed=True,
                                  facecolor=color, edgecolor="none",
                                  alpha=alpha, label=label))


# =========================
# КОНТРОЛЛЕРЫ + ДВУХДВИЖИТЕЛЬНАЯ МОДЕЛЬ (БФС ДРК)
# =========================
class YawPD:
    def __init__(self, K1_kurs, K2_kurs, use_err_psi, use_err_r, meas_scale):
        self.K1 = float(K1_kurs); self.K2 = float(K2_kurs)
        self.use_err_psi = bool(use_err_psi)
        self.use_err_r   = bool(use_err_r)
        self.meas_scale  = float(meas_scale)

    def _meas_psi(self, psi):  return self.meas_scale * psi if self.use_err_psi else psi
    def _meas_r(self, r):      return self.meas_scale * r   if self.use_err_r   else r

    def command(self, psi_ref_deg, psi_rad, r_rad_s) -> float:
        psi_m = self._meas_psi(psi_rad)
        r_m   = self._meas_r(r_rad_s)
        e_deg = psi_ref_deg - psi_m * RAD2DEG
        while e_deg <= -180.0: e_deg += 360.0
        while e_deg >   180.0: e_deg -= 360.0
        return self.K1 * e_deg - self.K2 * (r_m * RAD2DEG)


class SurgePDRef:
    """PD по продольной скорости (в теле), с фильтром ссылки; возвращает суммарную силу Tcmd (Н)."""
    def __init__(self, Meff, C1, C2, K1_marsh, K2_marsh, wn, zeta, use_err_u, meas_scale, dt):
        self.Meff = float(Meff); self.C1 = float(C1); self.C2 = float(C2)
        self.K1 = max(0.0, float(K1_marsh))   # P
        self.K2 = max(0.0, float(K2_marsh))   # D
        self.wn = max(1e-6, float(wn)); self.zeta = max(0.2, float(zeta))
        self.use_err_u = bool(use_err_u); self.meas_scale = float(meas_scale)
        self.dt = float(dt)
        self.ur = 0.0; self.urd = 0.0

    def _meas_u(self, u): return self.meas_scale * u if self.use_err_u else u

    def _update_ref_filter(self, u_ref):
        ydd = self.wn*self.wn*(u_ref - self.ur) - 2.0*self.zeta*self.wn*self.urd
        self.urd += self.dt * ydd; self.ur += self.dt * self.urd
        return self.ur, self.urd

    def command_T(self, u_ref_body, u_body, udot_body) -> float:
        uref, uref_dot = self._update_ref_filter(u_ref_body)
        u_meas = self._meas_u(u_body)
        e = uref - u_meas
        edot = uref_dot - udot_body
        Tff  = self.C2*u_meas + self.C1*abs(u_meas)*u_meas
        return self.Meff*(self.K1*e + self.K2*edot) + Tff


class AUV_BFS_DRC:
    """2 движителя + БФС (марш/курс) + гидродинамика yaw/surge."""
    def __init__(self, p: Dict[str, Any]):
        # геом/гидро yaw
        self.Jy = p["Jy"]; self.L55 = p["L55"]
        self.Cwy1 = p["Cwy1"]; self.Cwy2 = p["Cwy2"]
        # гидро surge
        self.Meff = p["m"] + p["lam11"]
        self.Cx1 = p["Cxu1"]; self.Cx2 = p["Cxu2"]
        # интегратор
        self.dt = p["dt"]

        # течения
        self.Fcur_yaw = p["Fcur_yaw_mag"]
        self.th_cur_yaw = p["Fcur_yaw_dir_deg"] * DEG2RAD
        self.l_yaw_arm = p["l_yaw_arm"]
        self.Ucur_mag = p["Ucur_mag"]
        self.Ucur_dir = p["Ucur_dir_deg"] * DEG2RAD

        # внешнее возмущение по курсу
        self.Mdist = p["Mdist"]

        # движители/микшер
        self.Kdx_thr = p["Kdx_thr"]; self.Tdx_thr = p["Tdx_thr"]; self.Umax_thr = abs(p["Umax_thr"])
        self.mix_surge = p["mix_gain_surge"]; self.mix_yaw = p["mix_gain_yaw"]

        # контроллеры
        self.yaw_ctrl = YawPD(p["K1_kurs"], p["K2_kurs"], p["use_5pct_error_psi"], p["use_5pct_error_r"], p["meas_scale"])
        self.surge_ctrl = SurgePDRef(self.Meff, self.Cx1, self.Cx2, p["K1_marsh"], p["K2_marsh"],
                                     p["wn_ref"], p["zeta_ref"], p["use_5pct_error_u"], p["meas_scale"], self.dt)

        # состояния
        self.psi = 0.0; self.r = 0.0; self.u = 0.0
        self.x = 0.0; self.y = 0.0
        self.TL = 0.0; self.TR = 0.0

        # логи моментов
        self._last_M_thr = 0.0
        self._last_M_net = 0.0

    # --- внутренняя физика ---
    def _drag_u(self, u): return self.Cx1*abs(u)*u + self.Cx2*u

    def _thruster_dyn(self, T, U):
        return T + self.dt * ((self.Kdx_thr*U - T) / self.Tdx_thr)

    def _mix_and_saturate(self, U_surge, U_yaw):
        UL = self.mix_surge*U_surge - self.mix_yaw*U_yaw
        UR = self.mix_surge*U_surge + self.mix_yaw*U_yaw
        m = max(abs(UL), abs(UR), 1e-9)
        if m > self.Umax_thr:
            scale = self.Umax_thr / m
            UL *= scale; UR *= scale
        return UL, UR

    def step(self, u_ref_ground, psi_ref_deg):
        # ground -> тело (компенсация проекции течения вдоль корпуса)
        Uc_par = self.Ucur_mag * math.cos(self.Ucur_dir - self.psi)
        u_ref_body = max(0.0, u_ref_ground - Uc_par)

        udot = (self.TL + self.TR - self._drag_u(self.u)) / self.Meff

        U_yaw_cmd = self.yaw_ctrl.command(psi_ref_deg, self.psi, self.r)
        Tcmd_sum  = self.surge_ctrl.command_T(u_ref_body, self.u, udot)
        U_surge_cmd = Tcmd_sum / (2.0 * self.Kdx_thr)

        UL, UR = self._mix_and_saturate(U_surge_cmd, U_yaw_cmd)

        self.TL = self._thruster_dyn(self.TL, UL)
        self.TR = self._thruster_dyn(self.TR, UR)

        M_thr = self.l_yaw_arm * (self.TR - self.TL)
        M_cur = self.l_yaw_arm * self.Fcur_yaw * math.sin(self.th_cur_yaw - self.psi)
        Jtot = self.Jy + self.L55
        rdot = (M_thr + self.Mdist + M_cur - self.Cwy1*abs(self.r)*self.r - self.Cwy2*self.r) / Jtot
        self.r  += self.dt * rdot
        self.psi = wrap_rad(self.psi + self.dt * self.r)

        udot = (self.TL + self.TR - self._drag_u(self.u)) / self.Meff
        self.u += self.dt * udot

        Vx = self.u * math.cos(self.psi) + self.Ucur_mag * math.cos(self.Ucur_dir)
        Vy = self.u * math.sin(self.psi) + self.Ucur_mag * math.sin(self.Ucur_dir)
        self.x += self.dt * Vx; self.y += self.dt * Vy

        self._last_M_thr = M_thr
        self._last_M_net = M_thr + self.Mdist + M_cur - self.Cwy1*abs(self.r)*self.r - self.Cwy2*self.r

        return UL, UR, M_thr, self._last_M_net  # логи


# =========================
# СИМУЛЯТОР МАРШРУТА + ПЕРЕХОДНЫЕ
# =========================
@dataclass
class SimResult:
    xs: List[float]; ys: List[float]
    wps: List[Tuple[float, float]]
    rect: Tuple[float, float, float, float]                 # расширенный прямоугольник
    acq_spans: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    camera_events: List[Tuple[str, Tuple[float, float], float]]
    t_hist: List[float]; psi_deg: List[float]; u_ground: List[float]; e_ct_hist: List[float]
    uL_hist: List[float]; uR_hist: List[float]; TL_hist: List[float]; TR_hist: List[float]
    Mthr_hist: List[float]; Mnet_hist: List[float]
    meta: Dict[str, Any]                                     # содержит rect_orig (исходный)


def _estimate_Ts(p: Dict[str, Any]) -> Tuple[float, float]:
    """Короткая оценка Ts по yaw/surge для авто-тюна."""
    dt = p["dt"]
    # yaw
    test = AUV_BFS_DRC(p)
    tloc=0.0; Ts_yaw=8.0; reached=False
    while tloc < 8.0:
        tloc += dt
        test.step(0.0, 1.0)
        err = abs(1.0 - test.psi*RAD2DEG)
        if not reached and err <= 0.02: reached=True; Ts_yaw=tloc
        if reached and err > 0.02: reached=False; Ts_yaw=8.0
    # surge (по ground-скорости для оценки)
    test = AUV_BFS_DRC(p); tloc=0.0; Ts_surge=12.0; reached=False
    while tloc < 12.0:
        tloc += dt
        test.step(1.0, 0.0)
        Vx = test.u*math.cos(test.psi) + p["Ucur_mag"]*math.cos(p["Ucur_dir_deg"]*DEG2RAD)
        Vy = test.u*math.sin(test.psi) + p["Ucur_mag"]*math.sin(p["Ucur_dir_deg"]*DEG2RAD)
        ug = math.hypot(Vx, Vy)
        err = abs(1.0 - ug)
        if not reached and err <= 0.02: reached=True; Ts_surge=tloc
        if reached and err > 0.02: reached=False; Ts_surge=12.0
    return Ts_yaw, Ts_surge


def _auto_tune_for_2s(p: Dict[str, Any]) -> Dict[str, Any]:
    """Мягкая автонастройка под Ts≈2 c (масштабирование коэффициентов)."""
    if not p.get("auto_tune_2s", False):
        return p
    p = dict(p)
    Ts_yaw, Ts_surge = _estimate_Ts(p)

    # yaw
    if Ts_yaw > 2.0:
        scale_yaw = min(4.0, max(0.5, Ts_yaw / 2.0))
        p["K1_kurs"] *= scale_yaw
        p["K2_kurs"] *= scale_yaw

    # surge
    if Ts_surge > 2.0:
        scale_surge = min(4.0, max(0.5, Ts_surge / 2.0))
        p["K1_marsh"] *= scale_surge
        p["K2_marsh"] *= scale_surge
        p["wn_ref"] = min(4.0*p["wn_ref"], p["wn_ref"]*scale_surge)

    return p


def _inflate_rect_from_center(rect: Tuple[float, float, float, float], frac: float) -> Tuple[float, float, float, float]:
    """Увеличить прямоугольник (+frac, например 0.10) ОТ ЦЕНТРА."""
    x, y, w, h = rect
    cx = x + w/2.0
    cy = y + h/2.0
    w2 = w * (1.0 + frac)
    h2 = h * (1.0 + frac)
    x2 = cx - w2/2.0
    y2 = cy - h2/2.0
    return (x2, y2, w2, h2)


def run_simulation(p_in: Dict[str, Any]) -> SimResult:
    # авто-подстройка
    p = _auto_tune_for_2s(p_in)

    # исходный прямоугольник
    rect_orig = (p["rect_x"], p["rect_y"], p["rect_w"], p["rect_h"])
    # расширяем от центра на +10%
    rect = _inflate_rect_from_center(rect_orig, 0.10)

    heading_deg = p["heading_deg"]

    h_alt = p["h_alt"]; R_slant = p["R_slant"]
    beta_deg = p["beta_deg"]; cross_overlap = p["cross_overlap"]; along_overlap = p["along_overlap"]

    U_cruise = p["U_cruise"]; U_acq = p["U_acq"]
    lookahead = p["lookahead"]; switch_R = p["switch_R"]
    a_lat_max = p["a_lat_max"]; dt = p["dt"]

    omega_max = p["b"] * p["Umax_yaw"]

    W_use, lane_spacing = sss_swath_and_spacing(R_slant, h_alt, beta_deg, cross_overlap)
    s_ping = (1.0 - along_overlap) * W_use
    dt_ping = s_ping / max(1e-6, U_acq)

    # строим маршрут в РАСШИРЕННОМ прямоугольнике
    wps = build_lawnmower_in_rect_from_corner(rect, lane_spacing, heading_deg, margin=0.0)

    # Оценка Ts (после возможной подстройки)
    Ts_yaw, Ts_surge = _estimate_Ts(p)
    turn_buffer = 1.2 * U_cruise * max(Ts_yaw, Ts_surge)  # расстояние на стабилизацию перед/после поворота

    # --- ПАРАМЕТРЫ ЗАЩИТЫ НАЧАЛА/КОНЦА ---
    guard_before = float(p.get("guard_before_turn_m", 10.0))  # END раньше поворота
    guard_after  = float(p.get("guard_after_turn_m", 8.0))    # START после поворота
    straight_tol = float(p.get("straight_angle_tol_deg", 8.0)) * DEG2RAD

    # Рабочие отрезки съёмки — строго внутри прямых галсов
    def build_acq_spans_on_straights(wps_list, buf_turn, guard_before_m, guard_after_m):
        spans=[]
        for i in range(1, len(wps_list)-1, 2):  # каждая пара i->i+1 — прямой галс
            A, B = wps_list[i], wps_list[i+1]
            L = math.hypot(B[0]-A[0], B[1]-A[1])
            if L <= (buf_turn + guard_after_m) + (buf_turn + guard_before_m):
                continue
            dx, dy = (B[0]-A[0])/L, (B[1]-A[1])/L
            # S — после выхода из поворота + дополнительный запас (guard_after)
            S = (A[0] + dx*(buf_turn + guard_after_m), A[1] + dy*(buf_turn + guard_after_m))
            # E — до входа в следующий поворот - дополнительный запас (guard_before)
            E = (B[0] - dx*(buf_turn + guard_before_m), B[1] - dy*(buf_turn + guard_before_m))
            spans.append((S, E))
        return spans

    acq_spans = build_acq_spans_on_straights(wps, turn_buffer, guard_before, guard_after)

    spans_ex=[]
    for S, E in acq_spans:
        dx, dy = E[0]-S[0], E[1]-S[1]; L = math.hypot(dx, dy)
        if L < 1e-6: continue
        ux, uy = dx/L, dy/L
        psi_span = math.atan2(uy, ux)
        spans_ex.append({
            "S":S,"E":E,"u":(ux,uy),"L":L,
            "psi_span": psi_span,
            "start_fired":False,"end_fired":False
        })

    # Симулятор
    auv = AUV_BFS_DRC(p)
    x0, y0 = wps[0]
    if len(wps) >= 2:
        dx0, dy0 = wps[1][0]-wps[0][0], wps[1][1]-wps[0][1]
        auv.psi = math.atan2(dy0, dx0)
    auv.x, auv.y = x0, y0

    xs, ys = [auv.x], [auv.y]
    psi_deg = [auv.psi*RAD2DEG]
    u_ground = []
    e_ct_hist = [0.0]
    t_hist = [0.0]
    uL_hist, uR_hist, TL_hist, TR_hist = [], [], [], []
    Mthr_hist, Mnet_hist = [], []
    camera_events: List[Tuple[str, Tuple[float, float], float]] = []

    Ucx = p["Ucur_mag"] * math.cos(p["Ucur_dir_deg"]*DEG2RAD)
    Ucy = p["Ucur_mag"] * math.sin(p["Ucur_dir_deg"]*DEG2RAD)

    total_dist = sum(math.hypot(wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1]) for i in range(len(wps)-1))
    span_len_total = sum(math.hypot(E[0]-S[0], E[1]-S[1]) for (S, E) in acq_spans)
    span_len_total = max(0.0, min(span_len_total, total_dist))
    T_nom = (span_len_total / max(1e-6, U_acq)) + ((total_dist - span_len_total) / max(1e-6, U_cruise))
    T_end = 1.25 * T_nom

    t = 0.0; i_seg = 0
    x_prev, y_prev = auv.x, auv.y

    # допуски для событий
    EVENT_PERP_TOL  = float(p.get("event_perp_tol", 5.0))   # м
    EVENT_POINT_TOL = float(p.get("event_point_tol", 3.0))  # м

    max_steps = int(2.0 * T_end / dt) + 10000
    steps = 0

    while t < T_end and i_seg < len(wps)-1 and steps < max_steps:
        steps += 1
        wp_i, wp_j = wps[i_seg], wps[i_seg+1]
        psi_ref, target_pt, e_ct = los_pure_pursuit((auv.x, auv.y), wp_i, wp_j, p["lookahead"])
        xt, yt = target_pt
        d_target = math.hypot(xt - auv.x, yt - auv.y)

        # события SSS_START/SSS_END
        capturing = False
        for sp in spans_ex:
            Sx, Sy = sp["S"]; Ex, Ey = sp["E"]
            ux, uy = sp["u"];  L = sp["L"]
            psi_span = sp["psi_span"]

            # проекции и ⟂ расстояние
            s_prev = (x_prev - Sx)*ux + (y_prev - Sy)*uy
            s      = (auv.x  - Sx)*ux + (auv.y  - Sy)*uy
            d_perp = abs((auv.x - Sx)*(-uy) + (auv.y - Sy)*ux)

            # проверка прямолинейности по курсу
            is_straight = abs(wrap_rad(auv.psi - psi_span)) <= straight_tol

            dS = math.hypot(auv.x - Sx, auv.y - Sy)
            dE = math.hypot(auv.x - Ex, auv.y - Ey)

            # START: пересечение начала рабочего отрезка И ТОЛЬКО на прямой
            if (not sp["start_fired"]) and is_straight and (
                ((s_prev < 0.0) and (s >= 0.0) and (d_perp <= EVENT_PERP_TOL)) or
                (dS <= EVENT_POINT_TOL)
            ):
                _ = send_sonar_cmd("SSS_START", bool(p.get("sonar_serial_enable", False)),
                                   str(p.get("sonar_port", "/dev/ttyUSB0")), int(p.get("sonar_baud", 9600)))
                camera_events.append(("SSS_START", (Sx, Sy), t))
                sp["start_fired"] = True

            # END: пересечение конца рабочего отрезка И ТОЛЬКО на прямой
            if (not sp["end_fired"]) and is_straight and (
                ((s_prev < L) and (s >= L) and (d_perp <= EVENT_PERP_TOL)) or
                (dE <= EVENT_POINT_TOL)
            ):
                _ = send_sonar_cmd("SSS_END", bool(p.get("sonar_serial_enable", False)),
                                   str(p.get("sonar_port", "/dev/ttyUSB0")), int(p.get("sonar_baud", 9600)))
                camera_events.append(("SSS_END", (Ex, Ey), t))
                sp["end_fired"] = True

            # статус "идёт съёмка" — только на прямых в пределах рабочего окна
            if is_straight and (0.0 <= s <= L) and (d_perp <= EVENT_PERP_TOL):
                capturing = True

        # профили скорости
        dist_to_end = math.hypot(wp_j[0]-auv.x, wp_j[1]-auv.y)
        if capturing: u_ref_base = p["U_acq"]
        else:         u_ref_base = p["U_cruise"] * max(0.3, min(1.0, dist_to_end/(2.0*p["lookahead"])))

        epsi = wrap_rad(psi_ref - auv.psi)
        u_ref_turn = limit_speed_by_turn(epsi, d_target, omega_max, a_lat_max, u_ref_base)
        u_ref_ground = max(0.0, min(u_ref_base, u_ref_turn))

        UL, UR, M_thr, M_net = auv.step(u_ref_ground, psi_ref*RAD2DEG)

        # логи
        Vx = auv.u*math.cos(auv.psi) + Ucx
        Vy = auv.u*math.sin(auv.psi) + Ucy
        ug = math.hypot(Vx, Vy)

        x_prev, y_prev = auv.x, auv.y
        t += dt
        xs.append(auv.x); ys.append(auv.y)
        psi_deg.append(auv.psi*RAD2DEG)
        u_ground.append(ug); e_ct_hist.append(e_ct); t_hist.append(t)
        uL_hist.append(UL); uR_hist.append(UR); TL_hist.append(auv.TL); TR_hist.append(auv.TR)
        Mthr_hist.append(M_thr); Mnet_hist.append(M_net)

        if math.hypot(wp_j[0]-auv.x, wp_j[1]-auv.y) < p["switch_R"]:
            i_seg += 1

    meta = dict(W_use=W_use, lane_spacing=lane_spacing, s_ping=s_ping, dt_ping=dt_ping,
                turn_buffer=turn_buffer, U_acq=p["U_acq"], U_cruise=p["U_cruise"],
                ts_yaw=Ts_yaw, ts_surge=Ts_surge, autotune=p.get("auto_tune_2s", False),
                rect_orig=rect_orig)
    return SimResult(xs, ys, wps, rect, acq_spans, camera_events, t_hist, psi_deg,
                     u_ground, e_ct_hist, uL_hist, uR_hist, TL_hist, TR_hist, Mthr_hist, Mnet_hist, meta)


# ===== ПЕРЕХОДНЫЕ =====
@dataclass
class StepResponses:
    t_yaw: List[float]; yaw_deg: List[float]; yaw_ref: List[float]
    yaw_band: Tuple[float, float]; yaw_Mp_pct: float; yaw_Ts: float | None

    t_u_body: List[float]; u_body: List[float]; u_ref_body: List[float]
    u_body_band: Tuple[float, float]; u_body_Mp_pct: float; u_body_Ts: float | None

    t_x: List[float]; x_coord: List[float]; x_ref: List[float]
    x_band: Tuple[float, float]; x_Mp_pct: float; x_Ts: float | None


def _step_metrics(t: List[float], y: List[float], ref: List[float], tol=0.05):
    if not t: return (0.0, 0.0), 0.0, None
    yr = ref[-1]; A = max(abs(yr), 1e-9)
    y_low, y_high = yr - tol*A, yr + tol*A
    sgn = 1.0 if yr >= 0 else -1.0
    peak = max((yi - yr) * sgn for yi in y)
    Mp_pct = max(0.0, 100.0 * peak / A)
    Ts = None
    for i in range(len(t)):
        if all(abs(y[j]-yr) <= tol*A for j in range(i, len(t))):
            Ts = t[i]; break
    return (y_low, y_high), Mp_pct, Ts


def compute_step_responses(p_in: Dict[str, Any]) -> StepResponses:
    # те же настройки, что и в основной симуляции
    p = _auto_tune_for_2s(p_in)
    dt = p["dt"]

    # yaw step (чистый поворот)
    auv = AUV_BFS_DRC(p)
    T = max(dt, float(p["yaw_step_T"])); amp = float(p["yaw_step_amp_deg"])
    t_yaw=[0.0]; yaw=[auv.psi*RAD2DEG]; yref=[amp]; tcur=0.0
    while tcur < T:
        tcur += dt
        auv.step(0.0, amp)
        t_yaw.append(tcur); yaw.append(auv.psi*RAD2DEG); yref.append(amp)
    yaw_band, yaw_Mp, yaw_Ts = _step_metrics(t_yaw, yaw, yref, tol=0.05)

    # surge body step: ступенька по u(тело)
    auv = AUV_BFS_DRC(p)
    T = max(dt, float(p["surge_step_T"]))
    amp_ub = float(p.get("surge_step_amp_body", 1.0))
    t_ub=[0.0]; u_b=[auv.u]; u_ref_b=[amp_ub]; tcur=0.0
    while tcur < T:
        tcur += dt
        Uc_par = p["Ucur_mag"] * math.cos((p["Ucur_dir_deg"]*DEG2RAD) - auv.psi)
        u_ref_ground = max(0.0, amp_ub + Uc_par)
        auv.step(u_ref_ground, 0.0)
        u_b.append(auv.u); u_ref_b.append(amp_ub); t_ub.append(tcur)
    u_b_band, u_b_Mp, u_b_Ts = _step_metrics(t_ub, u_b, u_ref_b, tol=0.05)

    # position step (по X) с внешним ПД по положению
    Kp_x = float(p.get("Kp_x", 0.9)); Kd_x = float(p.get("Kd_x", 0.6))
    u_max_pos = float(p.get("u_max_pos", 2.0))
    x_amp = float(p.get("x_step_amp", 1.0)); Tpos = float(p.get("x_step_T", 12.0))
    auv = AUV_BFS_DRC(p)
    t_x=[0.0]; xs=[auv.x]; xref=[x_amp]; tcur=0.0
    while tcur < Tpos:
        tcur += dt
        vx_ground = auv.u*math.cos(auv.psi) + p["Ucur_mag"]*math.cos(p["Ucur_dir_deg"]*DEG2RAD)
        ex = x_amp - auv.x
        edot = -vx_ground
        u_cmd = Kp_x*ex + Kd_x*edot
        u_cmd = max(0.0, min(u_max_pos, u_cmd))
        auv.step(u_cmd, 0.0)
        xs.append(auv.x); xref.append(x_amp); t_x.append(tcur)
    x_band, x_Mp, x_Ts = _step_metrics(t_x, xs, xref, tol=0.05)

    return StepResponses(
        t_yaw, yaw, yref, yaw_band, yaw_Mp, yaw_Ts,
        t_ub, u_b, u_ref_b, u_b_band, u_b_Mp, u_b_Ts,
        t_x, xs, xref, x_band, x_Mp, x_Ts
    )


# =========================
# GUI
# =========================
@dataclass
class ParamRow:
    category: str; key: str; label: str; value: Any; ptype: str; unit: str; desc: str


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

        # Движение/ограничения
        ParamRow("Движение", "U_cruise", "U_cruise", 2.0, "float", "м/с", "Крейсерская наземная"),
        ParamRow("Движение", "U_acq", "U_acq", 1.0, "float", "м/с", "Рабочая наземная"),
        ParamRow("Движение", "lookahead", "lookahead", 6.0, "float", "м", "Длина упреждения LOS"),
        ParamRow("Движение", "switch_R", "switch_R", 2.0, "float", "м", "Радиус переключения сегмента"),
        ParamRow("Движение", "a_lat_max", "a_lat_max", 0.6, "float", "м/с²", "Макс. поперечное ускорение"),
        ParamRow("Движение", "dt", "dt", 0.02, "float", "с", "Шаг интегрирования"),

        # Yaw-гидродинам. (для модели и эвристики поворота)
        ParamRow("Yaw", "Jy", "Jy", 90.0, "float", "кг·м²", "Момент инерции по курсу"),
        ParamRow("Yaw", "L55", "L55", 87.0, "float", "кг·м²", "Добавочный момент инерции"),
        ParamRow("Yaw", "Cwy1", "Cwy1", 75.0, "float", "-", "Квадратное демпфирование"),
        ParamRow("Yaw", "Cwy2", "Cwy2", 7.4, "float", "-", "Линейное демпфирование"),
        ParamRow("Yaw", "b", "b", 0.3, "float", "-", "эвристика: ω_max≈b·Umax_yaw"),
        ParamRow("Yaw", "Umax_yaw", "Umax_yaw", 10.0, "float", "В", "эвристика капа скорости"),

        # Yaw.ПД
        ParamRow("Yaw.ПД", "K1_kurs", "K1_kurs", 1.2, "float", "В/°", "P для курса"),
        ParamRow("Yaw.ПД", "K2_kurs", "K2_kurs", 1.0, "float", "В/(°/с)", "D для курса"),
        ParamRow("Yaw.Течение", "Fcur_yaw_mag", "F_cur (yaw)", 0.0, "float", "Н", "Сила течения для момента"),
        ParamRow("Yaw.Течение", "Fcur_yaw_dir_deg", "θ_cur (yaw)", 0.0, "float", "°", "Азимут течения"),
        ParamRow("Yaw.Течение", "l_yaw_arm", "l_arm", 1.0, "float", "м", "Плечо приложения силы"),
        ParamRow("Yaw.Внешнее", "Mdist", "Mdist", 0.0, "float", "Н·м", "Пост. возмущающий момент"),

        # Surge-гидродинам. и регулятор
        ParamRow("Surge", "m", "m", 90.0, "float", "кг", "Масса"),
        ParamRow("Surge", "lam11", "λ11", 78.0, "float", "кг", "Добавочная масса"),
        ParamRow("Surge", "Cxu1", "Cxu1", 37.0, "float", "-", "Квадратное сопротивление"),
        ParamRow("Surge", "Cxu2", "Cxu2", 3.7, "float", "-", "Линейное сопротивление"),
        ParamRow("Surge.ПД", "K1_marsh", "K1_marsh", 2.6, "float", "-", "P (марш)"),
        ParamRow("Surge.ПД", "K2_marsh", "K2_marsh", 1.8, "float", "1/с", "D (марш)"),
        ParamRow("Surge.ПД", "wn_ref", "ω_n", 1.8, "float", "рад/с", "Частота фильтра ссылки"),
        ParamRow("Surge.ПД", "zeta_ref", "ζ", 0.75, "float", "-", "Демпфирование фильтра ссылки"),
        ParamRow("Surge.Течение", "Ucur_mag", "U_cur", 0.0, "float", "м/с", "Скорость течения"),
        ParamRow("Surge.Течение", "Ucur_dir_deg", "θ_cur", 0.0, "float", "°", "Азимут течения"),

        # ДРК/микшер
        ParamRow("ДРК", "Kdx_thr", "Kdx_thr", 20.0, "float", "Н/В", "Коэф. привода движителя"),
        ParamRow("ДРК", "Tdx_thr", "Tdx_thr", 0.15, "float", "с", "Пост. времени движителя"),
        ParamRow("ДРК", "Umax_thr", "U_thr,max", 10.0, "float", "В", "Сатурация входа движителя"),
        ParamRow("ДРК", "mix_gain_surge", "mix_surge", 1.0, "float", "-", "Коэф. общего канала"),
        ParamRow("ДРК", "mix_gain_yaw", "mix_yaw", 1.0, "float", "-", "Коэф. дифф. канала"),

        # Ошибки измерений
        ParamRow("Ошибки", "use_5pct_error_psi", "err ψ +5%", False, "bool", "-", ""),
        ParamRow("Ошибки", "use_5pct_error_r",   "err r +5%", False, "bool", "-", ""),
        ParamRow("Ошибки", "use_5pct_error_u",   "err u +5%", False, "bool", "-", ""),
        ParamRow("Ошибки", "meas_scale", "meas_scale", 1.05, "float", "-", ""),

        # Переходные
        ParamRow("Переходные", "yaw_step_amp_deg", "Δψ_step", 1.0, "float", "°", "Ступенька по курсу"),
        ParamRow("Переходные", "yaw_step_T", "T_step(yaw)", 8.0, "float", "с", "Длительность (yaw)"),
        ParamRow("Переходные", "surge_step_amp_body", "Δu_body_step", 1.0, "float", "м/с", "Ступенька по u(тело)"),
        ParamRow("Переходные", "surge_step_T", "T_step(surge)", 12.0, "float", "с", "Длительность (surge)"),

        # Позиция (наружный ПД для координаты X)
        ParamRow("Позиция.ПД", "Kp_x", "Kp_x", 0.9, "float", "м⁻¹·(м/с)", "P по координате (генерирует u_ref)"),
        ParamRow("Позиция.ПД", "Kd_x", "Kd_x", 0.6, "float", "с⁻¹", "D по координате (через ẋ)"),
        ParamRow("Позиция.ПД", "u_max_pos", "u_max_pos", 2.0, "float", "м/с", "Ограничение u_ref в позиционном контуре"),
        ParamRow("Позиция.ПД", "x_step_amp", "Δx_step", 1.0, "float", "м", "Ступенька по X"),
        ParamRow("Позиция.ПД", "x_step_T", "T_step(x)", 12.0, "float", "с", "Длительность (x)"),

        # Автонастройка
        ParamRow("Автонастройка", "auto_tune_2s", "Автонастройка Ts≈2 c", True, "bool", "-", "Поджать Ts до ~2 c"),

        # SSS/Съёмка — защиты
        ParamRow("SSS.Съёмка", "acq_extend_pct", "Удлинение рабочего отрезка", 0.10, "float", "дол.", "±доля длины (не используется при явных guard*)"),
        ParamRow("SSS.Съёмка", "guard_before_turn_m", "Guard до поворота (END)", 12.0, "float", "м", "Завершить съёмку заранее"),
        ParamRow("SSS.Съёмка", "guard_after_turn_m", "Guard после поворота (START)", 8.0, "float", "м", "Начать съёмку после выхода"),
        ParamRow("SSS.Съёмка", "straight_angle_tol_deg", "Допуск прямолинейности, °", 8.0, "float", "°", "Макс. отклонение курса от оси галса"),
        ParamRow("SSS.Съёмка", "event_perp_tol", "Допуск ⟂, м", 5.0, "float", "м", "Порог перпендикулярной близости"),
        ParamRow("SSS.Съёмка", "event_point_tol", "Допуск точки, м", 3.0, "float", "м", "Порог близости к S/E"),
        ParamRow("SSS.Съёмка", "sonar_serial_enable", "Отпр. в COM", False, "bool", "-", "Вкл/выкл отправку"),
        ParamRow("SSS.Съёмка", "sonar_port", "Порт SSS", "/dev/ttyUSB0", "str", "-", "Напр. COM3 (Windows)"),
        ParamRow("SSS.Съёмка", "sonar_baud", "Baud SSS", 9600, "int", "бод", "Скорость порта"),
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
        self.spec_map: Dict[str, ParamRow] = {}   # ключ -> спецификация строки
        self.populate(rows)
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def populate(self, rows: List[ParamRow]):
        self.setRowCount(len(rows))
        self.spec_map = {r.key: r for r in rows}
        for i, r in enumerate(rows):
            cat = QtWidgets.QTableWidgetItem(r.category); cat.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            key = QtWidgets.QTableWidgetItem(r.label);   key.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)

            if r.ptype == "bool":
                val = QtWidgets.QTableWidgetItem()
                val.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable|QtCore.Qt.ItemFlag.ItemIsEnabled|QtCore.Qt.ItemFlag.ItemIsSelectable)
                val.setCheckState(QtCore.Qt.CheckState.Checked if bool(r.value) else QtCore.Qt.CheckState.Unchecked)
                val.setText("True" if bool(r.value) else "False")
            elif r.ptype == "int":
                val = QtWidgets.QTableWidgetItem(str(r.value))
                val.setFlags(QtCore.Qt.ItemFlag.ItemIsEditable|QtCore.Qt.ItemFlag.ItemIsEnabled|QtCore.Qt.ItemFlag.ItemIsSelectable)
                val.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignVCenter)
            elif r.ptype == "float":
                val = QtWidgets.QTableWidgetItem(str(r.value))
                val.setFlags(QtCore.Qt.ItemFlag.ItemIsEditable|QtCore.Qt.ItemFlag.ItemIsEnabled|QtCore.Qt.ItemFlag.ItemIsSelectable)
                val.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignVCenter)
            else:
                val = QtWidgets.QTableWidgetItem(str(r.value))
                val.setFlags(QtCore.Qt.ItemFlag.ItemIsEditable|QtCore.Qt.ItemFlag.ItemIsEnabled|QtCore.Qt.ItemFlag.ItemIsSelectable)

            unit = QtWidgets.QTableWidgetItem(r.unit); unit.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            desc = QtWidgets.QTableWidgetItem(r.desc); desc.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)

            for c, it in enumerate((cat, key, val, unit, desc)):
                self.setItem(i, c, it)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, r.key)

        self.setSortingEnabled(True)
        self.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)

    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for r in range(self.rowCount()):
            key = self.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            val_item = self.item(r, 2)
            spec = self.spec_map.get(key)
            if spec is None: continue
            if spec.ptype == "bool":
                params[key] = (val_item.checkState() == QtCore.Qt.CheckState.Checked)
            elif spec.ptype == "int":
                txt = (val_item.text() or "").strip()
                try:
                    params[key] = int(float(txt))
                except Exception:
                    val_item.setBackground(QtGui.QBrush(QtGui.QColor("#ffcccc")))
                    raise ValueError(f"Неверное целое в параметре: {spec.label}")
            elif spec.ptype == "float":
                txt = (val_item.text() or "").strip().replace(",", ".")
                try:
                    params[key] = float(txt)
                except Exception:
                    val_item.setBackground(QtGui.QBrush(QtGui.QColor("#ffcccc")))
                    raise ValueError(f"Неверное число в параметре: {spec.label}")
            else:
                params[key] = (val_item.text() or "").strip()
        missing = [r.key for r in self._rows if r.key not in params]
        if missing:
            raise ValueError("В таблице отсутствуют значения: " + ", ".join(missing))
        return params

    def reset_defaults(self): self.populate(default_param_rows())

    def load_from_dict(self, d: Dict[str, Any]):
        legacy = {
            "K1": "K1_kurs", "K2": "K2_kurs",
            "Kp": "K1_marsh", "Kd": "K2_marsh",
            "Kp_marsh": "K1_marsh", "Kd_marsh": "K2_marsh"
        }
        for old, new in legacy.items():
            if old in d and new not in d:
                d[new] = d[old]

        for r in range(self.rowCount()):
            key = self.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            if key not in d: continue
            spec = self.spec_map.get(key)
            if spec is None: continue
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
        layout.addWidget(self.toolbar); layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111); self.ax.set_title(title); self.ax.grid(True)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUV Survey — БФС ДРК, маршрут и переходные")
        self.resize(1600, 950)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        self.param_table = ParamTable(default_param_rows()); splitter.addWidget(self.param_table)

        self.tabs = QtWidgets.QTabWidget(); splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 7)

        self.tab_plan = PlotTab("План (траектория и покрытие)")
        self.tab_yaw = PlotTab("Курс ψ(t)")
        self.tab_speed = PlotTab("Скорость (ground) v(t)")
        self.tab_ect = PlotTab("Поперечная ошибка e_ct(t)")
        self.tab_step_yaw = PlotTab("Переходная по курсу")
        self.tab_step_u_body = PlotTab("Переходная u(тело)")
        self.tab_step_x = PlotTab("Переходная по координате X")
        self.tab_ul = PlotTab("Входы движителей U_L/U_R (В)")
        self.tab_tl = PlotTab("Тяга движителей T_L/T_R (Н)")
        self.tab_mom = PlotTab("Моменты (yaw): M_thr и M_net")

        for tab, name in [
            (self.tab_plan, "План"), (self.tab_yaw, "Курс"),
            (self.tab_speed, "Скорость"), (self.tab_ect, "e_ct"),
            (self.tab_step_yaw, "Переходная (курс)"),
            (self.tab_step_u_body, "Переходная u(тело)"),
            (self.tab_step_x, "Переходная по координате"),
            (self.tab_ul, "U_L/U_R"), (self.tab_tl, "T_L/T_R"),
            (self.tab_mom, "Моменты"),
        ]: self.tabs.addTab(tab, name)

        # Панель управления
        btn_run   = QtWidgets.QPushButton("Запустить")
        btn_reset = QtWidgets.QPushButton("Сброс")
        btn_save  = QtWidgets.QPushButton("Сохранить пресет…")
        btn_load  = QtWidgets.QPushButton("Загрузить пресет…")
        self.chk_legend = QtWidgets.QCheckBox("Легенда"); self.chk_legend.setChecked(True)

        btn_run.clicked.connect(self.on_run); btn_reset.clicked.connect(self.on_reset)
        btn_save.clicked.connect(self.on_save); btn_load.clicked.connect(self.on_load)

        top_bar = QtWidgets.QWidget(); top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.addWidget(btn_run); top_layout.addWidget(btn_reset); top_layout.addStretch(1)
        top_layout.addWidget(self.chk_legend); top_layout.addWidget(btn_save); top_layout.addWidget(btn_load)

        self.addToolBarBreak(); tool = QtWidgets.QToolBar("Управление"); tool.addWidget(top_bar)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tool)

        self.statusBar().showMessage("Готово")

    def on_reset(self):
        self.param_table.reset_defaults()
        self.statusBar().showMessage("Параметры сброшены.")

    def on_save(self):
        try:
            params = self.param_table.to_params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка параметров", str(e)); return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить пресет", filter="JSON (*.json)")
        if fn:
            with open(fn, "w", encoding="utf-8") as f: json.dump(params, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f"Пресет сохранён: {fn}")

    def on_load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Загрузить пресет", filter="JSON (*.json)")
        if fn:
            with open(fn, "r", encoding="utf-8") as f: data = json.load(f)
            self.param_table.load_from_dict(data)
            self.statusBar().showMessage(f"Пресет загружен: {fn}")

    def on_run(self):
        try:
            params = self.param_table.to_params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка параметров", str(e)); return

        try:
            res = run_simulation(params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка симуляции", str(e)); return

        try:
            steps = compute_step_responses(params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка переходных", str(e)); return

        # очистка осей
        for tab in (self.tab_plan, self.tab_yaw, self.tab_speed, self.tab_ect,
                    self.tab_step_yaw, self.tab_step_u_body,
                    self.tab_step_x, self.tab_ul, self.tab_tl, self.tab_mom):
            tab.fig.clf(); tab.ax = tab.fig.add_subplot(111); tab.ax.grid(True)

        # ---- План
        ax = self.tab_plan.ax

        # расширенный сектор (+10%) — тот, по которому строился маршрут
        xmin, ymin, W, H = res.rect
        ax.plot([xmin, xmin+W, xmin+W, xmin, xmin],
                [ymin, ymin, ymin+H, ymin+H, ymin],
                linestyle="-", color="k", linewidth=1.6,
                label=("Сектор (+10%)" if self.chk_legend.isChecked() else None))

        # исходный заданный сектор (для сравнения, пунктиром)
        if "rect_orig" in res.meta:
            x0, y0, W0, H0 = res.meta["rect_orig"]
            ax.plot([x0, x0+W0, x0+W0, x0, x0],
                    [y0, y0, y0+H0, y0+H0, y0],
                    linestyle="--", color="gray", linewidth=1.2,
                    label=("Сектор (заданный)" if self.chk_legend.isChecked() else None))

        # Маршрут (галсы)
        if len(res.wps) >= 2:
            wx, wy = zip(*res.wps)
            ax.plot(wx, wy, ":", color="gray", alpha=0.7, linewidth=1.2, label="Маршрут (галсы)")

        # Покрытие SSS — полная ширина
        W_use = res.meta["W_use"]
        drew_cov_label = False
        for i in range(1, len(res.wps)-1, 2):
            A, B = res.wps[i], res.wps[i+1]
            label = "Покрытие SSS (полная зона)" if (self.chk_legend.isChecked() and not drew_cov_label) else None
            draw_cov_rectangle(ax, A, B, width=W_use, alpha=0.18, label=label)
            drew_cov_label = True

        # Траектория и рабочие отрезки (зелёным — активная съёмка)
        ax.plot(res.xs, res.ys, label="Траектория ПА (ground)")
        for (S, E) in res.acq_spans:
            ax.plot([S[0], E[0]], [S[1], E[1]], linewidth=3.0, alpha=0.95, color="limegreen")

        # Точки SSS_START/SSS_END (яркие и поверх всего)
        added_start = added_end = False
        for ev, P, _tt in res.camera_events:
            if ev == "SSS_START":
                self.tab_plan.ax.scatter(
                    P[0], P[1], s=220, marker="*", zorder=12,
                    facecolors="yellow", edgecolors="black", linewidths=1.2,
                    label=("SSS_START" if self.chk_legend.isChecked() and not added_start else None)
                )
                added_start = True
            elif ev == "SSS_END":
                self.tab_plan.ax.scatter(
                    P[0], P[1], s=200, marker="P", zorder=12,
                    facecolors="crimson", edgecolors="black", linewidths=1.2,
                    label=("SSS_END" if self.chk_legend.isChecked() and not added_end else None)
                )
                added_end = True

        # Старт/финиш ПА
        ax.scatter(res.wps[0][0], res.wps[0][1], s=200, color="red", marker="o",
                   edgecolors="black", linewidths=1.2, zorder=11,
                   label=("Старт ПА" if self.chk_legend.isChecked() else None))
        ax.scatter(res.wps[-1][0], res.wps[-1][1], s=200, color="red", marker="X",
                   edgecolors="black", linewidths=1.2, zorder=11,
                   label=("Финиш ПА" if self.chk_legend.isChecked() else None))

        ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
        autotune_txt = " | auto:ON" if res.meta.get("autotune", False) else ""
        ax.set_title(
            f"SSS: W≈{res.meta['W_use']:.1f} м | lane≈{res.meta['lane_spacing']:.1f} м | "
            f"U_acq={res.meta['U_acq']:.1f} м/с | buffer≈{res.meta['turn_buffer']:.1f} м | "
            f"t_s(yaw)≈{res.meta['ts_yaw']:.2f}s | t_s(surge)≈{res.meta['ts_surge']:.2f}s{autotune_txt}"
        )
        if self.chk_legend.isChecked(): ax.legend(loc="best")
        self.tab_plan.canvas.draw()

        # ---- Курс
        ax = self.tab_yaw.ax
        ax.plot(res.t_hist, res.psi_deg)
        ax.set_xlabel("t, c"); ax.set_ylabel("ψ, °"); ax.set_title("Курс ψ(t)")
        self.tab_yaw.canvas.draw()

        # ---- Скорость (ground)
        ax = self.tab_speed.ax
        t_sp = res.t_hist[1:] if len(res.t_hist) == len(res.u_ground)+1 else res.t_hist[:len(res.u_ground)]
        ax.plot(t_sp, res.u_ground)
        ax.set_xlabel("t, c"); ax.set_ylabel("|V_ground|, м/с"); ax.set_title("Скорость (ground) v(t)")
        self.tab_speed.canvas.draw()

        # ---- e_ct
        ax = self.tab_ect.ax
        ax.plot(res.t_hist, res.e_ct_hist); ax.set_xlabel("t, c"); ax.set_ylabel("e_ct, м")
        ax.set_title("Поперечная ошибка e_ct(t)"); self.tab_ect.canvas.draw()

        # ---- Переходная по курсу (+числа)
        ax = self.tab_step_yaw.ax
        ax.plot(steps.t_yaw, steps.yaw_deg, label="ψ(t)")
        ax.plot(steps.t_yaw, steps.yaw_ref, linestyle="--", label="ψ_ref")
        ax.axhline(steps.yaw_band[0], linestyle="--", linewidth=1)
        ax.axhline(steps.yaw_band[1], linestyle="--", linewidth=1)
        if steps.yaw_Ts is not None: ax.axvline(steps.yaw_Ts, linestyle=":", linewidth=1, label=f"Ts≈{steps.yaw_Ts:.2f}s")
        ax.set_title("Переходная по курсу"); ax.set_xlabel("t, c"); ax.set_ylabel("ψ, °")
        if self.chk_legend.isChecked():
            yr = steps.yaw_ref[-1]
            txt = f"Mp ≈ {steps.yaw_Mp_pct:.1f}% | ±5% [{steps.yaw_band[0]:.2f}; {steps.yaw_band[1]:.2f}] | y*≈{yr:.2f}°"
            ax.legend(loc="best", title=txt)
        self.tab_step_yaw.canvas.draw()

        # ---- Переходная u(тело) (+числа)
        ax = self.tab_step_u_body.ax
        ax.plot(steps.t_u_body, steps.u_body, label="u(тело)")
        ax.plot(steps.t_u_body, steps.u_ref_body, linestyle="--", label="u_ref(body)")
        ax.axhline(steps.u_body_band[0], linestyle="--", linewidth=1)
        ax.axhline(steps.u_body_band[1], linestyle="--", linewidth=1)
        if steps.u_body_Ts is not None: ax.axvline(steps.u_body_Ts, linestyle=":", linewidth=1, label=f"Ts≈{steps.u_body_Ts:.2f}s")
        ax.set_title("Переходная по линейной скорости u (тело)"); ax.set_xlabel("t, c"); ax.set_ylabel("м/с")
        if self.chk_legend.isChecked():
            yr = steps.u_ref_body[-1]
            txt = f"Mp ≈ {steps.u_body_Mp_pct:.1f}% | ±5% [{steps.u_body_band[0]:.2f}; {steps.u_body_band[1]:.2f}] | y*≈{yr:.2f} м/с"
            ax.legend(loc="best", title=txt)
        self.tab_step_u_body.canvas.draw()

        # ---- Переходная по координате X (+числа)
        ax = self.tab_step_x.ax
        ax.plot(steps.t_x, steps.x_coord, label="x(t)")
        ax.plot(steps.t_x, steps.x_ref, linestyle="--", label="x_ref")
        ax.axhline(steps.x_band[0], linestyle="--", linewidth=1)
        ax.axhline(steps.x_band[1], linestyle="--", linewidth=1)
        if steps.x_Ts is not None: ax.axvline(steps.x_Ts, linestyle=":", linewidth=1, label=f"Ts≈{steps.x_Ts:.2f}s")
        ax.set_title("Переходная по координате X (наружный ПД→u_ref)"); ax.set_xlabel("t, c"); ax.set_ylabel("м")
        if self.chk_legend.isChecked():
            yr = steps.x_ref[-1]
            txt = f"Mp ≈ {steps.x_Mp_pct:.1f}% | ±5% [{steps.x_band[0]:.2f}; {steps.x_band[1]:.2f}] | y*≈{yr:.2f} м"
            ax.legend(loc="best", title=txt)
        self.tab_step_x.canvas.draw()

        # ---- Входы движителей
        ax = self.tab_ul.ax
        ax.plot(res.t_hist[1:len(res.uL_hist)+1], res.uL_hist, label="U_L")
        ax.plot(res.t_hist[1:len(res.uR_hist)+1], res.uR_hist, label="U_R")
        ax.axhline(params["Umax_thr"], linestyle=":", linewidth=1)
        ax.axhline(-params["Umax_thr"], linestyle=":", linewidth=1)
        ax.set_xlabel("t, c"); ax.set_ylabel("В"); ax.set_title("U_L/U_R (входы движителей)")
        if self.chk_legend.isChecked(): ax.legend(loc="best")
        self.tab_ul.canvas.draw()

        # ---- Тяги движителей
        ax = self.tab_tl.ax
        ax.plot(res.t_hist[1:len(res.TL_hist)+1], res.TL_hist, label="T_L")
        ax.plot(res.t_hist[1:len(res.TR_hist)+1], res.TR_hist, label="T_R")
        ax.set_xlabel("t, c"); ax.set_ylabel("Н"); ax.set_title("T_L/T_R (тяга)")
        if self.chk_legend.isChecked(): ax.legend(loc="best")
        self.tab_tl.canvas.draw()

        # ---- Моменты
        ax = self.tab_mom.ax
        t_m = res.t_hist[1:len(res.Mthr_hist)+1]
        ax.plot(t_m, res.Mthr_hist, label="M_thr (движители)")
        ax.plot(t_m, res.Mnet_hist, linestyle="--", label="M_net (суммарный)")
        ax.set_xlabel("t, c"); ax.set_ylabel("Н·м"); ax.set_title("Моменты по курсу")
        if self.chk_legend.isChecked(): ax.legend(loc="best")
        self.tab_mom.canvas.draw()

        # статус
        starts = sum(1 for ev,_,_ in res.camera_events if ev=="SSS_START")
        ends   = sum(1 for ev,_,_ in res.camera_events if ev=="SSS_END")
        total_L = sum(math.hypot(res.xs[i+1]-res.xs[i], res.ys[i+1]-res.ys[i]) for i in range(len(res.xs)-1))
        self.statusBar().showMessage(
            f"L≈{total_L:.1f} м; W≈{res.meta['W_use']:.1f} м; "
            f"lane≈{res.meta['lane_spacing']:.1f} м; s_ping≈{res.meta['s_ping']:.2f} м; dt_ping≈{res.meta['dt_ping']:.2f} c; "
            f"SSS_START={starts}, SSS_END={ends}"
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
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
