#!/usr/bin/env python3
"""
gui_panels.py — Mixin : construction et rafraîchissement des panneaux Tk
========================================================================
Pas de logique de contrôle. Les lectures de données passent par `self.state`
(RobotState) ; les attributs Tk (BooleanVar, StringVar, widgets) sont
initialisés dans RobotGUI.__init__.
"""

import tkinter as tk
from tkinter import messagebox
import math

from geometry import (
    MUSCLE_LABELS, VALVE_OF_MUSCLE, ADC_CH_OF_MUSCLE,
    stewart_inverse_kinematics,
)
from workspace import clamp_pose_to_workspace
from auto_sequence import summary as auto_seq_summary
from stewart_feedforward import P_MAX_BAR, P_MIN_BAR

from ui_theme import (
    BG, BG2, BG3, ACCENT, GREEN, RED, ORANGE, PURPLE, YELLOW, CYAN,
    TEXT, DIMTEXT, BORDER,
    FMONO, FLABEL, FBIG,
    AXIS_COLORS, GYRO_COLORS, SENSOR_COLORS,
    card as build_card,
)

# Cadences (Tk refresh indépendant de la control loop).
GUI_REFRESH_HZ  = 10
TARGET_HZ       = GUI_REFRESH_HZ 
CONTROL_LOOP_HZ = 13               # info seulement (vraie source : control_loop)
P_MAX = P_MAX_BAR
P_MIN = P_MIN_BAR


class GUIPanelsMixin:
    """Mixin _build_* / _update_* / _gui_update. Pas d'I2C, pas de contrôle.
    Toutes les données viennent de self.state (RobotState)."""

    # ─────────────────────────────────────────────────────────────────────────
    # Rafraîchissement GUI — 10 Hz
    # ─────────────────────────────────────────────────────────────────────────
    def _gui_update(self):
        if not self._running:
            return
        import time
        with self.state.lock:
            self.state.heartbeat_gui = time.perf_counter()
            estop_active = self.state.estop_requested
            imu   = self.state.imu
            pos   = self.state.positions_m[:]
            fk_t  = self.state.fk_translation_m
            fk_r  = self.state.fk_rotation_rad
            fk_info = dict(self.state.fk_info)
            p_meas = self.state.pressures_measured_bar[:]

        if estop_active:
            blink = (int(time.perf_counter() * 2) % 2) == 0
            self.emergency_btn.config(bg=RED if blink else "#ff8080")
            self.ack_estop_btn.config(bg=GREEN, fg=BG)
        else:
            self.emergency_btn.config(bg=RED)
            self.ack_estop_btn.config(bg=BG3, fg=GREEN)

        self._update_imu_display(imu)
        self._update_pos_display(pos)
        self._update_fk_display(fk_t, fk_r, fk_info)
        self._update_ff_status_display()
        self._update_pid_status_display()
        self._update_pressure_meas_display(p_meas)
        self.root.after(int(1000 / TARGET_HZ), self._gui_update)

    # ─────────────────────────────────────────────────────────────────────────
    # Mises à jour des valeurs numériques
    # ─────────────────────────────────────────────────────────────────────────
    def _update_pressure_meas_display(self, p_meas):
        for m in range(6):
            v = p_meas[m]
            pct = v / P_MAX if P_MAX > 0 else 0.0
            color = GREEN if pct < 0.4 else (ORANGE if pct < 0.75 else RED)
            self.pressure_meas_labels[m].config(text=f"{v:.2f} bar", fg=color)

    def _update_imu_display(self, d):
        self.accel_vals["X"].config(text=f"{d.accel.x:+7.3f}")
        self.accel_vals["Y"].config(text=f"{d.accel.y:+7.3f}")
        self.accel_vals["Z"].config(text=f"{d.accel.z:+7.3f}")
        self.gyro_vals["X"].config(text=f"{d.gyro.x:+7.3f}")
        self.gyro_vals["Y"].config(text=f"{d.gyro.y:+7.3f}")
        self.gyro_vals["Z"].config(text=f"{d.gyro.z:+7.3f}")

    def _update_pos_display(self, pos: list):
        bar_total_width = 140
        POS_MIN = 0.65
        POS_MAX = 1.010
        POS_SPAN = POS_MAX - POS_MIN
        marker_width = 3
        for i, p in enumerate(pos):
            self.pos_vals[i].config(text=f"{p:.4f} m")
            p_clamped = max(POS_MIN, min(POS_MAX, p))
            pct = (p_clamped - POS_MIN) / POS_SPAN
            x = int(pct * bar_total_width) - marker_width // 2
            x = max(0, min(bar_total_width - marker_width, x))
            if p < POS_MIN or p > POS_MAX:
                color = RED
            elif p < 0.75 or p > 0.99:
                color = ORANGE
            else:
                color = SENSOR_COLORS[i]
            self.pos_bars[i].place(x=x, y=0, width=marker_width, height=10)
            self.pos_bars[i].config(bg=color)
            self.pos_vals[i].config(
                fg=RED if (p < POS_MIN or p > POS_MAX) else
                   (ORANGE if (p < 0.75 or p > 0.99) else TEXT)
            )

    def _update_fk_display(self, t_m, r_rad, info):
        self.fk_pos_labels["x"].config(text=f"{t_m[0]:+.4f} m")
        self.fk_pos_labels["y"].config(text=f"{t_m[1]:+.4f} m")
        self.fk_pos_labels["z"].config(text=f"{t_m[2]:+.4f} m")
        self.fk_ori_labels["roll"].config(text=f"{math.degrees(r_rad[0]):+.2f}°")
        self.fk_ori_labels["pitch"].config(text=f"{math.degrees(r_rad[1]):+.2f}°")
        self.fk_ori_labels["yaw"].config(text=f"{math.degrees(r_rad[2]):+.2f}°")
        res_um = info.get('residual_norm', 0.0) * 1e6
        iters  = info.get('iterations', 0)
        converged = info.get('converged', False)
        self.fk_status_label.config(
            text=f"résidu: {res_um:6.1f} µm | iters: {iters:2d}",
            fg=GREEN if converged else ORANGE,
        )

    def _update_ff_status_display(self):
        with self.state.lock:
            tens   = list(self.state.ff_tensions_N)
            kappa  = list(self.state.ff_kappa)
            press  = list(self.state.ff_pressures_bar)
            sat    = list(self.state.ff_saturated)
            negT   = list(self.state.ff_neg_tension)
            ff_on  = self.state.config.ff_enabled
            err_t  = self.state.pose_error_t_m
            err_r  = self.state.pose_error_r_rad
            p_mes  = self.state.pressures_measured_bar[:]

        for m in range(6):
            col_t = ORANGE if negT[m] else TEXT
            self.ff_status_labels["tens"][m].config(
                text=f"{tens[m]:+8.1f}", fg=col_t)
            self.ff_status_labels["kappa"][m].config(
                text=f"{kappa[m]*100:5.2f}", fg=PURPLE)
            p = press[m]
            if not ff_on:
                cmd_color = DIMTEXT
            elif sat[m]:
                cmd_color = RED
            elif p < P_MAX * 0.4:
                cmd_color = GREEN
            else:
                cmd_color = ORANGE
            self.ff_status_labels["cmd"][m].config(text=f"{p:5.2f}", fg=cmd_color)
            pm = p_mes[m]
            pct_m = pm / P_MAX if P_MAX > 0 else 0.0
            pmes_color = GREEN if pct_m < 0.4 else (ORANGE if pct_m < 0.75 else RED)
            self.ff_status_labels["pmes"][m].config(text=f"{pm:5.2f}", fg=pmes_color)

        et_mm = tuple(x * 1000.0 for x in err_t)
        et_norm_mm = math.sqrt(sum(x * x for x in et_mm))
        self.err_t_label.config(
            text=(f"Δt=({et_mm[0]:+5.1f},{et_mm[1]:+5.1f},{et_mm[2]:+5.1f})mm "
                  f"‖‖={et_norm_mm:4.1f}")
        )
        er_deg = tuple(math.degrees(x) for x in err_r)
        er_norm_deg = math.sqrt(sum(x * x for x in er_deg))
        self.err_r_label.config(
            text=(f"Δr=({er_deg[0]:+4.1f},{er_deg[1]:+4.1f},{er_deg[2]:+4.1f})°  "
                  f"‖‖={er_norm_deg:4.2f}")
        )

    def _update_pid_status_display(self):
        """Termes P/I/D lus directement depuis pid_ctrl (force ou pression)."""
        if not hasattr(self, 'ff_pid_labels'):
            return
        with self.state.lock:
            errs    = list(self.state.pid_length_error_m)
            gates   = list(self.state.pid_gate_open)
            ff_p    = list(self.state.ff_pressures_bar)
            p_mes   = self.state.pressures_measured_bar[:]
            p_app   = self.state.pressures_commanded_bar[:]
            pid_on  = self.state.config.pid_enabled
            ff_on   = self.state.config.ff_enabled
            domain  = getattr(self.state.config, 'feedback_domain', 'pressure')
            dF_out  = list(self.state.pid_force_outputs_N)
            outs_bar = list(self.state.pid_outputs_bar)

        # FORCE → termes PID force (N) + ΔF. PRESSION → termes PID bar + sortie bar.
        force_mode = (domain == 'force')
        if pid_on and force_mode:
            p_terms = self.ctrl.force_pid_ctrl.p_terms_N.tolist()
            i_terms = self.ctrl.force_pid_ctrl.i_terms_N.tolist()
            d_terms = self.ctrl.force_pid_ctrl.d_terms_N.tolist()
            outs    = dF_out
        elif pid_on:
            p_terms = self.pid_ctrl.p_terms_bar.tolist()
            i_terms = self.pid_ctrl.i_terms_bar.tolist()
            d_terms = self.pid_ctrl.d_terms_bar.tolist()
            outs    = outs_bar
        else:
            p_terms = [0.0] * 6
            i_terms = [0.0] * 6
            d_terms = [0.0] * 6
            outs    = [0.0] * 6
        # Format : N en force mode (valeurs ~centaines), bar sinon.
        tfmt = "{:+6.0f}" if force_mode else "{:+5.2f}"

        for m in range(6):
            err_mm = errs[m] * 1000.0
            ae = abs(err_mm)
            if pid_on:
                err_col = GREEN if ae < 2.0 else (ORANGE if ae < 10.0 else RED)
            else:
                err_col = DIMTEXT
            self.ff_pid_labels["err"][m].config(
                text=f"{err_mm:+7.1f}" if pid_on else "—", fg=err_col)
            ff_v = ff_p[m]
            self.ff_pid_labels["ff"][m].config(
                text=f"{ff_v:5.2f}" if ff_on else "—",
                fg=YELLOW if ff_on else DIMTEXT)
            if pid_on and gates[m]:
                self.ff_pid_labels["dp_p"][m].config(text=tfmt.format(p_terms[m]), fg=GREEN)
                self.ff_pid_labels["dp_i"][m].config(text=tfmt.format(i_terms[m]), fg=PURPLE)
                self.ff_pid_labels["dp_d"][m].config(text=tfmt.format(d_terms[m]), fg=CYAN)
                self.ff_pid_labels["dp_tot"][m].config(
                    text=tfmt.format(outs[m]),
                    fg=ORANGE if abs(outs[m]) > 1e-3 else DIMTEXT)
            else:
                for k in ("dp_p", "dp_i", "dp_d", "dp_tot"):
                    self.ff_pid_labels[k][m].config(text="—", fg=DIMTEXT)
            tot = p_app[m]
            pct = tot / P_MAX if P_MAX > 0 else 0.0
            tot_col = GREEN if pct < 0.4 else (ORANGE if pct < 0.75 else RED)
            self.ff_pid_labels["total"][m].config(text=f"{tot:5.2f}", fg=tot_col)
            pm = p_mes[m]
            pct_m = pm / P_MAX if P_MAX > 0 else 0.0
            self.ff_pid_labels["pmes"][m].config(
                text=f"{pm:5.2f}",
                fg=GREEN if pct_m < 0.4 else (ORANGE if pct_m < 0.75 else RED))
            if pid_on:
                gate_txt = "▲" if gates[m] else "○"
                gate_col = GREEN if gates[m] else ORANGE
            else:
                gate_txt, gate_col = "—", DIMTEXT
            self.ff_pid_labels["gate"][m].config(text=gate_txt, fg=gate_col)

    def _update_valve_display(self, muscle: int, pressure: float):
        if not hasattr(self, 'valve_bars') or muscle >= len(self.valve_bars):
            return
        pct = pressure / P_MAX if P_MAX > 0 else 0.0
        color = GREEN if pct < 0.4 else (ORANGE if pct < 0.75 else RED)
        width = max(0, min(130, int(pct * 130)))
        self.valve_bars[muscle].place(x=0, y=0, height=10, width=width)
        self.valve_bars[muscle].config(bg=color)
        self.valve_val_labels[muscle].config(
            text=f"{pressure:.2f} bar", fg=color)

    # ─────────────────────────────────────────────────────────────────────────
    # Construction UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=BG, pady=4)
        hdr.pack(fill="x", padx=8)

        self.emergency_btn = tk.Button(
            hdr, text="🛑  ARRÊT", font=("Helvetica", 11, "bold"),
            bg=RED, fg=TEXT, relief="flat",
            activebackground="#ff6b6b", activeforeground=BG,
            cursor="hand2", padx=14, pady=6,
            command=self._emergency_zero)
        self.emergency_btn.pack(side="left", padx=(0, 10))

        self.ack_estop_btn = tk.Button(
            hdr, text="✅  ACQUITTER", font=("Helvetica", 11, "bold"),
            bg=BG3, fg=GREEN, relief="flat",
            activebackground=GREEN, activeforeground=BG,
            cursor="hand2", padx=14, pady=6,
            command=self._acknowledge_estop)
        self.ack_estop_btn.pack(side="left", padx=(0, 10))

        tk.Label(hdr, text="🤖  Robot Pneumatique", font=FBIG,
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr,
                 text=(f"  ctrl {CONTROL_LOOP_HZ} Hz  •  "
                       f"gui {GUI_REFRESH_HZ} Hz"),
                 font=FLABEL, bg=BG, fg=DIMTEXT).pack(side="left", padx=6)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=2)

        col_l = self._make_col(body)
        col_m = self._make_col(body)
        col_r = self._make_col(body)

        # Colonne gauche : positions, état FF, et panel cascade (sous l'état FF)
        self._build_pos_panel(col_l)
        self._build_ff_status_panel(col_l)
        self._build_ff_pid_panel(col_l)
        self._build_fk_panel(col_m)
        self._build_ik_panel(col_m)
        # Colonne droite : vannes, IMU (à la place de l'ancien panel cascade), contrôles
        self._build_valve_panel(col_r)
        self._build_imu_panel(col_r)
        self._build_control_panel(col_r)

        self.status_var = tk.StringVar(value="Prêt.")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Helvetica", 8), bg=BG3, fg=DIMTEXT,
                 anchor="w", padx=10, pady=2).pack(fill="x", side="bottom")

    @staticmethod
    def _make_col(parent):
        """Colonne sans scrollbar — tous les panneaux visibles d'un coup."""
        frame = tk.Frame(parent, bg=BG)
        frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        return frame

    def _build_imu_panel(self, parent):
        card = build_card(parent, "📡  IMU")
        self.accel_vals = {}
        self.gyro_vals  = {}
        for c, (txt, col) in enumerate([("", TEXT), ("X", ACCENT), ("Y", GREEN), ("Z", PURPLE)]):
            tk.Label(card, text=txt, font=("Courier New", 8), bg=BG2, fg=col,
                     width=4 if c == 0 else 9, anchor="center").grid(
                row=0, column=c, padx=2, pady=(0, 1))
        tk.Label(card, text="Accel", font=("Courier New", 8), bg=BG2,
                 fg=DIMTEXT, width=5, anchor="w").grid(row=1, column=0, padx=2, pady=1)
        for c, ax in enumerate(["X", "Y", "Z"], start=1):
            lbl = tk.Label(card, text="+0.000", font=FMONO,
                           bg=BG2, fg=AXIS_COLORS[ax], width=9, anchor="e")
            lbl.grid(row=1, column=c, padx=2, pady=1)
            self.accel_vals[ax] = lbl
        tk.Label(card, text="Gyro", font=("Courier New", 8), bg=BG2,
                 fg=DIMTEXT, width=5, anchor="w").grid(row=2, column=0, padx=2, pady=1)
        for c, ax in enumerate(["X", "Y", "Z"], start=1):
            lbl = tk.Label(card, text="+0.000", font=FMONO,
                           bg=BG2, fg=GYRO_COLORS[ax], width=9, anchor="e")
            lbl.grid(row=2, column=c, padx=2, pady=1)
            self.gyro_vals[ax] = lbl
        tk.Label(card, text="m/s²", font=("Courier New", 7), bg=BG2,
                 fg=DIMTEXT).grid(row=3, column=1, columnspan=3, sticky="w", padx=2, pady=(0, 2))
        tk.Frame(card, bg=BORDER, height=1).grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=(2, 2))
        tk.Label(card, text="Offsets calib", font=("Courier New", 7),
                 bg=BG2, fg=DIMTEXT).grid(row=5, column=0, columnspan=4, sticky="w", padx=2)
        self.imu_off_labels = []
        for i, lbl in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
            l = tk.Label(card, text=f"{lbl}:+0.000",
                         font=("Courier New", 7), bg=BG2, fg=DIMTEXT,
                         width=11, anchor="w")
            l.grid(row=6 + i // 3, column=i % 3 + (1 if i >= 3 else 0),
                   padx=2, pady=0, sticky="w")
            self.imu_off_labels.append(l)

    def _build_pos_panel(self, parent):
        card = build_card(parent, "📏  Capteurs de position (plage 0.65 m – 1.010 m)")
        self.pos_bars           = []
        self.pos_vals           = []
        self.pos_off_labels     = []
        self.pos_target_labels  = []
        self.pos_target_markers = []
        POS_MIN = 0.65; POS_MAX = 1.010; POS_SPAN = POS_MAX - POS_MIN
        for m in range(6):
            col   = SENSOR_COLORS[m]
            hw_ch = ADC_CH_OF_MUSCLE[m]
            tk.Label(card, text=f"M{m} {MUSCLE_LABELS[m]}", font=FMONO,
                     bg=BG2, fg=col, width=6).grid(
                row=m, column=0, padx=(2, 2), pady=1, sticky="w")
            bg_f = tk.Frame(card, bg=BG3, height=10, width=140)
            bg_f.grid(row=m, column=1, padx=4, pady=1, sticky="w")
            bg_f.grid_propagate(False)
            rest_x = int((1.0 - POS_MIN) / POS_SPAN * 140)
            tk.Frame(bg_f, bg=DIMTEXT, width=2, height=10).place(x=rest_x - 1, y=0)
            bar = tk.Frame(bg_f, bg=col, height=10, width=3)
            bar.place(x=0, y=0, width=3, height=10)
            self.pos_bars.append(bar)
            target_marker = tk.Frame(bg_f, bg=YELLOW, width=2, height=10)
            target_marker.place_forget()
            self.pos_target_markers.append(target_marker)
            val = tk.Label(card, text="0.0000 m", font=FMONO,
                           bg=BG2, fg=TEXT, width=10, anchor="e")
            val.grid(row=m, column=2, padx=(4, 4), pady=1)
            self.pos_vals.append(val)
            tk.Label(card, text=f"CH{hw_ch}", font=("Courier New", 8),
                     bg=BG2, fg=DIMTEXT, width=4, anchor="w").grid(
                row=m, column=3, padx=(4, 4), pady=1, sticky="w")
            off_l = tk.Label(card, text="off: +0.0000",
                             font=("Courier New", 8), bg=BG2, fg=DIMTEXT,
                             width=13, anchor="w")
            off_l.grid(row=m, column=4, padx=4, pady=1, sticky="w")
            self.pos_off_labels.append(off_l)
            tgt_l = tk.Label(card, text="cible: —",
                             font=("Courier New", 8), bg=BG2, fg=DIMTEXT,
                             width=14, anchor="w")
            tgt_l.grid(row=m, column=5, padx=4, pady=1, sticky="w")
            self.pos_target_labels.append(tgt_l)
        legend = tk.Frame(card, bg=BG2)
        legend.grid(row=6, column=1, padx=4, pady=(2, 0), sticky="w")
        tk.Label(legend, text="0.65m", font=("Courier New", 7), bg=BG2, fg=DIMTEXT).pack(side="left")
        tk.Label(legend, text="1m",    font=("Courier New", 7, "bold"), bg=BG2, fg=TEXT,
                 width=18, anchor="e").pack(side="left")
        tk.Label(legend, text="1.01m", font=("Courier New", 7), bg=BG2, fg=DIMTEXT).pack(side="left")

    def _build_ff_status_panel(self, parent):
        card = build_card(parent, "📊  État Feedforward + erreur de pose")
        self.ff_status_labels = {"tens": [], "kappa": [], "cmd": [], "pmes": []}
        _F7 = ("Courier New", 7)
        _F8 = ("Courier New", 8)
        headers = [("Muscle", TEXT, 7, "w"), ("Tens(N)", DIMTEXT, 8, "e"),
                   ("κ%", PURPLE, 5, "e"), ("CMD bar", YELLOW, 7, "e"), ("P_mes", ACCENT, 6, "e")]
        for c, (txt, fg, w, anchor) in enumerate(headers):
            tk.Label(card, text=txt, font=_F7, bg=BG2, fg=fg, width=w, anchor=anchor).grid(
                row=0, column=c, padx=1, pady=(0, 1), sticky=anchor)
        for m in range(6):
            col = SENSOR_COLORS[m]
            tk.Label(card, text=f"M{m} {MUSCLE_LABELS[m]}", font=_F8,
                     bg=BG2, fg=col, width=7, anchor="w").grid(
                row=m + 1, column=0, padx=1, pady=0, sticky="w")
            for c_idx, key, w in ((1, "tens", 8), (2, "kappa", 5), (3, "cmd", 7), (4, "pmes", 6)):
                lbl = tk.Label(card, text="—", font=_F8, bg=BG2, fg=TEXT, width=w, anchor="e")
                lbl.grid(row=m + 1, column=c_idx, padx=1, pady=0, sticky="e")
                self.ff_status_labels[key].append(lbl)
        tk.Frame(card, bg=BORDER, height=1).grid(row=7, column=0, columnspan=5, sticky="ew", pady=(3, 1))
        tk.Label(card, text="Erreur de pose (cible − FK)", font=_F7,
                 bg=BG2, fg=DIMTEXT).grid(row=8, column=0, columnspan=5, sticky="w", padx=1, pady=(0, 0))
        self.err_t_label = tk.Label(card, text="Δt=(—,—,—)mm ‖‖=—",
                                    font=_F8, bg=BG2, fg=ACCENT, anchor="w")
        self.err_t_label.grid(row=9, column=0, columnspan=5, sticky="w", padx=2, pady=0)
        self.err_r_label = tk.Label(card, text="Δr=(—,—,—)°  ‖‖=—",
                                    font=_F8, bg=BG2, fg=ORANGE, anchor="w")
        self.err_r_label.grid(row=10, column=0, columnspan=5, sticky="w", padx=2, pady=0)
        tk.Label(card, text="FF=Jᵀ·T→Sarosi | P_mes=ADC",
                 font=_F7, bg=BG2, fg=DIMTEXT, justify="left").grid(
            row=11, column=0, columnspan=5, sticky="w", padx=1, pady=(1, 0))

    def _build_fk_panel(self, parent):
        card = build_card(parent, "📐  Cinématique directe (capteurs)")
        self.fk_pos_labels = {}
        self.fk_ori_labels = {}
        tk.Label(card, text="Position (m)", font=FLABEL, bg=BG2, fg=DIMTEXT).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=2, pady=(0, 2))
        for i, (ax, col) in enumerate([("x", ACCENT), ("y", GREEN), ("z", PURPLE)]):
            tk.Label(card, text=ax.upper(), font=FMONO, bg=BG2, fg=col, width=3).grid(
                row=i + 1, column=0, padx=2, pady=1)
            lbl = tk.Label(card, text="+0.0000 m", font=FMONO, bg=BG2, fg=TEXT, width=12, anchor="e")
            lbl.grid(row=i + 1, column=1, padx=4, pady=1)
            self.fk_pos_labels[ax] = lbl
        tk.Label(card, text="Orientation (°)", font=FLABEL, bg=BG2, fg=DIMTEXT).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=2, pady=(6, 2))
        for i, (ax, col) in enumerate([("roll", ORANGE), ("pitch", RED), ("yaw", CYAN)]):
            tk.Label(card, text=ax.capitalize(), font=FMONO, bg=BG2, fg=col, width=6).grid(
                row=i + 5, column=0, padx=2, pady=1)
            lbl = tk.Label(card, text="+0.00°", font=FMONO, bg=BG2, fg=TEXT, width=12, anchor="e")
            lbl.grid(row=i + 5, column=1, padx=4, pady=1)
            self.fk_ori_labels[ax] = lbl
        tk.Label(card, text="Convergence", font=FLABEL, bg=BG2, fg=DIMTEXT).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=2, pady=(6, 1))
        self.fk_status_label = tk.Label(card, text="résidu:    — | iters:  —",
                                        font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=24, anchor="w")
        self.fk_status_label.grid(row=9, column=0, columnspan=2, sticky="w", padx=4, pady=1)
        tk.Label(card, text="Pose calculée par inversion numérique\nde l'IK à partir des longueurs mesurées.",
                 font=("Helvetica", 8), bg=BG2, fg=DIMTEXT, justify="left").grid(
            row=10, column=0, columnspan=2, sticky="w", padx=2, pady=(4, 0))

    def _build_ik_panel(self, parent):
        card = build_card(parent, "🎯  Cible & Cinématique inverse")
        tk.Label(card, text="Pose cible (plateforme)", font=FLABEL, bg=BG2, fg=DIMTEXT).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=2, pady=(0, 2))
        for i, (label, color, key) in enumerate([("X", ACCENT, "x"), ("Y", GREEN, "y"), ("Z", PURPLE, "z")]):
            r = 1 + i
            tk.Label(card, text=label, font=FMONO, bg=BG2, fg=color, width=3).grid(row=r, column=0, padx=2, pady=1)
            entry = tk.Entry(card, textvariable=self.target_vars[key], font=FMONO,
                             bg=BG3, fg=TEXT, insertbackground=TEXT, relief="flat", width=9, justify="right")
            entry.grid(row=r, column=1, padx=4, pady=1)
            entry.bind("<Return>",   lambda e: self._compute_inverse_kinematics())
            entry.bind("<KP_Enter>", lambda e: self._compute_inverse_kinematics())
            tk.Label(card, text="m", font=FMONO, bg=BG2, fg=DIMTEXT).grid(row=r, column=2, sticky="w", padx=(2, 6))
        for i, (label, color, key) in enumerate([("α (roll)", ORANGE, "alpha"),
                                                  ("β (pitch)", RED, "beta"),
                                                  ("γ (yaw)", CYAN, "gamma")]):
            r = 4 + i
            tk.Label(card, text=label, font=FMONO, bg=BG2, fg=color, width=9, anchor="w").grid(
                row=r, column=0, padx=2, pady=1, sticky="w")
            entry = tk.Entry(card, textvariable=self.target_vars[key], font=FMONO,
                             bg=BG3, fg=TEXT, insertbackground=TEXT, relief="flat", width=9, justify="right")
            entry.grid(row=r, column=1, padx=4, pady=1)
            entry.bind("<Return>",   lambda e: self._compute_inverse_kinematics())
            entry.bind("<KP_Enter>", lambda e: self._compute_inverse_kinematics())
            tk.Label(card, text="°", font=FMONO, bg=BG2, fg=DIMTEXT).grid(row=r, column=2, sticky="w", padx=(2, 6))
        tk.Button(card, text="▶  Calculer IK", font=FLABEL, bg=BG3, fg=ACCENT, relief="flat",
                  activebackground=ACCENT, activeforeground=BG, cursor="hand2", padx=10, pady=3,
                  command=self._compute_inverse_kinematics).grid(
            row=7, column=0, columnspan=3, padx=4, pady=(4, 2), sticky="ew")
        tk.Label(card, text="Longueurs calculées affichées comme\nmarqueurs (▼) dans « Capteurs ».",
                 font=("Helvetica", 8), bg=BG2, fg=DIMTEXT, justify="left").grid(
            row=8, column=0, columnspan=4, sticky="w", padx=2, pady=(4, 0))

    def _compute_inverse_kinematics(self):
        try:
            tx = float(self.target_vars["x"].get())
            ty = float(self.target_vars["y"].get())
            tz = float(self.target_vars["z"].get())
            alpha_deg = float(self.target_vars["alpha"].get())
            beta_deg  = float(self.target_vars["beta"].get())
            gamma_deg = float(self.target_vars["gamma"].get())
        except ValueError:
            messagebox.showerror("Cible invalide",
                                 "Valeurs numériques attendues pour x, y, z (m) et α, β, γ (°).")
            return
        translation = (tx, ty, tz)
        rotation = (math.radians(alpha_deg), math.radians(beta_deg), math.radians(gamma_deg))
        try:
            t_clamped, r_clamped, ws_info = clamp_pose_to_workspace(
                translation, rotation,
                reference_translation_m=(0.0, 0.0, self._workspace_center_z),
            )
        except RuntimeError as e:
            messagebox.showerror("Workspace", f"Projection impossible :\n{e}")
            return
        clamped = not ws_info["in_workspace"]
        if clamped:
            self.target_vars["x"].set(f"{t_clamped[0]:.3f}")
            self.target_vars["y"].set(f"{t_clamped[1]:.3f}")
            self.target_vars["z"].set(f"{t_clamped[2]:.3f}")
            self.target_vars["alpha"].set(f"{math.degrees(r_clamped[0]):.2f}")
            self.target_vars["beta"].set(f"{math.degrees(r_clamped[1]):.2f}")
            self.target_vars["gamma"].set(f"{math.degrees(r_clamped[2]):.2f}")
            translation = tuple(float(v) for v in t_clamped)
            rotation    = tuple(float(v) for v in r_clamped)
        try:
            lengths = stewart_inverse_kinematics(translation, rotation)
        except Exception as e:
            messagebox.showerror("Erreur IK", f"Calcul impossible :\n{e}")
            return
        with self.state.lock:
            self.target_lengths = lengths.tolist()
            self.state.target_lengths_m = lengths.tolist()
            self.state.target_translation_m = translation
            self.state.target_rotation_rad  = rotation
        self._place_target_markers()
        if clamped:
            tx_e, ty_e, tz_e = translation
            a_e = math.degrees(rotation[0]); b_e = math.degrees(rotation[1]); g_e = math.degrees(rotation[2])
            self.status_var.set(
                f"⚠️  Pose hors workspace — clampée à λ={ws_info['scale']:.3f} : "
                f"({tx_e:+.3f}, {ty_e:+.3f}, {tz_e:+.3f}) m, ({a_e:+.1f}, {b_e:+.1f}, {g_e:+.1f})°")
        else:
            self.status_var.set(
                f"IK OK — cible ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) m, "
                f"({alpha_deg:+.1f}, {beta_deg:+.1f}, {gamma_deg:+.1f})°")

    def _place_target_markers(self):
        bar_total_width = 140
        POS_MIN = 0.65; POS_MAX = 1.010; POS_SPAN = POS_MAX - POS_MIN
        marker_width = 2
        for i in range(6):
            L = self.target_lengths[i]
            L_rest = self.rest_lengths[i]
            target_pos = 1.0 + (L - L_rest)
            out_of_range = target_pos < POS_MIN or target_pos > POS_MAX
            self.pos_target_labels[i].config(
                text=f"cible: {target_pos:.3f} m", fg=RED if out_of_range else DIMTEXT)
            if out_of_range:
                self.pos_target_markers[i].place_forget()
                continue
            pct = (target_pos - POS_MIN) / POS_SPAN
            x = int(pct * bar_total_width) - marker_width // 2
            x = max(0, min(bar_total_width - marker_width, x))
            self.pos_target_markers[i].place(x=x, y=0, width=marker_width, height=10)

    def _build_valve_panel(self, parent):
        card = build_card(parent, "💨  Commande des muscles (0 – 6 bar)")
        self.valve_bars       = []
        self.valve_val_labels = []
        self.pressure_meas_labels = []
        tk.Label(card, text="ON",          font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=3).grid(row=0, column=0, padx=(2, 2), pady=(2, 2))
        tk.Label(card, text="Mus.",        font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=6).grid(row=0, column=1, padx=(2, 2), pady=(2, 2))
        tk.Label(card, text="V#",          font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=3).grid(row=0, column=2, padx=(0, 2), pady=(2, 2))
        tk.Label(card, text="cmd",         font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=6).grid(row=0, column=3, padx=4, pady=(2, 2))
        tk.Label(card, text="set",         font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=4).grid(row=0, column=4, padx=4, pady=(2, 2))
        tk.Label(card, text="appliquée",   font=("Courier New", 8), bg=BG2, fg=DIMTEXT).grid(row=0, column=5, columnspan=2, padx=4, pady=(2, 2), sticky="w")
        tk.Label(card, text="P_mes (bar)", font=("Courier New", 8), bg=BG2, fg=ACCENT, width=11).grid(row=0, column=7, padx=(8, 4), pady=(2, 2))
        for m in range(6):
            row = m + 1
            col = SENSOR_COLORS[m]
            hw_valve = VALVE_OF_MUSCLE[m]
            cb = tk.Checkbutton(card, variable=self.muscle_active[m],
                                bg=BG2, fg=col, selectcolor=BG3,
                                activebackground=BG2, activeforeground=col,
                                bd=0, highlightthickness=0,
                                command=lambda idx=m: self._on_muscle_active_toggled(idx))
            cb.grid(row=row, column=0, padx=(2, 2), pady=2)
            tk.Label(card, text=f"M{m} {MUSCLE_LABELS[m]}", font=FMONO, bg=BG2, fg=col, width=6).grid(
                row=row, column=1, padx=(2, 2), pady=2, sticky="w")
            tk.Label(card, text=f"V{hw_valve}", font=("Courier New", 8), bg=BG2, fg=DIMTEXT, width=3).grid(
                row=row, column=2, padx=(0, 2), pady=2)
            entry = tk.Entry(card, textvariable=self.pressure_vars[m], font=FMONO,
                             bg=BG3, fg=TEXT, insertbackground=TEXT, relief="flat", width=6, justify="right")
            entry.grid(row=row, column=3, padx=4, pady=2)
            entry.bind("<Return>",   lambda e, idx=m: self._apply_pressure(idx))
            entry.bind("<KP_Enter>", lambda e, idx=m: self._apply_pressure(idx))
            tk.Button(card, text="SET", font=FLABEL, bg=BG3, fg=ACCENT, relief="flat",
                      activebackground=ACCENT, activeforeground=BG, cursor="hand2", padx=6,
                      command=lambda idx=m: self._apply_pressure(idx)).grid(row=row, column=4, padx=4, pady=2)
            bg_f = tk.Frame(card, bg=BG3, height=10, width=130)
            bg_f.grid(row=row, column=5, padx=4, pady=2, sticky="w")
            bg_f.grid_propagate(False)
            bar = tk.Frame(bg_f, bg=GREEN, height=10, width=0)
            bar.place(x=0, y=0, height=10)
            self.valve_bars.append(bar)
            val_lbl = tk.Label(card, text="0.00 bar", font=FMONO, bg=BG2, fg=GREEN, width=9, anchor="e")
            val_lbl.grid(row=row, column=6, padx=(4, 4), pady=2)
            self.valve_val_labels.append(val_lbl)
            meas_lbl = tk.Label(card, text="—", font=FMONO, bg=BG2, fg=ACCENT, width=11, anchor="e")
            meas_lbl.grid(row=row, column=7, padx=(8, 4), pady=2)
            self.pressure_meas_labels.append(meas_lbl)
        sel_frame = tk.Frame(card, bg=BG2)
        sel_frame.grid(row=7, column=0, columnspan=8, padx=4, pady=(4, 1), sticky="ew")
        tk.Label(sel_frame, text="Sélection :", font=("Courier New", 8), bg=BG2, fg=DIMTEXT).pack(side="left", padx=(2, 6))
        tk.Button(sel_frame, text="Tout activer", font=FLABEL, bg=BG3, fg=GREEN, relief="flat",
                  activebackground=GREEN, activeforeground=BG, cursor="hand2", padx=8,
                  command=lambda: self._set_all_muscles_active(True)).pack(side="left", padx=2)
        tk.Button(sel_frame, text="Tout désactiver", font=FLABEL, bg=BG3, fg=ORANGE, relief="flat",
                  activebackground=ORANGE, activeforeground=BG, cursor="hand2", padx=8,
                  command=lambda: self._set_all_muscles_active(False)).pack(side="left", padx=2)
        tk.Button(card, text="▶  Appliquer tout (muscles actifs uniquement)", font=FLABEL,
                  bg=BG3, fg=GREEN, relief="flat", activebackground=GREEN, activeforeground=BG,
                  cursor="hand2", padx=10, pady=3, command=self._apply_all_pressures).grid(
            row=8, column=0, columnspan=8, padx=4, pady=(2, 2), sticky="ew")

    def _set_all_muscles_active(self, active: bool):
        for m in range(6):
            was = self.muscle_active[m].get()
            self.muscle_active[m].set(active)
            if was != active:
                self._on_muscle_active_toggled(m)
        self.status_var.set(
            "▶  Tous les muscles activés" if active else "⏸  Tous les muscles désactivés (vannes à 0)")

    def _build_ff_pid_panel(self, parent):
        card = build_card(parent, "🎯  Contrôle Cascade : FF + PID length + PID pression")
        row = 0

        params_row = tk.Frame(card, bg=BG2)
        params_row.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 2))
        tk.Label(params_row, text="Masse (kg)", font=FLABEL, bg=BG2, fg=DIMTEXT).pack(side="left", padx=(0, 2))
        tk.Entry(params_row, textvariable=self.mass_var, font=FMONO,
                 bg=BG3, fg=TEXT, insertbackground=TEXT, relief="flat", width=8, justify="right").pack(side="left", padx=(0, 8))
        tk.Label(params_row, text="L_min (m)", font=FLABEL, bg=BG2, fg=DIMTEXT).pack(side="left", padx=(0, 2))
        tk.Entry(params_row, textvariable=self.min_length_var, font=FMONO,
                 bg=BG3, fg=TEXT, insertbackground=TEXT, relief="flat", width=7, justify="right").pack(side="left")
        row += 1

        btn_row = tk.Frame(card, bg=BG2)
        btn_row.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 4))
        self.ff_btn = tk.Button(btn_row, text="▶  Feedforward", font=FLABEL,
                                bg=BG3, fg=GREEN, relief="flat",
                                activebackground=GREEN, activeforeground=BG,
                                cursor="hand2", padx=8, pady=3,
                                command=lambda: (self.ff_enabled.set(not self.ff_enabled.get()),
                                                 self._toggle_feedforward()))
        self.ff_btn.pack(side="left", padx=(0, 4), fill="x", expand=True)
        self.pid_btn = tk.Button(btn_row, text="▶  PID", font=FLABEL,
                                 bg=BG3, fg=GREEN, relief="flat",
                                 activebackground=GREEN, activeforeground=BG,
                                 cursor="hand2", padx=8, pady=3,
                                 command=lambda: (self.pid_enabled.set(not self.pid_enabled.get()),
                                                  self._toggle_pid()))
        self.pid_btn.pack(side="left", padx=(0, 4), fill="x", expand=True)
        tk.Button(btn_row, text="↺ Reset", font=FLABEL, bg=BG3, fg=ORANGE, relief="flat",
                  activebackground=ORANGE, activeforeground=BG, cursor="hand2", padx=6, pady=3,
                  command=self._reset_pid_state).pack(side="left", fill="x", expand=True)
        row += 1

        # Gains PID (pliable)
        gains_hdr = tk.Frame(card, bg=BG3)
        gains_hdr.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 0))
        self._pid_gains_expanded = tk.BooleanVar(value=False)

        gains_body = tk.Frame(card, bg=BG2)
        gains_toggle_lbl = tk.Label(gains_hdr, text="▶  Gains PID (cliquer pour éditer)",
                                    font=FLABEL, bg=BG3, fg=ACCENT, cursor="hand2", pady=2)
        gains_toggle_lbl.pack(fill="x", padx=4)
        gains_body.grid(row=row + 1, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 2))
        gains_body.grid_remove()

        def _toggle_gains():
            if self._pid_gains_expanded.get():
                gains_body.grid()
                gains_toggle_lbl.config(text="▼  Gains PID")
            else:
                gains_body.grid_remove()
                gains_toggle_lbl.config(text="▶  Gains PID (cliquer pour éditer)")

        gains_hdr.bind("<Button-1>", lambda e: (self._pid_gains_expanded.set(not self._pid_gains_expanded.get()), _toggle_gains()))
        gains_toggle_lbl.bind("<Button-1>", lambda e: (self._pid_gains_expanded.set(not self._pid_gains_expanded.get()), _toggle_gains()))

        for c, (lbl, var, unit, col) in enumerate([
            ("Kp", self.pid_kp_var,  "bar/m",     GREEN),
            ("Ki", self.pid_ki_var,  "bar/(m·s)", YELLOW),
            ("Kd", self.pid_kd_var,  "bar·s/m",  PURPLE),
        ]):
            sub = tk.Frame(gains_body, bg=BG2)
            sub.grid(row=0, column=c, sticky="ew", padx=2, pady=(0, 2))
            tk.Label(sub, text=lbl, font=FMONO, bg=BG2, fg=col, width=3).pack(side="left")
            tk.Entry(sub, textvariable=var, font=FMONO, bg=BG3, fg=TEXT, insertbackground=TEXT,
                     relief="flat", width=7, justify="right").pack(side="left", padx=(2, 2))
            tk.Label(sub, text=unit, font=("Courier New", 7), bg=BG2, fg=DIMTEXT).pack(side="left")
        tk.Button(gains_body, text="↻ Apply", font=FLABEL, bg=BG3, fg=ACCENT, relief="flat",
                  activebackground=ACCENT, activeforeground=BG, cursor="hand2", padx=4,
                  command=self._apply_pid_settings_from_gui).grid(row=0, column=3, sticky="ew", padx=2, pady=(0, 2))
        for c in range(4):
            gains_body.columnconfigure(c, weight=1)
        row += 2

        # Domaine de retour : 'pressure' (PID→bar+FF) ou 'force' (Gattringer).
        dom_row = tk.Frame(card, bg=BG2)
        dom_row.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 2))
        tk.Label(dom_row, text="Retour PID :", font=FLABEL, bg=BG2, fg=DIMTEXT).pack(side="left", padx=(0, 4))
        tk.Radiobutton(dom_row, text="Pression", variable=self.feedback_domain_var,
                       value="pressure", command=self._toggle_feedback_domain,
                       font=FLABEL, bg=BG2, fg=TEXT, selectcolor=BG3,
                       activebackground=BG2, activeforeground=TEXT,
                       cursor="hand2").pack(side="left", padx=(0, 4))
        tk.Radiobutton(dom_row, text="Force (Gattringer)", variable=self.feedback_domain_var,
                       value="force", command=self._toggle_feedback_domain,
                       font=FLABEL, bg=BG2, fg=TEXT, selectcolor=BG3,
                       activebackground=BG2, activeforeground=TEXT,
                       cursor="hand2").pack(side="left")
        row += 1

        # Gains PID force (pliable)
        fg_hdr = tk.Frame(card, bg=BG3)
        fg_hdr.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 0))
        self._force_gains_expanded = tk.BooleanVar(value=False)
        fg_body = tk.Frame(card, bg=BG2)
        fg_toggle_lbl = tk.Label(fg_hdr, text="▶  Gains PID Force (cliquer pour éditer)",
                                 font=FLABEL, bg=BG3, fg=ACCENT, cursor="hand2", pady=2)
        fg_toggle_lbl.pack(fill="x", padx=4)
        fg_body.grid(row=row + 1, column=0, columnspan=6, sticky="ew", padx=2, pady=(0, 2))
        fg_body.grid_remove()

        def _toggle_force_gains():
            if self._force_gains_expanded.get():
                fg_body.grid()
                fg_toggle_lbl.config(text="▼  Gains PID Force")
            else:
                fg_body.grid_remove()
                fg_toggle_lbl.config(text="▶  Gains PID Force (cliquer pour éditer)")

        fg_hdr.bind("<Button-1>", lambda e: (self._force_gains_expanded.set(not self._force_gains_expanded.get()), _toggle_force_gains()))
        fg_toggle_lbl.bind("<Button-1>", lambda e: (self._force_gains_expanded.set(not self._force_gains_expanded.get()), _toggle_force_gains()))

        for c, (lbl, var, unit, col) in enumerate([
            ("Kp", self.pid_force_kp_var, "N/m",     GREEN),
            ("Ki", self.pid_force_ki_var, "N/(m·s)", YELLOW),
            ("Kd", self.pid_force_kd_var, "N·s/m",  PURPLE),
        ]):
            sub = tk.Frame(fg_body, bg=BG2)
            sub.grid(row=0, column=c, sticky="ew", padx=2, pady=(0, 2))
            tk.Label(sub, text=lbl, font=FMONO, bg=BG2, fg=col, width=3).pack(side="left")
            tk.Entry(sub, textvariable=var, font=FMONO, bg=BG3, fg=TEXT, insertbackground=TEXT,
                     relief="flat", width=7, justify="right").pack(side="left", padx=(2, 2))
            tk.Label(sub, text=unit, font=("Courier New", 7), bg=BG2, fg=DIMTEXT).pack(side="left")
        tk.Button(fg_body, text="↻ Apply", font=FLABEL, bg=BG3, fg=ACCENT, relief="flat",
                  activebackground=ACCENT, activeforeground=BG, cursor="hand2", padx=4,
                  command=self._apply_force_pid_settings_from_gui).grid(row=0, column=3, sticky="ew", padx=2, pady=(0, 2))
        # Deadband commun (mm), appliqué par le ↻ Apply ci-dessus.
        db_sub = tk.Frame(fg_body, bg=BG2)
        db_sub.grid(row=1, column=0, columnspan=4, sticky="ew", padx=2, pady=(0, 2))
        tk.Label(db_sub, text="Deadband", font=FMONO, bg=BG2, fg=ACCENT).pack(side="left")
        tk.Entry(db_sub, textvariable=self.pid_deadband_var, font=FMONO, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=7, justify="right").pack(side="left", padx=(4, 2))
        tk.Label(db_sub, text="mm  (commun force+pression ; 0 = off)",
                 font=("Courier New", 7), bg=BG2, fg=DIMTEXT).pack(side="left")
        for c in range(4):
            fg_body.columnconfigure(c, weight=1)
        row += 2

        tk.Frame(card, bg=BORDER, height=1).grid(row=row, column=0, columnspan=6, sticky="ew", pady=(2, 2))
        row += 1

        # Tableau par muscle
        hdrs = [("Muscle", TEXT, 7, "w"), ("Err mm", ACCENT, 7, "e"), ("FF bar", YELLOW, 7, "e"),
                ("ΔP_P", GREEN, 6, "e"), ("ΔP_I", PURPLE, 6, "e"), ("ΔP_D", CYAN, 6, "e"),
                ("ΔP tot", ORANGE, 6, "e"), ("Total", TEXT, 7, "e"), ("P_mes", ACCENT, 7, "e"), ("Gate", GREEN, 5, "e")]
        for c, (txt, fg, w, anchor) in enumerate(hdrs):
            tk.Label(card, text=txt, font=("Courier New", 7), bg=BG2, fg=fg, width=w, anchor=anchor).grid(
                row=row, column=c, padx=1, pady=(0, 1), sticky=anchor)
        row += 1

        self.ff_pid_labels = {"err": [], "ff": [], "dp_p": [], "dp_i": [], "dp_d": [],
                              "dp_tot": [], "total": [], "pmes": [], "gate": []}
        for m in range(6):
            col = SENSOR_COLORS[m]
            tk.Label(card, text=f"M{m} {MUSCLE_LABELS[m]}", font=("Courier New", 8),
                     bg=BG2, fg=col, width=7, anchor="w").grid(row=row + m, column=0, padx=1, pady=0, sticky="w")
            for c, (key, fg, w) in enumerate([
                ("err", ACCENT, 7), ("ff", YELLOW, 7), ("dp_p", GREEN, 6), ("dp_i", PURPLE, 6),
                ("dp_d", CYAN, 6), ("dp_tot", ORANGE, 6), ("total", TEXT, 7), ("pmes", ACCENT, 7), ("gate", GREEN, 5)
            ], start=1):
                lbl = tk.Label(card, text="—", font=("Courier New", 8), bg=BG2, fg=fg, width=w, anchor="e")
                lbl.grid(row=row + m, column=c, padx=1, pady=0, sticky="e")
                self.ff_pid_labels[key].append(lbl)
        row += 6

        tk.Label(card, text="IK→Eq.stat→Sarosi(FF) | PID=Kp·err+Ki·∫err+Kd·d(err) | Total=FF+PID clampé",
                 font=("Helvetica", 7), bg=BG2, fg=DIMTEXT, justify="left").grid(
            row=row, column=0, columnspan=10, sticky="w", padx=2, pady=(2, 0))
        for c in range(10):
            card.columnconfigure(c, weight=1)

    def _build_control_panel(self, parent):
        card = build_card(parent, "⚙️  Contrôles")

        def make_collapsible(parent_frame, title, expanded=False):
            hdr = tk.Frame(parent_frame, bg=BG3, cursor="hand2")
            hdr.pack(fill="x", pady=(2, 0))
            state = {"expanded": expanded}
            lbl_text = tk.StringVar(value=f"{'▼' if expanded else '▶'}  {title}")
            lbl = tk.Label(hdr, textvariable=lbl_text, font=FLABEL, bg=BG3, fg=DIMTEXT, anchor="w", pady=2)
            lbl.pack(fill="x", padx=4)
            body = tk.Frame(parent_frame, bg=BG2)
            if expanded:
                body.pack(fill="x", padx=2)

            def toggle(e=None):
                state["expanded"] = not state["expanded"]
                if state["expanded"]:
                    body.pack(fill="x", padx=2)
                    lbl_text.set(f"▼  {title}")
                else:
                    body.forget()
                    lbl_text.set(f"▶  {title}")
            hdr.bind("<Button-1>", toggle)
            lbl.bind("<Button-1>", toggle)
            return body

        self.log_btn = tk.Button(card, text="📝  Démarrer log", font=FLABEL,
                                 bg=BG3, fg=PURPLE, relief="flat",
                                 activebackground=PURPLE, activeforeground=BG,
                                 cursor="hand2", padx=8, pady=4, command=self._toggle_logging)
        self.log_btn.pack(fill="x", pady=(0, 2))

        self.auto_seq_btn = tk.Button(card, text="▶  Démarrer séquence auto", font=FLABEL,
                                      bg=BG3, fg=GREEN, relief="flat",
                                      activebackground=GREEN, activeforeground=BG,
                                      cursor="hand2", padx=8, pady=4, command=self._toggle_auto_sequence)
        self.auto_seq_btn.pack(fill="x", pady=(0, 2))
        try:
            seq_info = auto_seq_summary()
        except Exception:
            seq_info = ""
        if seq_info:
            tk.Label(card, text=seq_info, font=("Helvetica", 7), bg=BG2, fg=DIMTEXT,
                     wraplength=260, justify="left").pack(anchor="w", pady=(0, 2))

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=4)

        # Calibration (collapsible)
        calib_body = make_collapsible(card, "Calibration", expanded=True)
        tk.Label(calib_body, text="Position statique", font=("Courier New", 7), bg=BG2, fg=DIMTEXT).pack(anchor="w")
        self.calib_btn = tk.Button(calib_body, text="🎯  Calibrer", font=FLABEL,
                                   bg=BG3, fg=YELLOW, relief="flat",
                                   activebackground=YELLOW, activeforeground=BG,
                                   cursor="hand2", padx=8, pady=3, command=self._start_calibration)
        self.calib_btn.pack(fill="x", pady=(0, 2))
        tk.Button(calib_body, text="↺  Reset offsets", font=FLABEL, bg=BG3, fg=DIMTEXT, relief="flat",
                  cursor="hand2", padx=8, pady=2, command=self._reset_calibration).pack(fill="x", pady=(0, 1))

        # Centre de masse (collapsible)
        com_body = make_collapsible(card, "Centre de masse (m)", expanded=False)
        com_frame = tk.Frame(com_body, bg=BG2)
        com_frame.pack(fill="x", pady=(2, 0))
        for axis_idx, (axis_name, axis_color, var) in enumerate([
            ("X", ACCENT, self.com_offset_x_var),
            ("Y", GREEN,  self.com_offset_y_var),
            ("Z", PURPLE, self.com_offset_z_var),
        ]):
            sub = tk.Frame(com_frame, bg=BG2)
            sub.grid(row=0, column=axis_idx, padx=2, sticky="w")
            tk.Label(sub, text=axis_name, font=FMONO, bg=BG2, fg=axis_color, width=2).pack(side="left")
            entry = tk.Entry(sub, textvariable=var, font=FMONO, bg=BG3, fg=TEXT,
                             insertbackground=TEXT, relief="flat", width=7, justify="right")
            entry.pack(side="left")
            entry.bind("<FocusOut>", lambda e: self._refresh_com_display())
            entry.bind("<Return>",   lambda e: self._refresh_com_display())
        tk.Button(com_body, text="↺  Reset COM (0,0,0)", font=FLABEL, bg=BG3, fg=DIMTEXT, relief="flat",
                  cursor="hand2", padx=8, pady=2, command=self._reset_com_offset).pack(fill="x", pady=(2, 1))
