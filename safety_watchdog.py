#!/usr/bin/env python3
"""
safety_watchdog.py — Surveillance hors-bande de la plateforme
==============================================================
Thread indépendant à 60 Hz, garantit la coupure des vannes même si la
control loop freeze. Surveille (par ordre de gravité décroissante) :
estop, main loop figé (heartbeat), sur-contraction multiple/seule,
pression commandée > P_MAX (bug), pression mesurée > P_MAX + marge
(vanne défectueuse).

Stratégies : coupure individuelle (`set_muscle_pressure(bus, m, 0)`) ou
totale (`clear_all(bus)` + `request_estop()`). Une fois l'estop armé,
seul un `clear_estop()` explicite le réarme.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Callable, Optional

import smbus2

from robot_state import RobotState, SafetyEvent
from dac import set_muscle_pressure, clear_all, P_MAX
from geometry import VALVE_OF_MUSCLE


# ═════════════════════════════════════════════════════════════════════════════
# Paramètres par défaut — modifiables via constructeur
# ═════════════════════════════════════════════════════════════════════════════
# Watchdog à 60 Hz (~17 ms réactivité). Lit le snapshot partagé, ne fait
# pas de I/O ADC.
DEFAULT_HZ = 60.0
DEFAULT_LOOP_BLOCKED_TIMEOUT_S = 1
DEFAULT_GUI_BLOCKED_TIMEOUT_S = 2.0
DEFAULT_MUSCLE_MIN_LENGTH_M = 0.76                # κ_max=0.25 → l_min=0.75 m
DEFAULT_PRESSURE_MAX_BAR = P_MAX
DEFAULT_PRESSURE_MEASURED_MARGIN_BAR = 0.5        # tolérance bruit capteur

# Multi-violation contraction → coupure totale au lieu d'individuelle.
MULTI_VIOLATION_THRESHOLD = 2


# ═════════════════════════════════════════════════════════════════════════════
# SafetyWatchdog
# ═════════════════════════════════════════════════════════════════════════════
class SafetyWatchdog(threading.Thread):
    """Thread indépendant : surveille et coupe les vannes en cas de
    condition critique.
    """

    def __init__(
        self,
        state: RobotState,
        bus_adc: smbus2.SMBus,
        bus_lock: Optional[threading.Lock] = None,
        *,
        hz: float = DEFAULT_HZ,
        loop_blocked_timeout_s: float = DEFAULT_LOOP_BLOCKED_TIMEOUT_S,
        gui_blocked_timeout_s: float = DEFAULT_GUI_BLOCKED_TIMEOUT_S,
        muscle_min_length_m: float = DEFAULT_MUSCLE_MIN_LENGTH_M,
        pressure_max_bar: float = DEFAULT_PRESSURE_MAX_BAR,
        pressure_measured_margin_bar: float = DEFAULT_PRESSURE_MEASURED_MARGIN_BAR,
        on_event: Optional[Callable[[SafetyEvent], None]] = None,
    ):
        super().__init__(name='SafetyWatchdog', daemon=True)
        self.state = state
        self.bus_adc = bus_adc
        self.bus_lock = bus_lock
        self.on_event = on_event

        self.period_s = 1.0 / hz
        self.loop_blocked_timeout_s = loop_blocked_timeout_s
        self.gui_blocked_timeout_s = gui_blocked_timeout_s
        self.muscle_min_length_m = muscle_min_length_m
        self.pressure_max_bar = pressure_max_bar
        self.pressure_measured_margin_bar = pressure_measured_margin_bar

        # Évènement signalant l'arrêt du thread. wait() permet une réaction
        # plus rapide que time.sleep() lors de l'arrêt.
        self._stop_event = threading.Event()

        # Compteur de ticks pour debug / monitoring
        self._tick_count = 0
        self._last_tick_duration_s = 0.0

        # On évite de spammer les events identiques : si la même condition
        # (même type, même muscle) est déjà détectée depuis le tick précédent,
        # on ne réémet pas l'event — on attend qu'elle disparaisse puis
        # réapparaisse. Sinon on remplit le buffer en quelques secondes.
        self._active_violations: set = set()

    # ── API publique ──────────────────────────────────────────────────────────
    def stop(self) -> None:
        """Demande l'arrêt du thread (le tick en cours se termine d'abord)."""
        self._stop_event.set()

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # ── Boucle interne ────────────────────────────────────────────────────────
    def run(self) -> None:
        """Boucle principale du watchdog. Tourne jusqu'à ce que stop() soit
        appelé. Toute exception est attrapée et loggée — le watchdog ne meurt
        JAMAIS silencieusement (sinon on perd le filet de sécurité)."""
        print(f"[SAFETY] Watchdog démarré @ {1.0 / self.period_s:.0f} Hz")
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._tick(t0)
            except Exception as e:
                print(f"[SAFETY] Exception dans _tick : {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # On continue : il vaut mieux un watchdog qui rate un tick
                # que pas de watchdog du tout.

            self._tick_count += 1
            self._last_tick_duration_s = time.perf_counter() - t0

            # wait(timeout) au lieu de sleep : permet l'arrêt immédiat
            sleep_t = self.period_s - self._last_tick_duration_s
            if sleep_t > 0:
                self._stop_event.wait(sleep_t)
        print(f"[SAFETY] Watchdog arrêté après {self._tick_count} ticks")

    def _tick(self, now: float) -> None:
        """Un cycle de surveillance."""
        snap = self.state.snapshot()

        # 1. Estop déjà demandé → coupure totale, rien d'autre à vérifier.
        if snap['estop_requested']:
            self._cut_all_valves('estop_requested')
            return

        # 2. Main loop figé. Ignoré tant que heartbeat = 0 (avant 1er tick).
        if snap['heartbeat_main_loop'] > 0.0:
            age = now - snap['heartbeat_main_loop']
            if age > self.loop_blocked_timeout_s:
                self._emit_event(SafetyEvent(
                    timestamp=now, kind='loop_blocked',
                    muscle=None, value=age,
                    threshold=self.loop_blocked_timeout_s,
                    message=f"Main loop figé depuis {age * 1000:.0f} ms",
                ))
                self._cut_all_valves('loop_blocked')
                self.state.request_estop(source='watchdog/loop_blocked')
                return

        # 3. Pressions commandées > P_MAX (bug — set_muscle_pressure clamp déjà).
        for m, p_cmd in enumerate(snap['pressures_commanded_bar']):
            if p_cmd > self.pressure_max_bar:
                self._emit_event(SafetyEvent(
                    timestamp=now, kind='over_pressure_cmd',
                    muscle=m, value=p_cmd,
                    threshold=self.pressure_max_bar,
                    message=(f"Pression COMMANDÉE muscle {m} = "
                             f"{p_cmd:.2f} bar > {self.pressure_max_bar} bar"),
                ))
                self._cut_one_valve(m, 'over_pressure_cmd')

        # 4. Pression mesurée > P_MAX + marge : vanne bloquée, fuite, capteur
        #    à recalibrer. Coupe le muscle, l'opérateur investigue.
        p_meas_max = self.pressure_max_bar + self.pressure_measured_margin_bar
        for m, p_meas in enumerate(snap['pressures_measured_bar']):
            if p_meas > p_meas_max:
                self._emit_event(SafetyEvent(
                    timestamp=now, kind='over_pressure_meas',
                    muscle=m, value=p_meas,
                    threshold=p_meas_max,
                    message=(f"Pression MESURÉE muscle {m} = "
                             f"{p_meas:.2f} bar > {p_meas_max:.2f} bar"),
                ))
                self._cut_one_valve(m, 'over_pressure_meas')

        # 5. Sur-contraction (κ > 0.25 ⇔ l < 0.75 m). Skip avant calibration
        #    (offsets nuls → fausses alarmes).
        if snap['calibration_done'] and not snap['is_calibrating']:
            over_contracted = []
            for m, L in enumerate(snap['positions_m']):
                if L < self.muscle_min_length_m:
                    over_contracted.append((m, L))

            if len(over_contracted) >= MULTI_VIOLATION_THRESHOLD:
                # Multi-violation : configuration suspecte → coupure totale + estop.
                muscles_str = ', '.join(f"M{m}" for m, _ in over_contracted)
                self._emit_event(SafetyEvent(
                    timestamp=now, kind='over_contraction',
                    muscle=None,
                    value=float(len(over_contracted)),
                    threshold=float(MULTI_VIOLATION_THRESHOLD),
                    message=(f"{len(over_contracted)} muscles sur-contractés "
                             f"({muscles_str}) → coupure totale"),
                ))
                self._cut_all_valves('multi_over_contraction')
                self.state.request_estop(
                    source=f'watchdog/multi_contraction({muscles_str})')
            else:
                # Mono-violation : on relâche juste le muscle, les autres
                # soutiennent la plateforme.
                for m, L in over_contracted:
                    self._emit_event(SafetyEvent(
                        timestamp=now, kind='over_contraction',
                        muscle=m, value=L,
                        threshold=self.muscle_min_length_m,
                        message=(f"Muscle {m} sur-contracté : "
                                 f"L = {L * 1000:.0f} mm < "
                                 f"{self.muscle_min_length_m * 1000:.0f} mm"),
                    ))
                    self._cut_one_valve(m, 'over_contraction')

        # Cleanup : retire les violations résolues pour autoriser ré-émission.
        self._cleanup_resolved_violations(snap, now)

    # ── Helpers internes ──────────────────────────────────────────────────────
    def _cut_one_valve(self, muscle: int, reason: str) -> None:
        """Coupe la vanne d'un muscle spécifique. Le muscle conserve sa
        commande à 0 jusqu'à ce que la condition critique disparaisse."""
        hw_valve = VALVE_OF_MUSCLE[muscle]
        try:
            if self.bus_lock is not None:
                with self.bus_lock:
                    set_muscle_pressure(self.bus_adc, hw_valve, 0.0)
            else:
                set_muscle_pressure(self.bus_adc, hw_valve, 0.0)
            # On reflète l'action dans l'état partagé pour que la GUI le voie
            with self.state.lock:
                self.state.pressures_commanded_bar[muscle] = 0.0
        except Exception as e:
            print(f"[SAFETY] Échec coupure muscle {muscle} ({reason}) : {e}",
                  file=sys.stderr)

    def _cut_all_valves(self, reason: str) -> None:
        """Coupure totale : clear_all + écriture individuelle (résistance
        croisée aux glitchs I2C)."""
        try:
            if self.bus_lock is not None:
                with self.bus_lock:
                    clear_all(self.bus_adc)
            else:
                clear_all(self.bus_adc)
        except Exception as e:
            print(f"[SAFETY] clear_all() raté ({reason}) : {e}",
                  file=sys.stderr)

        for m in range(6):
            self._cut_one_valve(m, reason)

    def _emit_event(self, event: SafetyEvent) -> None:
        """Enregistre + callback. Dédupliqué sur (kind, muscle) tant que la
        condition persiste (évite le spam du ring-buffer)."""
        key = (event.kind, event.muscle)
        if key in self._active_violations:
            return  # déjà signalé, on attend la résolution

        self._active_violations.add(key)
        with self.state.lock:
            self.state.record_safety_event(event)
        # Affichage console — utile au dev même sans GUI
        print(f"[SAFETY] {event.message}", file=sys.stderr)

        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as e:
                print(f"[SAFETY] on_event callback raté : {e}", file=sys.stderr)

    def _cleanup_resolved_violations(self, snap: dict, now: float) -> None:
        """Retire les marqueurs des violations résolues → autorise la
        ré-émission si la condition réapparaît."""
        still_active = set()

        # Estop : actif tant que le flag est True
        if snap['estop_requested']:
            still_active.add(('estop', None))

        # Loop blocked : actif tant que le heartbeat est dépassé
        if snap['heartbeat_main_loop'] > 0.0:
            age = now - snap['heartbeat_main_loop']
            if age > self.loop_blocked_timeout_s:
                still_active.add(('loop_blocked', None))

        # Pressions commandées
        for m, p in enumerate(snap['pressures_commanded_bar']):
            if p > self.pressure_max_bar:
                still_active.add(('over_pressure_cmd', m))

        # Pressions mesurées
        p_meas_max = self.pressure_max_bar + self.pressure_measured_margin_bar
        for m, p in enumerate(snap['pressures_measured_bar']):
            if p > p_meas_max:
                still_active.add(('over_pressure_meas', m))

        # Sur-contractions
        if snap['calibration_done'] and not snap['is_calibrating']:
            over = [m for m, L in enumerate(snap['positions_m'])
                    if L < self.muscle_min_length_m]
            if len(over) >= MULTI_VIOLATION_THRESHOLD:
                still_active.add(('over_contraction', None))
            for m in over:
                still_active.add(('over_contraction', m))

        # On garde uniquement les violations qui sont toujours actives
        self._active_violations = self._active_violations & still_active


