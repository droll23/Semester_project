# Plateforme Stewart — Contrôleur Python

Contrôle d'une plateforme Stewart 6-DDL actionnée par 6 muscles pneumatiques
Festo DMSP (projet Aeropoly, simulateur de vol Pilatus PC-12NG).

## Matériel piloté

- 6 capteurs de longueur (potentiomètres à fil) via 2 × ADC128D818 (I²C)
- 6 vannes proportionnelles ITV via DAC5578 (I²C)
- 1 IMU LSM9DS1 (I²C) — loggué pour diagnostic, plus utilisé pour la pose
- 6 capteurs de pression (monitoring vannes)

## Stratégie de contrôle

Approche feedforward d'abord, feedback ensuite :

1. Inverse kinematics : pose cartésienne cible → 6 longueurs muscle cibles
2. Équilibre statique de la plateforme (poids + charge) → tensions cibles
3. Inversion du modèle Sarosi (P, ε) → pressions cibles
4. Pressions envoyées directement aux vannes (pas de PID interne sur la pression)

La cinématique directe (Levenberg-Marquardt) reconstruit la pose réelle à
partir des longueurs mesurées, utilisée pour monitoring et diagnostic.

Un PID longueur (FF + PID sur erreur de longueur muscle) est activable depuis
la GUI, avec deux domaines de retour sélectionnables (A/B) :

- `pressure` : PID → correction bar ajoutée au feedforward (historique)
- `force` : PID → ΔF (N), F_total inversé par Sarosi (Gattringer 2009)


## Architecture

```
robot_control_gui.py       Entry point — Tk GUI + orchestration
├── gui_panels.py          Mixin : panneaux Tkinter
├── ui_theme.py            Couleurs, fonts
├── control_loop.py        Boucle 13 Hz : FF + PID + I/O
│   ├── stewart_feedforward.py    Modèle Sarosi + équilibre statique
│   ├── stewart_forward_kinematics.py   Levenberg-Marquardt FK
│   ├── geometry.py        Points d'attache base/plateforme, IK
│   ├── pid_controller.py  MuscleLengthPID + MuscleLengthForcePID
│   ├── workspace.py       Clamp de pose dans l'enveloppe atteignable
│   ├── auto_sequence.py   Séquence de waypoints pour caractérisation
│   ├── calibration.py     Offsets capteurs et IMU au repos
│   ├── position_filters.py   IIR/médian sur les longueurs mesurées
│   └── platform_pose.py   Vec3, IMUSnap (logging)
├── hardware.py            Init des bus I²C
├── dac.py                 Pilotage vannes via DAC5578
├── position_adc.py        Lecture potentiomètres via ADC128D818
├── pressure_adc.py        Lecture pressions vannes
├── imu.py                 Lecture IMU LSM9DS1
├── robot_state.py         État partagé thread-safe (RobotState, ControlConfig)
├── safety_watchdog.py     Surveillance hors-bande + E-stop
├── data_logger.py         Logger CSV non-bloquant (thread dédié)
└── diagnose_position_noise.py   Outil hors-ligne : caractérisation du bruit
```

## Lancement

```bash
python3 robot_control_gui.py
```

## Calibration

1. Plateforme immobile en pose de repos
2. Cliquer « 🎯 Calibrer » dans la GUI
3. Attendre 3 secondes
4. Les offsets capteurs (et IMU) sont enregistrés

## Logging

Le `LogWriter` écrit un CSV thread-safe (écriture déléguée à un thread dédié,
non bloquant pour la control loop). Contenu par muscle :

- longueurs cibles (IK) + mesurées (capteurs) — m
- pressions cibles (vanne) + mesurées (ADC) — bar
- tensions feedforward `target_F_M*_N` (sortie équilibre statique) — N
- κ feedforward, drapeaux saturation / tension négative
- erreur PID longueur, termes P / I / D
- sortie PID en bar (domaine `pressure`) et sortie ΔF en N (domaine `force`)
- force totale commandée `cmd_F_M*_N = T_ff + ΔF` — N

Plus, en colonnes scalaires : timestamp, pose cible et pose FK (3 + 3),
IMU brut (accel, gyro), drapeaux ff_enabled / pid_enabled / calibrating,
et le domaine de retour actif.

Métadonnées (masse, géométrie, gains, cadence) en lignes commentées `# ...`
en tête du fichier
