# Filtrage des longueurs capteurs

Le filtrage IMU pour reconstruire la pose a été abandonné : la cinématique
directe (FK) sur les longueurs muscles est plus précise et n'accumule pas
de dérive. Le seul filtrage actif est sur les longueurs des potentiomètres.

## Pipeline

`position_filters.py` expose `PositionFilterBank`, instancié dans
`control_loop._run` : un IIR passe-bas 1er ordre par muscle (fc paramétrable).

Activé depuis la GUI via `ControlConfig.position_filter_enabled`. Défauts
dans `robot_state.ControlConfig` :

```python
position_filter_enabled: True
position_filter_fc_hz:   6.0    # 0 = IIR off
```

fc = 6 Hz aligné avec le filtre D du PID → cascade homogène (~2e ordre
Butterworth) sans dégrader la marge de phase à 3 Hz.

## Caractérisation

`diagnose_position_noise.py` logue à la cadence native de l'ADC, plateforme
immobile, puis calcule par muscle :
- sigma (bruit RMS) brut et en mètres
- pic-à-pic, crête 3σ / 5σ (outliers)
