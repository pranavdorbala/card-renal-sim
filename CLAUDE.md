# Cardiorenal Coupling Model

## Project Overview
Coupled heart-kidney simulation modeling cardiorenal syndrome (CRS). Combines a cardiac model with Hallow et al. 2017 renal physiology module via bidirectional message passing.

## Key Files
- `cardiorenal_coupling.py` — Original CircAdapt-based coupled simulation (HFrEF + CKD, 3 scenarios, static plots)
- `dashboard.py` — Interactive Flask + Plotly.js dashboard for HFpEF + CKD scenarios (standalone, no CircAdapt dependency)
- `P_ref_VanOsta2024.npy` — CircAdapt reference data
- `app.py` — Flask app using CircAdapt (generated, depends on circadapt package)

## Running the Dashboard
```bash
pip install flask plotly numpy
python dashboard.py
# Open http://localhost:8050
```

## Architecture
- **Heart model** (dashboard.py): Time-varying elastance with exponential EDPVR. Key HFpEF parameter: `stiffness_scale` (1.0 = normal, >1 = diastolic dysfunction)
- **Kidney model**: Hallow et al. 2017 — glomerular hemodynamics, TGF, RAAS, tubular handling, volume balance. Key CKD parameter: `Kf_scale` (1.0 = normal, <1 = nephron loss)
- **Coupling**: Bidirectional message passing. Heart sends MAP/CO/CVP to kidney; kidney sends V_blood/SVR back. `coupling_intensity` parameter (0=independent, 1=normal, 2=amplified)

## Calibration Notes
- Renal resistances: R_preAA=12, R_AA0=26, R_EA0=43 (calibrated for standalone model, different from CircAdapt-tuned values)
- Kf=8.0 (calibrated to give baseline GFR ~120 mL/min)
- Heart: E_max=2.3, beta=0.028, A_ed=0.5, R_sys=0.018 mmHg·min/mL

## Scenarios (all HFpEF + CKD focused)
- Type 1 CRS: Acute HFpEF decompensation → AKI
- Type 2 CRS: Chronic HFpEF → progressive CKD
- Type 4 CRS: Primary CKD → secondary HFpEF
- Combined: Simultaneous deterioration
- Isolated HFpEF: No primary CKD
