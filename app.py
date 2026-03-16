#!/usr/bin/env python3
"""
CircAdapt Cardiorenal Coupling — Flask Web Application
=======================================================

Interactive web interface for the coupled CircAdapt VanOsta2024 heart model
and Hallow et al. renal physiology module.

Uses the *published* CircAdapt Python package (pip install circadapt) which
wraps a validated, peer-reviewed C++ finite-element cardiovascular simulator.

WHY A STANDALONE HEART MODEL?
------------------------------
CircAdapt is not a toy or a simplified lumped-parameter script — it is a
full biophysical model of the heart and circulation published by the
Maastricht University group (Arts, Lumens, Delhaas, VanOsta et al.).

Using the published CircAdapt module rather than writing our own cardiac
equations gives us several critical advantages:

1. VALIDATED PHYSIOLOGY — The VanOsta2024 model has been calibrated against
   clinical hemodynamic data across healthy and pathological states.  Its
   sarcomere mechanics (the one-fiber model), TriSeg ventricular interaction,
   and circulation modules have been peer-reviewed and used in dozens of
   publications.  Reimplementing these equations from scratch would introduce
   untested numerical choices and lose years of validation work.

2. MULTI-SCALE FIDELITY — CircAdapt couples sarcomere-level mechanics
   (active fiber stress, passive stiffness, contractile duration) with
   organ-level hemodynamics (PV loops, valve dynamics, arteriovenous
   coupling).  A hand-rolled time-varying elastance model (e.g. a simple
   E(t) sinusoid) cannot reproduce the load-dependent Frank-Starling
   behavior, interventricular septal interaction, or realistic pressure
   waveform morphology that CircAdapt provides.

3. PARAMETER INTERPRETABILITY — CircAdapt parameters map to measurable
   physiological quantities: active fiber stress (Sf_act), passive
   stiffness (k1), wall volume, arterial compliance.  This means
   the "stiffness knob" (k1_scale) corresponds to a real biophysical
   mechanism (increased passive ventricular stiffness in HFpEF), not
   an arbitrary gain.

4. REPRODUCIBILITY — By depending on the published pip package (CircAdapt
   v2602+), anyone can install, run, and reproduce our results without
   needing custom C code or Matlab.  The model state can be exported,
   saved, and loaded deterministically.

5. MODULARITY — The coupling architecture (heart <-> kidney message
   passing) treats CircAdapt as a black-box hemodynamic server.  The
   renal module (Hallow et al. 2017) only needs MAP, CO, and CVP from
   the heart, and returns blood volume + SVR adjustments.  This clean
   interface means either module can be swapped independently — e.g.
   replacing Hallow with a more detailed nephron model, or replacing
   VanOsta2024 with a future CircAdapt release — without rewriting the
   coupling logic.

In short: the standalone heart model is the *right* level of abstraction.
It gives physiologically faithful cardiac output without burdening this
project with maintaining and validating a cardiac simulator from scratch.

References:
    - CircAdapt: https://framework.circadapt.org  (VanOsta et al. 2024)
    - Renal model: Hallow & Gebremichael, CPT:PSP 6:383-392, 2017
    - Coupling: Basu et al., PLoS Comput Biol 19(11):e1011598, 2023

Usage:
    pip install circadapt flask plotly numpy
    python app.py
    # Open http://127.0.0.1:5000 in a browser
"""

import json
import io
import base64
import traceback
from dataclasses import asdict

import math
import numpy as np
from flask import Flask, render_template_string, request, jsonify

# ── Import the coupling module (lives alongside this file) ──────────────
from cardiorenal_coupling import (
    CircAdaptHeartModel,
    HallowRenalModel,
    InflammatoryState,
    update_inflammatory_state,
    update_renal_model,
    heart_to_kidney,
    kidney_to_heart,
    PA_TO_MMHG, MMHG_TO_PA, M3_TO_ML, ML_TO_M3,
)

app = Flask(__name__)


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None so JSON serialization succeeds."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


# =========================================================================
# SIMULATION BACKEND
# =========================================================================

def run_simulation(n_steps, dt_hours, cardiac_schedule, kidney_schedule,
                   stiffness_schedule=None,
                   inflammation_schedule=None, diabetes_schedule=None):
    """
    Run the coupled cardiorenal simulation and return JSON-serialisable
    results for the frontend.
    """
    if stiffness_schedule is None:
        stiffness_schedule = [1.0] * n_steps
    if inflammation_schedule is None:
        inflammation_schedule = [0.0] * n_steps
    if diabetes_schedule is None:
        diabetes_schedule = [0.0] * n_steps

    heart = CircAdaptHeartModel()
    renal = HallowRenalModel()
    ist = InflammatoryState()
    last_valid_hemo = None

    results = {
        'steps': [],
        'pv_lv': [],      # list of {V: [...], P: [...]}
        'pv_rv': [],
        'sbp': [], 'dbp': [], 'map': [],
        'co': [], 'sv': [], 'ef': [],
        'hr': [], 'edv': [], 'esv': [],
        'v_blood': [], 'gfr': [], 'na_excr': [], 'p_glom': [],
        'sf_scale': [], 'kf_scale': [], 'k1_scale': [],
        'inflammation_scale': [], 'diabetes_scale': [],
        'effective_sf': [], 'effective_kf': [], 'effective_k1': [],
        'pven': [],
        'pressure_waveforms': [],   # {t: [...], p_SyArt: [...], p_LV: [...]}
        'solver_crashed': [],       # True if CircAdapt diverged at this step
    }

    for s in range(n_steps):
        sf = cardiac_schedule[s] if s < len(cardiac_schedule) else cardiac_schedule[-1]
        kf = kidney_schedule[s] if s < len(kidney_schedule) else kidney_schedule[-1]
        k1 = stiffness_schedule[s] if s < len(stiffness_schedule) else stiffness_schedule[-1]
        infl = inflammation_schedule[s] if s < len(inflammation_schedule) else inflammation_schedule[-1]
        diab = diabetes_schedule[s] if s < len(diabetes_schedule) else diabetes_schedule[-1]

        # 0 - Update inflammatory mediator layer
        ist = update_inflammatory_state(ist, infl, diab)

        # 1 - Apply inflammatory modifiers to heart
        heart.apply_inflammatory_modifiers(ist)

        # 2 - Apply stiffness (HFpEF diastolic dysfunction)
        effective_k1 = k1 * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # 3 - Apply deterioration (Sf composed with inflammatory factor)
        effective_sf = max(sf * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)
        renal.Kf_scale = kf
        effective_kf = kf * ist.Kf_factor

        # 4 - Heart to steady state
        hemo = heart.run_to_steady_state()

        # Check for solver crash (NaN in output) — use last valid state
        solver_crashed = math.isnan(hemo['MAP'])
        if solver_crashed and last_valid_hemo is not None:
            hemo = last_valid_hemo
        elif not solver_crashed:
            last_valid_hemo = hemo

        # 5 - Heart -> Kidney
        h2k = heart_to_kidney(hemo)

        # 6 - Kidney update (with inflammatory effects)
        renal = update_renal_model(renal, h2k.MAP, h2k.CO, h2k.Pven,
                                   dt_hours, inflammatory_state=ist)

        # 7 - Kidney -> Heart
        k2h = kidney_to_heart(renal, h2k.MAP, h2k.CO, h2k.Pven)

        # 8 - Apply kidney feedback
        heart.apply_kidney_feedback(
            V_blood_m3=k2h.V_blood * ML_TO_M3,
            SVR_ratio=k2h.SVR_ratio,
        )

        # 9 - Record
        results['steps'].append(s + 1)
        results['pv_lv'].append({
            'V': hemo['V_LV'].tolist(),
            'P': hemo['p_LV'].tolist(),
        })
        results['pv_rv'].append({
            'V': hemo['V_RV'].tolist(),
            'P': hemo['p_RV'].tolist(),
        })
        results['pressure_waveforms'].append({
            't': hemo['t'].tolist(),
            'p_SyArt': hemo['p_SyArt'].tolist(),
            'p_LV': hemo['p_LV'].tolist(),
        })
        results['sbp'].append(round(hemo['SBP'], 2))
        results['dbp'].append(round(hemo['DBP'], 2))
        results['map'].append(round(hemo['MAP'], 2))
        results['co'].append(round(hemo['CO'], 3))
        results['sv'].append(round(hemo['SV'], 2))
        results['ef'].append(round(hemo['EF'], 2))
        results['hr'].append(round(hemo['HR'], 1))
        results['edv'].append(round(hemo['EDV'], 2))
        results['esv'].append(round(hemo['ESV'], 2))
        results['v_blood'].append(round(renal.V_blood, 1))
        results['gfr'].append(round(renal.GFR, 2))
        results['na_excr'].append(round(renal.Na_excretion, 1))
        results['p_glom'].append(round(renal.P_glom, 2))
        results['sf_scale'].append(round(sf, 3))
        results['kf_scale'].append(round(kf, 3))
        results['k1_scale'].append(round(k1, 3))
        results['inflammation_scale'].append(round(infl, 3))
        results['diabetes_scale'].append(round(diab, 3))
        results['effective_sf'].append(round(effective_sf, 3))
        results['effective_kf'].append(round(effective_kf, 3))
        results['effective_k1'].append(round(effective_k1, 3))
        results['pven'].append(round(hemo['Pven'], 2))
        results['solver_crashed'].append(solver_crashed)

    return results


# =========================================================================
# ROUTES
# =========================================================================

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/simulate', methods=['POST'])
def simulate():
    """Run a coupled simulation with user-specified parameters."""
    try:
        data = request.get_json()

        n_steps = int(data.get('n_steps', 8))
        n_steps = max(2, min(n_steps, 20))

        dt_hours = float(data.get('dt_hours', 6.0))
        dt_hours = max(1.0, min(dt_hours, 48.0))

        scenario = data.get('scenario', 'custom')

        # Pre-built scenarios — all HFpEF-focused
        # Defaults: Sf_act near 1.0, k1 drives diastolic dysfunction
        inflammation_schedule = [0.0] * n_steps
        diabetes_schedule = [0.0] * n_steps
        stiffness_schedule = [1.0] * n_steps

        if scenario == 'hfpef':
            # Isolated HFpEF: progressive diastolic dysfunction
            cardiac_schedule = [1.0] * n_steps
            kidney_schedule = [1.0] * n_steps
            stiffness_schedule = np.linspace(1.0, 2.5, n_steps).tolist()
        elif scenario == 'ckd':
            # Progressive CKD (kidney only)
            cardiac_schedule = [1.0] * n_steps
            kidney_schedule = np.linspace(1.0, 0.30, n_steps).tolist()
        elif scenario == 'hfpef_ckd':
            # HFpEF + CKD (Type 2 CRS)
            cardiac_schedule = [0.95] * n_steps
            kidney_schedule = np.linspace(1.0, 0.40, n_steps).tolist()
            stiffness_schedule = np.linspace(1.0, 2.0, n_steps).tolist()
        elif scenario == 'ckd_hfpef':
            # CKD → HFpEF (Type 4 CRS): kidney first, heart follows
            cardiac_schedule = [1.0] * n_steps
            kidney_schedule = np.linspace(1.0, 0.35, n_steps).tolist()
            # k1 stays at 1.0 for first half, then ramps to 1.8
            half = n_steps // 2
            stiffness_schedule = ([1.0] * half +
                np.linspace(1.0, 1.8, n_steps - half).tolist())

        # ── Inflammatory / metabolic HFpEF scenarios ───────────────
        elif scenario == 'diabetic_hfpef':
            # Diabetic HFpEF: AGEs drive stiffness + nephropathy
            cardiac_schedule = [1.0] * n_steps
            kidney_schedule = np.linspace(1.0, 0.50, n_steps).tolist()
            stiffness_schedule = np.linspace(1.0, 1.5, n_steps).tolist()
            inflammation_schedule = np.linspace(0.0, 0.3, n_steps).tolist()
            diabetes_schedule = np.linspace(0.2, 0.85, n_steps).tolist()
        elif scenario == 'inflammatory_hfpef':
            # Inflammatory HFpEF: fibrosis-driven diastolic dysfunction
            cardiac_schedule = [0.9] * n_steps
            kidney_schedule = np.linspace(1.0, 0.50, n_steps).tolist()
            stiffness_schedule = np.linspace(1.0, 2.0, n_steps).tolist()
            inflammation_schedule = np.linspace(0.1, 0.8, n_steps).tolist()
        elif scenario == 'sepsis_aki':
            # Septic shock with transient recovery (bell curve inflammation)
            t_norm = np.linspace(0, 1, n_steps)
            infl_curve = np.exp(-((t_norm - 0.35) / 0.2)**2)
            inflammation_schedule = (infl_curve * 0.95).tolist()
            cardiac_schedule = (1.0 - 0.30 * infl_curve).tolist()
            kidney_schedule = (1.0 - 0.25 * infl_curve).tolist()

        elif scenario == 'custom':
            cardiac_schedule = data.get('cardiac_schedule', [1.0] * n_steps)
            kidney_schedule = data.get('kidney_schedule', [1.0] * n_steps)
            stiffness_schedule = data.get('stiffness_schedule', [1.0] * n_steps)
            inflammation_schedule = data.get('inflammation_schedule', [0.0] * n_steps)
            diabetes_schedule = data.get('diabetes_schedule', [0.0] * n_steps)
            cardiac_schedule = [float(x) for x in cardiac_schedule]
            kidney_schedule = [float(x) for x in kidney_schedule]
            stiffness_schedule = [float(x) for x in stiffness_schedule]
            inflammation_schedule = [float(x) for x in inflammation_schedule]
            diabetes_schedule = [float(x) for x in diabetes_schedule]
        else:
            cardiac_schedule = [1.0] * n_steps
            kidney_schedule = [1.0] * n_steps

        # ── Layer modifiers on top of scenario if requested ──────────
        layer = data.get('layer_modifiers', False)
        if layer and scenario != 'custom':
            infl_val = float(data.get('inflammation_value', 0.0))
            diab_val = float(data.get('diabetes_value', 0.0))
            if infl_val > 0:
                inflammation_schedule = [
                    min(x + infl_val, 1.0) for x in inflammation_schedule
                ]
            if diab_val > 0:
                diabetes_schedule = [
                    min(x + diab_val, 1.0) for x in diabetes_schedule
                ]

        results = run_simulation(n_steps, dt_hours, cardiac_schedule,
                                 kidney_schedule, stiffness_schedule,
                                 inflammation_schedule, diabetes_schedule)
        return jsonify({'status': 'ok', 'data': sanitize_for_json(results)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/single_beat', methods=['POST'])
def single_beat():
    """Run a single CircAdapt beat with given parameters and return waveforms.

    Accepts: sf_scale, stiffness_scale, V_blood, SVR_ratio,
             inflammation_scale, diabetes_scale
    """
    try:
        data = request.get_json()
        sf_scale = float(data.get('sf_scale', 1.0))
        sf_scale = max(0.1, min(sf_scale, 1.5))
        stiffness_scale = float(data.get('stiffness_scale', 1.0))
        stiffness_scale = max(1.0, min(stiffness_scale, 3.5))
        V_blood = float(data.get('V_blood', 5000.0))
        V_blood = max(3000.0, min(V_blood, 8000.0))
        SVR_ratio = float(data.get('SVR_ratio', 1.0))
        SVR_ratio = max(0.5, min(SVR_ratio, 2.0))
        infl_scale = float(data.get('inflammation_scale', 0.0))
        diab_scale = float(data.get('diabetes_scale', 0.0))

        heart = CircAdaptHeartModel()

        # Apply inflammatory mediator layer
        ist = InflammatoryState()
        ist = update_inflammatory_state(ist, infl_scale, diab_scale)
        heart.apply_inflammatory_modifiers(ist)

        # Apply stiffness (composed with inflammatory k1 factor)
        effective_k1 = stiffness_scale * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # Apply contractility
        effective_sf = max(sf_scale * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)

        # Apply kidney feedback (volume + SVR)
        heart.apply_kidney_feedback(
            V_blood_m3=V_blood * ML_TO_M3,
            SVR_ratio=SVR_ratio * ist.p0_factor,
        )

        hemo = heart.run_to_steady_state()

        result = {
            'pv_lv': {'V': hemo['V_LV'].tolist(), 'P': hemo['p_LV'].tolist()},
            'pv_rv': {'V': hemo['V_RV'].tolist(), 'P': hemo['p_RV'].tolist()},
            'waveform': {
                't': hemo['t'].tolist(),
                'p_SyArt': hemo['p_SyArt'].tolist(),
                'p_LV': hemo['p_LV'].tolist(),
            },
            'MAP': round(hemo['MAP'], 2),
            'SBP': round(hemo['SBP'], 2),
            'DBP': round(hemo['DBP'], 2),
            'CO': round(hemo['CO'], 3),
            'SV': round(hemo['SV'], 2),
            'EF': round(hemo['EF'], 2),
            'HR': round(hemo['HR'], 1),
            'EDV': round(hemo['EDV'], 2),
            'ESV': round(hemo['ESV'], 2),
            'Pven': round(hemo['Pven'], 2),
            'V_blood_total': round(hemo['V_blood_total'], 1),
        }
        return jsonify({'status': 'ok', 'data': sanitize_for_json(result)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/kidney_step', methods=['POST'])
def kidney_step():
    """Run a single kidney evaluation given heart outputs.

    Accepts: MAP, CO, Pven, Kf_scale, inflammation_scale, diabetes_scale
    Returns: GFR, RBF, P_glom, Na_excretion, V_blood, SVR_ratio, etc.
    """
    try:
        data = request.get_json()
        MAP = float(data.get('MAP', 93.0))
        MAP = max(40.0, min(MAP, 200.0))
        CO = float(data.get('CO', 5.0))
        CO = max(0.5, min(CO, 10.0))
        Pven = float(data.get('Pven', 3.0))
        Pven = max(0.0, min(Pven, 30.0))
        Kf_scale = float(data.get('Kf_scale', 1.0))
        Kf_scale = max(0.05, min(Kf_scale, 1.0))
        infl_scale = float(data.get('inflammation_scale', 0.0))
        diab_scale = float(data.get('diabetes_scale', 0.0))

        # Fresh renal model (stateless "what-if")
        renal = HallowRenalModel()
        renal.Kf_scale = Kf_scale

        # Inflammatory mediator effects
        ist = InflammatoryState()
        ist = update_inflammatory_state(ist, infl_scale, diab_scale)

        V_blood_before = renal.V_blood
        renal = update_renal_model(renal, MAP, CO, Pven, dt_hours=6.0,
                                   inflammatory_state=ist)

        # Compute SVR_ratio the same way as kidney_to_heart
        k2h = kidney_to_heart(renal, MAP, CO, Pven)

        result = {
            'GFR': round(renal.GFR, 2),
            'RBF': round(renal.RBF, 1),
            'P_glom': round(renal.P_glom, 2),
            'Na_excretion': round(renal.Na_excretion, 1),
            'water_excretion': round(renal.water_excretion, 2),
            'V_blood': round(renal.V_blood, 1),
            'V_blood_change': round(renal.V_blood - V_blood_before, 1),
            'SVR_ratio': round(k2h.SVR_ratio, 3),
            'C_Na': round(renal.C_Na, 1),
        }
        return jsonify({'status': 'ok', 'data': sanitize_for_json(result)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =========================================================================
# HTML / JS FRONTEND  (single-file, uses Plotly.js from CDN)
# =========================================================================

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CircAdapt Cardiorenal Coupling Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root {
  --bg:      #0b0b14;
  --surface: #12122a;
  --border:  #1e1e3a;
  --text:    #d0d0e0;
  --accent:  #6c8cff;
  --accent2: #4ecdc4;
  --red:     #ff6b6b;
  --gold:    #ffd93d;
  --mint:    #a8e6cf;
  --lilac:   #c9b1ff;
  --peach:   #ffb385;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}
header {
  background: linear-gradient(135deg, #141432 0%, #0e1628 100%);
  border-bottom: 1px solid var(--border);
  padding: 18px 30px;
  display: flex;
  align-items: center;
  gap: 18px;
}
header h1 {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--accent);
}
header .subtitle {
  font-size: 0.82rem;
  color: #888;
  margin-top: 2px;
}
.container {
  max-width: 1600px;
  margin: 0 auto;
  padding: 20px 24px;
}

/* ── Control Panel ────────────────────────────────────── */
.controls {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.control-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
}
.control-card h3 {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  margin-bottom: 12px;
}
label {
  display: block;
  font-size: 0.8rem;
  color: #999;
  margin-bottom: 4px;
  margin-top: 10px;
}
label:first-of-type { margin-top: 0; }
select, input[type=number], input[type=range] {
  width: 100%;
  padding: 7px 10px;
  background: #0a0a1a;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 0.88rem;
}
input[type=range] {
  padding: 0;
  margin-top: 2px;
  accent-color: var(--accent);
}
.range-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.range-row input[type=range] { flex: 1; }
.range-row .range-val {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--accent2);
  min-width: 40px;
  text-align: right;
}
button {
  display: inline-block;
  margin-top: 14px;
  padding: 10px 22px;
  border: none;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-primary {
  background: var(--accent);
  color: #fff;
}
.btn-primary:hover { background: #5570e6; }
.btn-primary:disabled {
  background: #333;
  cursor: wait;
}
.btn-secondary {
  background: #1e1e3a;
  color: var(--accent2);
  border: 1px solid var(--border);
  margin-left: 8px;
}
.btn-secondary:hover { background: #262650; }

/* ── Status bar ───────────────────────────────────────── */
#status {
  font-size: 0.82rem;
  color: #888;
  margin-bottom: 16px;
  min-height: 1.2em;
}
#status.running { color: var(--gold); }
#status.error { color: var(--red); }

/* ── Metrics cards ────────────────────────────────────── */
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 10px;
  margin-bottom: 20px;
}
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  text-align: center;
}
.metric-card .value {
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--accent2);
}
.metric-card .label {
  font-size: 0.7rem;
  color: #777;
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

/* ── Plot grid ────────────────────────────────────────── */
.plot-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}
.plot-grid.three-col {
  grid-template-columns: 1fr 1fr 1fr;
}
.plot-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px;
  min-height: 340px;
}
.plot-box.wide {
  grid-column: span 2;
}

/* ── Tabs ─────────────────────────────────────────────── */
.tabs {
  display: flex;
  gap: 0;
  margin-bottom: 20px;
  border-bottom: 2px solid var(--border);
}
.tab {
  padding: 10px 24px;
  font-size: 0.9rem;
  font-weight: 600;
  color: #666;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  transition: color 0.15s, border-color 0.15s;
}
.tab:hover { color: var(--text); }
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Info section ─────────────────────────────────────── */
.info-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 24px 28px;
  margin-top: 24px;
  line-height: 1.7;
  font-size: 0.88rem;
}
.info-section h2 {
  color: var(--accent);
  font-size: 1.1rem;
  margin-bottom: 12px;
}
.info-section h3 {
  color: var(--accent2);
  font-size: 0.95rem;
  margin-top: 16px;
  margin-bottom: 6px;
}
.info-section p { margin-bottom: 8px; }
.info-section ul {
  margin-left: 20px;
  margin-bottom: 8px;
}

@media (max-width: 900px) {
  .plot-grid, .plot-grid.three-col {
    grid-template-columns: 1fr;
  }
  .plot-box.wide { grid-column: span 1; }
  .controls { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<header>
  <div>
    <h1>CircAdapt Cardiorenal Coupling Simulator</h1>
    <div class="subtitle">
      Heart: CircAdapt VanOsta2024 &nbsp;|&nbsp;
      Kidney: Hallow et al. 2017 &nbsp;|&nbsp;
      Inflammation: mediator layer &nbsp;|&nbsp;
      Coupled via asynchronous message passing
    </div>
  </div>
</header>

<div class="container">

<!-- ── Tabs ─────────────────────────────────────────── -->
<div class="tabs">
  <div class="tab active" data-tab="coupled">Coupled Simulation</div>
  <div class="tab" data-tab="interactive">Interactive Simulator</div>
  <div class="tab" data-tab="about">About the Model</div>
</div>

<!-- ════════════════════════════════════════════════════ -->
<!-- TAB 1: Coupled Simulation                            -->
<!-- ════════════════════════════════════════════════════ -->
<div class="tab-content active" id="tab-coupled">

<div class="controls">
  <div class="control-card">
    <h3>Scenario</h3>
    <label for="scenario">Pre-built scenario</label>
    <select id="scenario" onchange="onScenarioChange()">
      <optgroup label="HFpEF (Diastolic Dysfunction)">
        <option value="hfpef">Isolated HFpEF (progressive stiffness)</option>
        <option value="hfpef_ckd">HFpEF + CKD (Type 2 CRS)</option>
        <option value="ckd_hfpef">CKD &rarr; HFpEF (Type 4 CRS)</option>
      </optgroup>
      <option value="ckd">Progressive CKD (kidney only)</option>
      <optgroup label="Inflammatory / Metabolic">
        <option value="diabetic_hfpef">Diabetic HFpEF (AGEs + nephropathy)</option>
        <option value="inflammatory_hfpef">Inflammatory HFpEF (fibrosis-driven)</option>
        <option value="sepsis_aki">Septic Cardiorenal (acute)</option>
      </optgroup>
      <option value="custom">Custom schedule</option>
    </select>
  </div>
  <div class="control-card">
    <h3>Simulation Settings</h3>
    <label for="n_steps">Coupling steps</label>
    <input type="number" id="n_steps" value="8" min="2" max="20">
    <label for="dt_hours">Renal dt (hours)</label>
    <input type="number" id="dt_hours" value="6" min="1" max="48" step="1">
  </div>
  <div class="control-card" id="custom-controls" style="display:none;">
    <h3>Custom Schedules</h3>
    <label for="cardiac_csv">Contractility Sf_act_scale (comma-separated)</label>
    <input type="text" id="cardiac_csv" value="1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"
           style="width:100%;padding:7px 10px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;">
    <label for="stiff_csv">Stiffness k1_scale (comma-separated, &ge;1.0)</label>
    <input type="text" id="stiff_csv" value="1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5"
           style="width:100%;padding:7px 10px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;">
    <label for="kidney_csv">Kidney Kf_scale (comma-separated)</label>
    <input type="text" id="kidney_csv" value="1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"
           style="width:100%;padding:7px 10px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;">
    <label for="infl_csv">Inflammation schedule (comma-separated)</label>
    <input type="text" id="infl_csv" value="0, 0, 0, 0, 0, 0, 0, 0"
           style="width:100%;padding:7px 10px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;">
    <label for="diab_csv">Diabetes schedule (comma-separated)</label>
    <input type="text" id="diab_csv" value="0, 0, 0, 0, 0, 0, 0, 0"
           style="width:100%;padding:7px 10px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:0.85rem;">
  </div>
  <div class="control-card" id="modifier-controls">
    <h3>Pathology Modifiers</h3>
    <label>Inflammation (0 = none, 1 = severe)</label>
    <div class="range-row">
      <input type="range" id="infl-slider" min="0" max="1" step="0.05" value="0"
             oninput="document.getElementById('infl-val').textContent=this.value">
      <span class="range-val" id="infl-val">0</span>
    </div>
    <label>Diabetes (0 = none, 1 = severe)</label>
    <div class="range-row">
      <input type="range" id="diab-slider" min="0" max="1" step="0.05" value="0"
             oninput="document.getElementById('diab-val').textContent=this.value">
      <span class="range-val" id="diab-val">0</span>
    </div>
    <label style="margin-top:12px;display:flex;align-items:center;gap:6px;cursor:pointer;">
      <input type="checkbox" id="layer-modifiers" style="width:auto;">
      Layer on top of scenario
    </label>
  </div>
  <div class="control-card" style="display:flex;align-items:flex-end;">
    <div>
      <button class="btn-primary" id="btn-run" onclick="runSimulation()">Run Simulation</button>
      <button class="btn-secondary" onclick="clearResults()">Clear</button>
    </div>
  </div>
</div>

<div id="status"></div>

<!-- Metrics row (hidden until results) -->
<div class="metrics" id="metrics-row" style="display:none;"></div>

<!-- Plots -->
<div class="plot-grid" id="coupled-plots" style="display:none;">
  <div class="plot-box wide" id="plot-pv-lv"></div>
  <div class="plot-box" id="plot-pv-rv"></div>
  <div class="plot-box" id="plot-pressure-waveform"></div>
  <div class="plot-box" id="plot-bp"></div>
  <div class="plot-box" id="plot-co"></div>
  <div class="plot-box" id="plot-sv-ef"></div>
  <div class="plot-box" id="plot-gfr"></div>
  <div class="plot-box" id="plot-vblood"></div>
  <div class="plot-box" id="plot-pglom"></div>
  <div class="plot-box" id="plot-na"></div>
  <div class="plot-box" id="plot-params"></div>
</div>

</div><!-- /tab-coupled -->


<!-- ════════════════════════════════════════════════════ -->
<!-- TAB 2: Interactive Simulator                         -->
<!-- ════════════════════════════════════════════════════ -->
<div class="tab-content" id="tab-interactive">

<p style="font-size:0.85rem;color:#888;margin-bottom:16px;">
  Manually adjust inter-organ messages and parameters to see how each organ responds.
  Use the transfer buttons to close the coupling loop.
</p>

<div style="display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:start;">

  <!-- ── HEART PANEL ──────────────────────────────────── -->
  <div class="control-card" style="min-height:400px;">
    <h3 style="color:var(--red);">Heart Response</h3>
    <p style="font-size:0.75rem;color:#666;margin-bottom:8px;">Inputs from kidney + cardiac parameters</p>

    <label>Blood Volume (mL) <span style="color:#666;">[from kidney]</span></label>
    <div class="range-row">
      <input type="range" id="ix-vblood" min="3000" max="8000" step="50" value="5000"
             oninput="document.getElementById('ix-vblood-val').textContent=this.value">
      <span class="range-val" id="ix-vblood-val">5000</span>
    </div>

    <label>SVR Ratio <span style="color:#666;">[from kidney]</span></label>
    <div class="range-row">
      <input type="range" id="ix-svr" min="0.5" max="2.0" step="0.01" value="1.0"
             oninput="document.getElementById('ix-svr-val').textContent=this.value">
      <span class="range-val" id="ix-svr-val">1.0</span>
    </div>

    <label>Contractility (Sf_act_scale)</label>
    <div class="range-row">
      <input type="range" id="ix-sf" min="0.5" max="1.2" step="0.01" value="1.0"
             oninput="document.getElementById('ix-sf-val').textContent=this.value">
      <span class="range-val" id="ix-sf-val">1.0</span>
    </div>

    <label>Diastolic Stiffness (k1_scale)</label>
    <div class="range-row">
      <input type="range" id="ix-k1" min="1.0" max="3.5" step="0.05" value="1.0"
             oninput="document.getElementById('ix-k1-val').textContent=this.value">
      <span class="range-val" id="ix-k1-val">1.0</span>
    </div>

    <label>Inflammation</label>
    <div class="range-row">
      <input type="range" id="ix-h-infl" min="0" max="1" step="0.05" value="0"
             oninput="document.getElementById('ix-h-infl-val').textContent=this.value">
      <span class="range-val" id="ix-h-infl-val">0</span>
    </div>

    <label>Diabetes</label>
    <div class="range-row">
      <input type="range" id="ix-h-diab" min="0" max="1" step="0.05" value="0"
             oninput="document.getElementById('ix-h-diab-val').textContent=this.value">
      <span class="range-val" id="ix-h-diab-val">0</span>
    </div>

    <button class="btn-primary" id="btn-heart" onclick="runHeartPanel()" style="margin-top:10px;">
      Run Heart Beat
    </button>
    <div id="ix-heart-status" style="font-size:0.8rem;color:#888;margin-top:6px;min-height:1.2em;"></div>

    <!-- Heart outputs -->
    <div id="ix-heart-outputs" style="display:none;margin-top:12px;">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
        <div class="metric-card"><div class="value" id="ix-h-map">--</div><div class="label">MAP</div></div>
        <div class="metric-card"><div class="value" id="ix-h-co">--</div><div class="label">CO</div></div>
        <div class="metric-card"><div class="value" id="ix-h-ef">--</div><div class="label">EF%</div></div>
        <div class="metric-card"><div class="value" id="ix-h-sbp">--</div><div class="label">SBP</div></div>
        <div class="metric-card"><div class="value" id="ix-h-dbp">--</div><div class="label">DBP</div></div>
        <div class="metric-card"><div class="value" id="ix-h-pven">--</div><div class="label">CVP</div></div>
        <div class="metric-card"><div class="value" id="ix-h-sv">--</div><div class="label">SV</div></div>
        <div class="metric-card"><div class="value" id="ix-h-edv">--</div><div class="label">EDV</div></div>
        <div class="metric-card"><div class="value" id="ix-h-esv">--</div><div class="label">ESV</div></div>
      </div>
      <button class="btn-secondary" onclick="transferHeartToKidney()" style="margin-top:8px;width:100%;margin-left:0;">
        Send MAP/CO/CVP to Kidney &rarr;
      </button>
    </div>
  </div>

  <!-- ── MESSAGE ARROWS ───────────────────────────────── -->
  <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;gap:24px;min-width:100px;padding-top:60px;">
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px;text-align:center;font-size:0.72rem;">
      <div style="color:var(--red);font-weight:600;">Heart &rarr; Kidney</div>
      <div style="color:#888;margin-top:4px;" id="ix-msg-h2k">MAP: -- | CO: -- | CVP: --</div>
    </div>
    <div style="font-size:1.5rem;color:#444;">&harr;</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px;text-align:center;font-size:0.72rem;">
      <div style="color:var(--accent2);font-weight:600;">Kidney &rarr; Heart</div>
      <div style="color:#888;margin-top:4px;" id="ix-msg-k2h">V_blood: -- | SVR: --</div>
    </div>
  </div>

  <!-- ── KIDNEY PANEL ─────────────────────────────────── -->
  <div class="control-card" style="min-height:400px;">
    <h3 style="color:var(--accent2);">Kidney Response</h3>
    <p style="font-size:0.75rem;color:#666;margin-bottom:8px;">Inputs from heart + renal parameters</p>

    <label>MAP (mmHg) <span style="color:#666;">[from heart]</span></label>
    <div class="range-row">
      <input type="range" id="ix-map" min="40" max="180" step="1" value="93"
             oninput="document.getElementById('ix-map-val').textContent=this.value; runKidneyPanel();">
      <span class="range-val" id="ix-map-val">93</span>
    </div>

    <label>CO (L/min) <span style="color:#666;">[from heart]</span></label>
    <div class="range-row">
      <input type="range" id="ix-co" min="0.5" max="10" step="0.1" value="5.0"
             oninput="document.getElementById('ix-co-val').textContent=this.value; runKidneyPanel();">
      <span class="range-val" id="ix-co-val">5.0</span>
    </div>

    <label>CVP (mmHg) <span style="color:#666;">[from heart]</span></label>
    <div class="range-row">
      <input type="range" id="ix-pven" min="0" max="30" step="0.5" value="3.0"
             oninput="document.getElementById('ix-pven-val').textContent=this.value; runKidneyPanel();">
      <span class="range-val" id="ix-pven-val">3.0</span>
    </div>

    <label>Kf Scale (CKD severity)</label>
    <div class="range-row">
      <input type="range" id="ix-kf" min="0.05" max="1.0" step="0.01" value="1.0"
             oninput="document.getElementById('ix-kf-val').textContent=this.value; runKidneyPanel();">
      <span class="range-val" id="ix-kf-val">1.0</span>
    </div>

    <label>Inflammation</label>
    <div class="range-row">
      <input type="range" id="ix-k-infl" min="0" max="1" step="0.05" value="0"
             oninput="document.getElementById('ix-k-infl-val').textContent=this.value; runKidneyPanel();">
      <span class="range-val" id="ix-k-infl-val">0</span>
    </div>

    <label>Diabetes</label>
    <div class="range-row">
      <input type="range" id="ix-k-diab" min="0" max="1" step="0.05" value="0"
             oninput="document.getElementById('ix-k-diab-val').textContent=this.value; runKidneyPanel();">
      <span class="range-val" id="ix-k-diab-val">0</span>
    </div>

    <div id="ix-kidney-status" style="font-size:0.8rem;color:#888;margin-top:10px;min-height:1.2em;"></div>

    <!-- Kidney outputs -->
    <div id="ix-kidney-outputs" style="margin-top:12px;">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
        <div class="metric-card"><div class="value" id="ix-k-gfr">--</div><div class="label">GFR</div></div>
        <div class="metric-card"><div class="value" id="ix-k-rbf">--</div><div class="label">RBF</div></div>
        <div class="metric-card"><div class="value" id="ix-k-pglom">--</div><div class="label">P_glom</div></div>
        <div class="metric-card"><div class="value" id="ix-k-na">--</div><div class="label">Na excr</div></div>
        <div class="metric-card"><div class="value" id="ix-k-vbl">--</div><div class="label">V_blood</div></div>
        <div class="metric-card"><div class="value" id="ix-k-svr">--</div><div class="label">SVR ratio</div></div>
      </div>
      <button class="btn-secondary" onclick="transferKidneyToHeart()" style="margin-top:8px;width:100%;margin-left:0;">
        &larr; Send V_blood/SVR to Heart
      </button>
    </div>
  </div>

</div><!-- /grid -->

<!-- Interactive plots -->
<div class="plot-grid" style="margin-top:16px;">
  <div class="plot-box" id="ix-plot-pv"></div>
  <div class="plot-box" id="ix-plot-wave"></div>
</div>

</div><!-- /tab-interactive -->


<!-- ════════════════════════════════════════════════════ -->
<!-- TAB 3: About                                         -->
<!-- ════════════════════════════════════════════════════ -->
<div class="tab-content" id="tab-about">
<div class="info-section">
  <h2>Why Use the Published CircAdapt Heart Model?</h2>
  <p>
    This application uses the <strong>CircAdapt VanOsta2024</strong> model &mdash;
    a published, peer-reviewed cardiovascular simulator from Maastricht University &mdash;
    rather than a simplified hand-written cardiac model.  Here is why this matters:
  </p>

  <h3>1. Validated Multi-Scale Physiology</h3>
  <p>
    CircAdapt couples sarcomere-level mechanics (active fiber stress, passive stiffness,
    contractile timing) with organ-level hemodynamics (pressure-volume loops, valve dynamics,
    arteriovenous coupling, and interventricular septal interaction via the TriSeg model).
    A simple time-varying elastance model (E(t)) cannot reproduce load-dependent
    Frank-Starling behavior or realistic pressure waveform morphology.
  </p>

  <h3>2. Clinically Calibrated Parameters</h3>
  <p>
    Every CircAdapt parameter maps to a measurable physiological quantity: <code>Sf_act</code>
    is the maximum active fiber stress in kPa, <code>k1</code> controls passive myocardial
    stiffness, <code>p0</code> sets the arterial pressure set-point, wall volumes are in
    m&sup3;.  When we increase <code>k1_scale</code> to simulate HFpEF diastolic dysfunction,
    we are modelling increased passive ventricular stiffness &mdash; the hallmark of HFpEF
    &mdash; a real biophysical mechanism, not an arbitrary gain knob.
  </p>

  <h3>3. Reproducibility</h3>
  <p>
    The model is distributed as a pip-installable package (<code>pip install circadapt</code>)
    wrapping validated C++ numerics.  Anyone can install it, load the same reference state,
    and reproduce results deterministically.
  </p>

  <h3>4. Clean Modular Coupling</h3>
  <p>
    The coupling architecture treats CircAdapt as a black-box hemodynamic server.
    The Hallow renal module needs only three inputs (MAP, CO, CVP) and returns two
    outputs (blood volume, SVR ratio).  Either side can be upgraded independently.
  </p>

  <h3>5. Renal Model: Hallow et al. 2017</h3>
  <p>
    The kidney module implements the key equations from Hallow &amp; Gebremichael
    (CPT:PSP 2017): glomerular filtration via Starling forces, tubuloglomerular feedback
    (TGF), RAAS-mediated efferent arteriolar tone and collecting duct reabsorption,
    tubular sodium handling with pressure-natriuresis, and whole-body volume/Na balance.
  </p>

  <h3>6. Inflammatory Mediator Layer</h3>
  <p>
    Inflammation is modelled as a <strong>mediator layer</strong> sitting between the heart
    and kidney on the coupling graph.  It is not a standalone organ &mdash; it is the
    biochemical medium through which organ damage in one compartment propagates to others.
    Two input schedules control the inflammatory state:
  </p>
  <ul>
    <li><strong>inflammation_scale</strong> (0&ndash;1): systemic inflammation (TNF-&alpha;, IL-6, CRP).
        Affects cardiac contractility (&minus;25%), vascular resistance (+15%), arterial stiffness (+30%),
        glomerular filtration (&minus;20%), afferent arteriolar tone (+20%), RAAS gain (+30%),
        proximal tubule Na reabsorption, and MAP setpoint.</li>
    <li><strong>diabetes_scale</strong> (0&ndash;1): type 2 DM / metabolic syndrome.
        Affects cardiac contractility (&minus;20%), passive myocardial stiffness (+40%, AGE cross-linking),
        arterial stiffness (+50%, AGE), glomerular filtration (biphasic: early hyperfiltration then decline),
        efferent arteriolar constriction (+25%), proximal tubule Na reabsorption (SGLT2), and MAP setpoint.</li>
  </ul>
  <p>
    Both scales compose <em>multiplicatively</em> with the existing Sf_act_scale / Kf_scale
    disease schedules, allowing them to be layered on top of any scenario.
  </p>

  <h3>References</h3>
  <ul>
    <li>CircAdapt framework: <a href="https://framework.circadapt.org" style="color:var(--accent);">framework.circadapt.org</a></li>
    <li>VanOsta et al. (2024) &mdash; VanOsta2024 model</li>
    <li>Hallow &amp; Gebremichael, CPT:PSP 6:383-392, 2017</li>
    <li>Basu et al., PLoS Comput Biol 19(11):e1011598, 2023</li>
    <li>Feldman et al. (2000) &mdash; TNF-&alpha; cardiodepressant effects</li>
    <li>Vlachopoulos et al. (2005) &mdash; CRP and arterial stiffness</li>
    <li>Brenner et al. (1996) &mdash; Diabetic hyperfiltration and nephropathy</li>
    <li>van Heerebeek et al. (2008) &mdash; AGE-mediated diastolic stiffness</li>
    <li>Prenner &amp; Chirinos (2015) &mdash; Diabetes and arterial stiffness</li>
    <li>Vallon (2003) &mdash; Afferent arteriolar dilation in diabetes</li>
  </ul>
</div>
</div><!-- /tab-about -->

</div><!-- /container -->

<script>
// ── Tab switching ─────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

// ── Scenario selector ─────────────────────────────────────────────
function onScenarioChange() {
  const v = document.getElementById('scenario').value;
  document.getElementById('custom-controls').style.display = v === 'custom' ? '' : 'none';
}

// ── Plotly dark layout ────────────────────────────────────────────
const DARK = {
  paper_bgcolor: '#12122a',
  plot_bgcolor:  '#0a0a1a',
  font: { color: '#d0d0e0', size: 11 },
  margin: { l: 55, r: 20, t: 40, b: 45 },
  xaxis: { gridcolor: '#1e1e3a', zerolinecolor: '#1e1e3a' },
  yaxis: { gridcolor: '#1e1e3a', zerolinecolor: '#1e1e3a' },
};
function darkLayout(title, xlab, ylab, extra) {
  return Object.assign({}, DARK, {
    title: { text: title, font: { size: 13, color: '#d0d0e0' } },
    xaxis: Object.assign({}, DARK.xaxis, { title: xlab }),
    yaxis: Object.assign({}, DARK.yaxis, { title: ylab }),
  }, extra || {});
}
const PLOTCFG = { responsive: true, displayModeBar: false };

// ── Color palette for steps ────────────────────────────────────────
function stepColors(n) {
  const colors = [];
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0;
    const r = Math.round(46 + t * 209);
    const g = Math.round(205 - t * 98);
    const b = Math.round(196 - t * 89);
    colors.push(`rgb(${r},${g},${b})`);
  }
  return colors;
}

// ── Set status ─────────────────────────────────────────────────────
function setStatus(el, msg, cls) {
  const s = document.getElementById(el);
  s.textContent = msg;
  s.className = cls || '';
}

// ── Clear results ──────────────────────────────────────────────────
function clearResults() {
  document.getElementById('coupled-plots').style.display = 'none';
  document.getElementById('metrics-row').style.display = 'none';
  setStatus('status', '');
}

// ════════════════════════════════════════════════════════════════════
// COUPLED SIMULATION
// ════════════════════════════════════════════════════════════════════
async function runSimulation() {
  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  setStatus('status', 'Running coupled simulation (this may take 30-90 seconds)...', 'running');

  const scenario = document.getElementById('scenario').value;
  const n_steps = parseInt(document.getElementById('n_steps').value) || 8;
  const dt_hours = parseFloat(document.getElementById('dt_hours').value) || 6;

  const body = { scenario, n_steps, dt_hours };

  if (scenario === 'custom') {
    body.cardiac_schedule = document.getElementById('cardiac_csv').value
      .split(',').map(Number);
    body.stiffness_schedule = document.getElementById('stiff_csv').value
      .split(',').map(Number);
    body.kidney_schedule = document.getElementById('kidney_csv').value
      .split(',').map(Number);
    body.inflammation_schedule = document.getElementById('infl_csv').value
      .split(',').map(Number);
    body.diabetes_schedule = document.getElementById('diab_csv').value
      .split(',').map(Number);
  }

  // Layer modifiers on top of scenario
  const layerChecked = document.getElementById('layer-modifiers').checked;
  if (layerChecked && scenario !== 'custom') {
    body.layer_modifiers = true;
    body.inflammation_value = parseFloat(document.getElementById('infl-slider').value) || 0;
    body.diabetes_value = parseFloat(document.getElementById('diab-slider').value) || 0;
  }

  try {
    const resp = await fetch('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message || 'Unknown error');
    renderCoupledResults(json.data);
    setStatus('status', 'Simulation complete.', '');
  } catch (e) {
    setStatus('status', 'Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

function renderCoupledResults(d) {
  const n = d.steps.length;
  const colors = stepColors(n);

  // ── Metrics row ────────────────────────────────────────────────
  const lastIdx = n - 1;
  const metricsData = [
    { value: d.map[lastIdx].toFixed(0), label: 'MAP (mmHg)' },
    { value: d.sbp[lastIdx].toFixed(0) + '/' + d.dbp[lastIdx].toFixed(0), label: 'BP (mmHg)' },
    { value: d.co[lastIdx].toFixed(2), label: 'CO (L/min)' },
    { value: d.ef[lastIdx].toFixed(0) + '%', label: 'EF' },
    { value: d.sv[lastIdx].toFixed(0), label: 'SV (mL)' },
    { value: d.gfr[lastIdx].toFixed(0), label: 'GFR (mL/min)' },
    { value: d.v_blood[lastIdx].toFixed(0), label: 'V_blood (mL)' },
    { value: d.pven[lastIdx].toFixed(1), label: 'CVP (mmHg)' },
  ];
  // Show inflammation/diabetes metrics if non-zero
  if (d.inflammation_scale && d.inflammation_scale[lastIdx] > 0) {
    metricsData.push({ value: d.inflammation_scale[lastIdx].toFixed(2), label: 'Inflammation' });
  }
  if (d.diabetes_scale && d.diabetes_scale[lastIdx] > 0) {
    metricsData.push({ value: d.diabetes_scale[lastIdx].toFixed(2), label: 'Diabetes' });
  }
  const mrow = document.getElementById('metrics-row');
  mrow.innerHTML = metricsData.map(m =>
    `<div class="metric-card"><div class="value">${m.value}</div><div class="label">${m.label}</div></div>`
  ).join('');
  mrow.style.display = '';

  // ── LV PV loops ────────────────────────────────────────────────
  const pvTraces = d.pv_lv.map((pv, i) => ({
    x: pv.V, y: pv.P, type: 'scatter', mode: 'lines',
    name: `Step ${i+1} (k1=${(d.k1_scale||[])[i]?.toFixed(1)||'1.0'}, Sf=${d.sf_scale[i].toFixed(2)})`,
    line: { color: colors[i], width: 2 },
  }));
  Plotly.newPlot('plot-pv-lv', pvTraces,
    darkLayout('LV Pressure-Volume Loops (CircAdapt VanOsta2024)', 'Volume [mL]', 'Pressure [mmHg]',
      { showlegend: true, legend: { font: { size: 9 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' } }),
    PLOTCFG);

  // ── RV PV loops ────────────────────────────────────────────────
  const rvTraces = d.pv_rv.map((pv, i) => ({
    x: pv.V, y: pv.P, type: 'scatter', mode: 'lines',
    name: `Step ${i+1}`, line: { color: colors[i], width: 2 },
  }));
  Plotly.newPlot('plot-pv-rv', rvTraces,
    darkLayout('RV Pressure-Volume Loops', 'Volume [mL]', 'Pressure [mmHg]',
      { showlegend: true, legend: { font: { size: 9 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' } }),
    PLOTCFG);

  // ── Pressure waveform (last step) ─────────────────────────────
  const wf = d.pressure_waveforms[lastIdx];
  Plotly.newPlot('plot-pressure-waveform', [
    { x: wf.t, y: wf.p_SyArt, type: 'scatter', mode: 'lines',
      name: 'Aortic', line: { color: '#ff6b6b', width: 2 } },
    { x: wf.t, y: wf.p_LV, type: 'scatter', mode: 'lines',
      name: 'LV', line: { color: '#6c8cff', width: 2 } },
  ], darkLayout('Pressure Waveforms (final step)', 'Time [ms]', 'Pressure [mmHg]',
    { showlegend: true, legend: { font: { size: 10 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' } }),
  PLOTCFG);

  // ── Blood pressure trend ──────────────────────────────────────
  Plotly.newPlot('plot-bp', [
    { x: d.steps, y: d.sbp, type: 'scatter', mode: 'lines+markers',
      name: 'SBP', line: { color: '#ff6b6b' }, marker: { size: 7 } },
    { x: d.steps, y: d.dbp, type: 'scatter', mode: 'lines+markers',
      name: 'DBP', line: { color: '#4ecdc4' }, marker: { size: 7 } },
    { x: d.steps, y: d.map, type: 'scatter', mode: 'lines+markers',
      name: 'MAP', line: { color: '#ffd93d', dash: 'dash' }, marker: { size: 5 } },
  ], darkLayout('Blood Pressure', 'Coupling Step', 'mmHg',
    { showlegend: true, legend: { font: { size: 10 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' } }),
  PLOTCFG);

  // ── Cardiac output ────────────────────────────────────────────
  Plotly.newPlot('plot-co', [
    { x: d.steps, y: d.co, type: 'scatter', mode: 'lines+markers',
      name: 'CO', line: { color: '#ffd93d', width: 2.5 }, marker: { size: 7 } },
  ], darkLayout('Cardiac Output', 'Coupling Step', 'L/min',
    { shapes: [{ type: 'line', x0: d.steps[0], x1: d.steps[n-1], y0: 5, y1: 5,
      line: { color: '#555', width: 1, dash: 'dash' } }] }),
  PLOTCFG);

  // ── SV & EF ───────────────────────────────────────────────────
  Plotly.newPlot('plot-sv-ef', [
    { x: d.steps, y: d.sv, type: 'scatter', mode: 'lines+markers',
      name: 'SV (mL)', line: { color: '#87ceeb', width: 2.5 }, marker: { size: 7 }, yaxis: 'y' },
    { x: d.steps, y: d.ef, type: 'scatter', mode: 'lines+markers',
      name: 'EF (%)', line: { color: '#ffb385', width: 2, dash: 'dash' }, marker: { size: 5 }, yaxis: 'y2' },
  ], darkLayout('Stroke Volume & Ejection Fraction', 'Coupling Step', 'SV [mL]', {
    yaxis2: { title: 'EF [%]', overlaying: 'y', side: 'right', gridcolor: '#1e1e3a',
              titlefont: { color: '#ffb385' }, tickfont: { color: '#ffb385' } },
    showlegend: true, legend: { font: { size: 10 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' },
  }), PLOTCFG);

  // ── GFR ───────────────────────────────────────────────────────
  Plotly.newPlot('plot-gfr', [
    { x: d.steps, y: d.gfr, type: 'scatter', mode: 'lines+markers',
      name: 'GFR', line: { color: '#c9b1ff', width: 2.5 }, marker: { size: 7 } },
  ], darkLayout('GFR (Hallow Kidney)', 'Coupling Step', 'mL/min',
    { shapes: [{ type: 'line', x0: d.steps[0], x1: d.steps[n-1], y0: 120, y1: 120,
      line: { color: '#555', width: 1, dash: 'dash' } }] }),
  PLOTCFG);

  // ── Blood volume ──────────────────────────────────────────────
  Plotly.newPlot('plot-vblood', [
    { x: d.steps, y: d.v_blood, type: 'scatter', mode: 'lines+markers',
      name: 'V_blood', line: { color: '#a8e6cf', width: 2.5 }, marker: { size: 7 } },
  ], darkLayout('Blood Volume (Kidney -> Heart)', 'Coupling Step', 'mL',
    { shapes: [{ type: 'line', x0: d.steps[0], x1: d.steps[n-1], y0: 5000, y1: 5000,
      line: { color: '#555', width: 1, dash: 'dash' } }] }),
  PLOTCFG);

  // ── Glomerular pressure ───────────────────────────────────────
  Plotly.newPlot('plot-pglom', [
    { x: d.steps, y: d.p_glom, type: 'scatter', mode: 'lines+markers',
      name: 'P_gc', line: { color: '#ffb385', width: 2.5 }, marker: { size: 7 } },
  ], darkLayout('Glomerular Pressure', 'Coupling Step', 'P_gc [mmHg]'), PLOTCFG);

  // ── Na excretion ──────────────────────────────────────────────
  Plotly.newPlot('plot-na', [
    { x: d.steps, y: d.na_excr, type: 'scatter', mode: 'lines+markers',
      name: 'Na excretion', line: { color: '#a8e6cf', width: 2.5 }, marker: { size: 7 } },
  ], darkLayout('Sodium Excretion', 'Coupling Step', 'mEq/day',
    { shapes: [{ type: 'line', x0: d.steps[0], x1: d.steps[n-1], y0: 150, y1: 150,
      line: { color: '#555', width: 1, dash: 'dash' } }] }),
  PLOTCFG);

  // ── Disease parameters + inflammatory modifiers ─────────────
  const paramTraces = [
    { x: d.steps, y: d.sf_scale, type: 'scatter', mode: 'lines+markers',
      name: 'Contractility (Sf)', line: { color: '#ff6b6b', width: 2.5 }, marker: { size: 7 } },
    { x: d.steps, y: d.kf_scale, type: 'scatter', mode: 'lines+markers',
      name: 'Kidney (Kf)', line: { color: '#4ecdc4', width: 2.5 }, marker: { size: 7 } },
  ];

  // Show k1 stiffness traces
  const hasK1 = d.k1_scale && d.k1_scale.some(v => v > 1.01);
  if (hasK1) {
    paramTraces.push({
      x: d.steps, y: d.k1_scale, type: 'scatter', mode: 'lines+markers',
      name: 'Stiffness (k1)', line: { color: '#ffb385', width: 2.5 }, marker: { size: 7 },
    });
    if (d.effective_k1 && d.effective_k1.some((v,i) => Math.abs(v - d.k1_scale[i]) > 0.01)) {
      paramTraces.push({
        x: d.steps, y: d.effective_k1, type: 'scatter', mode: 'lines+markers',
        name: 'Effective k1', line: { color: '#ffb385', width: 1.5, dash: 'dash' }, marker: { size: 4 },
      });
    }
  }

  // Show inflammation/diabetes traces if any are non-zero
  const hasInflam = d.inflammation_scale && d.inflammation_scale.some(v => v > 0);
  const hasDiab = d.diabetes_scale && d.diabetes_scale.some(v => v > 0);

  if (hasInflam) {
    paramTraces.push({
      x: d.steps, y: d.inflammation_scale, type: 'scatter', mode: 'lines+markers',
      name: 'Inflammation', line: { color: '#ffd93d', width: 2, dash: 'dot' }, marker: { size: 5 },
    });
  }
  if (hasDiab) {
    paramTraces.push({
      x: d.steps, y: d.diabetes_scale, type: 'scatter', mode: 'lines+markers',
      name: 'Diabetes', line: { color: '#c9b1ff', width: 2, dash: 'dot' }, marker: { size: 5 },
    });
  }
  if (hasInflam || hasDiab) {
    paramTraces.push({
      x: d.steps, y: d.effective_sf, type: 'scatter', mode: 'lines+markers',
      name: 'Effective Sf', line: { color: '#ff6b6b', width: 1.5, dash: 'dash' }, marker: { size: 4 },
    });
    paramTraces.push({
      x: d.steps, y: d.effective_kf, type: 'scatter', mode: 'lines+markers',
      name: 'Effective Kf', line: { color: '#4ecdc4', width: 1.5, dash: 'dash' }, marker: { size: 4 },
    });
  }

  // Compute y-axis range: accommodate k1 > 1.0
  const allParamVals = paramTraces.flatMap(t => t.y || []).filter(v => v != null);
  const paramMax = Math.max(...allParamVals, 1.15);

  Plotly.newPlot('plot-params', paramTraces,
    darkLayout('Disease Parameters', 'Coupling Step', 'Scale', {
      yaxis: Object.assign({}, DARK.yaxis, { title: 'Scale', range: [0, paramMax * 1.1] }),
      showlegend: true, legend: { font: { size: 9 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' },
    }), PLOTCFG);

  document.getElementById('coupled-plots').style.display = '';
}


// ════════════════════════════════════════════════════════════════════
// INTERACTIVE SIMULATOR
// ════════════════════════════════════════════════════════════════════

// Store last heart output for transfer
let lastHeartOutput = null;
let lastKidneyOutput = null;

async function runHeartPanel() {
  const btn = document.getElementById('btn-heart');
  btn.disabled = true;
  document.getElementById('ix-heart-status').textContent = 'Running CircAdapt...';
  document.getElementById('ix-heart-status').style.color = '#ffd93d';

  const body = {
    sf_scale: parseFloat(document.getElementById('ix-sf').value),
    stiffness_scale: parseFloat(document.getElementById('ix-k1').value),
    V_blood: parseFloat(document.getElementById('ix-vblood').value),
    SVR_ratio: parseFloat(document.getElementById('ix-svr').value),
    inflammation_scale: parseFloat(document.getElementById('ix-h-infl').value),
    diabetes_scale: parseFloat(document.getElementById('ix-h-diab').value),
  };

  try {
    const resp = await fetch('/single_beat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message || 'Unknown error');
    const d = json.data;
    lastHeartOutput = d;

    // Update metric cards
    document.getElementById('ix-h-map').textContent = d.MAP.toFixed(0);
    document.getElementById('ix-h-co').textContent = d.CO.toFixed(2);
    document.getElementById('ix-h-ef').textContent = d.EF.toFixed(0);
    document.getElementById('ix-h-sbp').textContent = d.SBP.toFixed(0);
    document.getElementById('ix-h-dbp').textContent = d.DBP.toFixed(0);
    document.getElementById('ix-h-pven').textContent = d.Pven.toFixed(1);
    document.getElementById('ix-h-sv').textContent = d.SV.toFixed(0);
    document.getElementById('ix-h-edv').textContent = d.EDV.toFixed(0);
    document.getElementById('ix-h-esv').textContent = d.ESV.toFixed(0);
    document.getElementById('ix-heart-outputs').style.display = '';

    // Update message display
    document.getElementById('ix-msg-h2k').textContent =
      `MAP: ${d.MAP.toFixed(0)} | CO: ${d.CO.toFixed(2)} | CVP: ${d.Pven.toFixed(1)}`;

    // PV loop
    Plotly.newPlot('ix-plot-pv', [
      { x: d.pv_lv.V, y: d.pv_lv.P, type: 'scatter', mode: 'lines',
        name: 'LV', line: { color: '#6c8cff', width: 2.5 } },
      { x: d.pv_rv.V, y: d.pv_rv.P, type: 'scatter', mode: 'lines',
        name: 'RV', line: { color: '#4ecdc4', width: 2 } },
    ], darkLayout(`PV Loops (k1=${body.stiffness_scale.toFixed(1)}, Sf=${body.sf_scale.toFixed(2)})`,
      'Volume [mL]', 'Pressure [mmHg]',
      { showlegend: true, legend: { font: { size: 10 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' } }),
    PLOTCFG);

    // Pressure waveform
    Plotly.newPlot('ix-plot-wave', [
      { x: d.waveform.t, y: d.waveform.p_SyArt, type: 'scatter', mode: 'lines',
        name: 'Aortic', line: { color: '#ff6b6b', width: 2 } },
      { x: d.waveform.t, y: d.waveform.p_LV, type: 'scatter', mode: 'lines',
        name: 'LV', line: { color: '#6c8cff', width: 2 } },
    ], darkLayout('Pressure Waveforms', 'Time [ms]', 'Pressure [mmHg]',
      { showlegend: true, legend: { font: { size: 10 }, bgcolor: '#14142a', bordercolor: '#1e1e3a' } }),
    PLOTCFG);

    document.getElementById('ix-heart-status').textContent = '';
  } catch (e) {
    document.getElementById('ix-heart-status').textContent = 'Error: ' + e.message;
    document.getElementById('ix-heart-status').style.color = '#ff6b6b';
  } finally {
    btn.disabled = false;
  }
}

let kidneyDebounce = null;
async function runKidneyPanel() {
  clearTimeout(kidneyDebounce);
  kidneyDebounce = setTimeout(async () => {
    document.getElementById('ix-kidney-status').textContent = 'Computing...';
    document.getElementById('ix-kidney-status').style.color = '#ffd93d';

    const body = {
      MAP: parseFloat(document.getElementById('ix-map').value),
      CO: parseFloat(document.getElementById('ix-co').value),
      Pven: parseFloat(document.getElementById('ix-pven').value),
      Kf_scale: parseFloat(document.getElementById('ix-kf').value),
      inflammation_scale: parseFloat(document.getElementById('ix-k-infl').value),
      diabetes_scale: parseFloat(document.getElementById('ix-k-diab').value),
    };

    try {
      const resp = await fetch('/kidney_step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const json = await resp.json();
      if (json.status !== 'ok') throw new Error(json.message || 'Unknown error');
      const d = json.data;
      lastKidneyOutput = d;

      document.getElementById('ix-k-gfr').textContent = d.GFR.toFixed(0);
      document.getElementById('ix-k-rbf').textContent = d.RBF.toFixed(0);
      document.getElementById('ix-k-pglom').textContent = d.P_glom.toFixed(0);
      document.getElementById('ix-k-na').textContent = d.Na_excretion.toFixed(0);
      document.getElementById('ix-k-vbl').textContent = d.V_blood.toFixed(0);
      document.getElementById('ix-k-svr').textContent = d.SVR_ratio.toFixed(3);

      // Update message display
      document.getElementById('ix-msg-k2h').textContent =
        `V_blood: ${d.V_blood.toFixed(0)} | SVR: ${d.SVR_ratio.toFixed(3)}`;

      document.getElementById('ix-kidney-status').textContent = '';
    } catch (e) {
      document.getElementById('ix-kidney-status').textContent = 'Error: ' + e.message;
      document.getElementById('ix-kidney-status').style.color = '#ff6b6b';
    }
  }, 200);  // 200ms debounce for auto-run on slider change
}

function transferHeartToKidney() {
  if (!lastHeartOutput) return;
  const d = lastHeartOutput;
  // Set kidney input sliders to heart outputs
  document.getElementById('ix-map').value = Math.round(d.MAP);
  document.getElementById('ix-map-val').textContent = Math.round(d.MAP);
  document.getElementById('ix-co').value = d.CO.toFixed(1);
  document.getElementById('ix-co-val').textContent = d.CO.toFixed(1);
  document.getElementById('ix-pven').value = Math.min(d.Pven, 30).toFixed(1);
  document.getElementById('ix-pven-val').textContent = Math.min(d.Pven, 30).toFixed(1);
  // Auto-run kidney
  runKidneyPanel();
}

function transferKidneyToHeart() {
  if (!lastKidneyOutput) return;
  const d = lastKidneyOutput;
  // Set heart input sliders to kidney outputs
  const vb = Math.max(3000, Math.min(8000, d.V_blood));
  document.getElementById('ix-vblood').value = vb;
  document.getElementById('ix-vblood-val').textContent = Math.round(vb);
  const svr = Math.max(0.5, Math.min(2.0, d.SVR_ratio));
  document.getElementById('ix-svr').value = svr.toFixed(2);
  document.getElementById('ix-svr-val').textContent = svr.toFixed(2);
  // Auto-run heart
  runHeartPanel();
}

// Run kidney on page load for initial values
document.addEventListener('DOMContentLoaded', () => { runKidneyPanel(); });

</script>
</body>
</html>
"""

# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  CircAdapt Cardiorenal Coupling — Flask App")
    print("  Heart:  CircAdapt VanOsta2024 (published module)")
    print("  Kidney: Hallow et al. 2017")
    print("  Open:   http://127.0.0.1:5010")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5010)
