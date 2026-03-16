#!/usr/bin/env python3
"""
CircAdapt Cardiorenal Coupling Dashboard
=========================================
Uses the published CircAdapt VanOsta2024 heart model + Hallow renal module
+ ARIC emission functions for echocardiographic and renal variables.

Heart:   CircAdapt VanOsta2024 (pip install circadapt)
Kidney:  Hallow et al. 2017 CPT:PSP
Coupling: Bidirectional message passing with adjustable intensity
Emissions: ARIC echocardiographic + renal variables (emission_functions.py)

Usage:
    pip install circadapt flask plotly numpy
    python dashboard_circadapt.py
    # Open http://127.0.0.1:8050
"""

import json
import os
import math
import traceback
import copy
from dataclasses import asdict

import numpy as np
from flask import Flask, render_template_string, request, jsonify

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
from emission_functions import extract_all_aric_variables

app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for JSON serialization."""
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


def interpolate_schedule(schedule, n_steps):
    """Interpolate a schedule array to match n_steps."""
    if len(schedule) == n_steps:
        return list(schedule)
    x_old = np.linspace(0, 1, len(schedule))
    x_new = np.linspace(0, 1, n_steps)
    return np.interp(x_new, x_old, schedule).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(params):
    """
    Run coupled cardiorenal simulation using CircAdapt heart model
    with Hallow renal module, feedback loops, and adjustable coupling.
    """
    n_steps = params.get('n_steps', 8)
    dt_hours = params.get('dt_hours', 6.0)
    coupling = params.get('coupling_intensity', 1.0)
    feedback_rate = params.get('feedback_rate', 1.0)

    # Schedules
    stiff_sched = interpolate_schedule(
        params.get('stiffness_schedule', [1.0] * n_steps), n_steps)
    kf_sched = interpolate_schedule(
        params.get('kf_schedule', [1.0] * n_steps), n_steps)
    sf_sched = interpolate_schedule(
        params.get('sf_schedule', [1.0] * n_steps), n_steps)
    infl_sched = interpolate_schedule(
        params.get('inflammation_schedule', [0.0] * n_steps), n_steps)
    diab_sched = interpolate_schedule(
        params.get('diabetes_schedule', [0.0] * n_steps), n_steps)

    heart = CircAdaptHeartModel()
    renal = HallowRenalModel()
    ist = InflammatoryState()
    last_valid_hemo = None

    # Pre-equilibrate: run baseline heart, then equilibrate renal
    baseline_hemo = heart.run_to_steady_state()
    for _ in range(5):
        renal = update_renal_model(
            renal, baseline_hemo['MAP'], baseline_hemo['CO'],
            baseline_hemo['Pven'], dt_hours)
    renal.V_blood = 5000.0
    renal.Na_total = 2100.0

    results = {k: [] for k in [
        'steps',
        'pv_lv', 'pv_rv', 'pressure_waveforms',
        'sbp', 'dbp', 'map', 'co', 'sv', 'ef',
        'hr', 'edv', 'esv', 'v_blood', 'gfr',
        'na_excr', 'p_glom', 'rbf', 'pven',
        'sf_scale', 'kf_scale', 'k1_scale',
        'inflammation_scale', 'diabetes_scale',
        'effective_sf', 'effective_kf', 'effective_k1',
        'solver_crashed',
        'h2k_MAP', 'h2k_CO', 'h2k_CVP',
        'k2h_Vblood', 'k2h_SVR',
        'stiffness_prescribed', 'stiffness_feedback', 'stiffness_total',
        'kf_prescribed', 'kf_feedback_loss', 'kf_total',
        'cardiac_stress', 'renal_stress',
    ]}

    MAP_base = CO_base = Pven_base = None
    SVR_ratio = 1.0

    # Feedback state
    stiffness_fb = 0.0
    kf_fb_loss = 0.0
    STIFF_RATE = 0.10
    KF_LOSS_RATE = 0.02
    dt_factor = dt_hours / 6.0

    for step in range(n_steps):
        sf_prescribed = sf_sched[step]
        k1_prescribed = stiff_sched[step]
        kf_prescribed = kf_sched[step]
        infl = infl_sched[step]
        diab = diab_sched[step]

        # Compose effective values: prescribed + emergent feedback
        k1_eff = k1_prescribed + stiffness_fb
        kf_eff = max(kf_prescribed - kf_fb_loss, 0.05)

        # Update inflammatory state
        ist = update_inflammatory_state(ist, infl, diab)

        # Apply to heart
        heart.apply_inflammatory_modifiers(ist)
        effective_k1 = k1_eff * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)
        effective_sf = max(sf_prescribed * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)
        renal.Kf_scale = kf_eff
        effective_kf = kf_eff * ist.Kf_factor

        # Apply kidney feedback
        heart.apply_kidney_feedback(
            V_blood_m3=renal.V_blood * ML_TO_M3,
            SVR_ratio=SVR_ratio,
        )

        # Run heart to steady state
        hemo = heart.run_to_steady_state()

        # Check for solver crash
        solver_crashed = math.isnan(hemo['MAP'])
        if solver_crashed and last_valid_hemo is not None:
            hemo = last_valid_hemo
        elif not solver_crashed:
            last_valid_hemo = hemo

        if step == 0:
            MAP_base = hemo['MAP']
            CO_base = hemo['CO']
            Pven_base = hemo['Pven']

        # Heart -> Kidney (with coupling intensity)
        h2k_MAP = MAP_base + coupling * (hemo['MAP'] - MAP_base)
        h2k_CO = max(CO_base + coupling * (hemo['CO'] - CO_base), 0.5)
        h2k_Pven = max(Pven_base + coupling * (hemo['Pven'] - Pven_base), 0.5)

        # Update renal
        renal = update_renal_model(
            renal, h2k_MAP, h2k_CO, h2k_Pven, dt_hours, ist)

        # Kidney -> Heart
        k2h = kidney_to_heart(renal, h2k_MAP, h2k_CO, h2k_Pven)
        SVR_ratio = 1.0 + coupling * (k2h.SVR_ratio - 1.0)

        # ── Cross-organ stress indices ──
        volume_excess = max(renal.V_blood - 5000.0, 0.0) / 1000.0
        svr_excess = max(SVR_ratio - 1.0, 0.0)
        raas_excess = max(k2h.SVR_ratio - 1.0, 0.0)
        cardiac_stress = (0.5 * volume_excess + 0.25 * svr_excess
                          + 0.25 * raas_excess)

        co_deficit = max((CO_base or 5.0) - hemo['CO'], 0.0) / 2.0
        map_deficit = max(93.0 - hemo['MAP'], 0.0) / 20.0
        perfusion_stress = 0.5 * co_deficit + 0.5 * map_deficit
        congestion_stress = max(hemo['Pven'] - 3.0, 0.0) / 10.0
        renal_stress = 0.5 * perfusion_stress + 0.5 * congestion_stress

        # Accumulate feedback damage
        stiffness_fb += STIFF_RATE * feedback_rate * cardiac_stress * dt_factor
        stiffness_fb = min(stiffness_fb, 3.0)
        kf_fb_loss += KF_LOSS_RATE * feedback_rate * renal_stress * dt_factor
        kf_fb_loss = min(kf_fb_loss, 0.90)

        # ── Record results ──
        results['steps'].append(step + 1)
        results['pv_lv'].append({
            'V': hemo['V_LV'].tolist(), 'P': hemo['p_LV'].tolist()})
        results['pv_rv'].append({
            'V': hemo['V_RV'].tolist(), 'P': hemo['p_RV'].tolist()})
        results['pressure_waveforms'].append({
            't': hemo['t'].tolist(),
            'p_SyArt': hemo['p_SyArt'].tolist(),
            'p_LV': hemo['p_LV'].tolist()})
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
        results['rbf'].append(round(renal.RBF, 1))
        results['pven'].append(round(hemo['Pven'], 2))
        results['sf_scale'].append(round(sf_prescribed, 3))
        results['kf_scale'].append(round(kf_prescribed, 3))
        results['k1_scale'].append(round(k1_prescribed, 3))
        results['inflammation_scale'].append(round(infl, 3))
        results['diabetes_scale'].append(round(diab, 3))
        results['effective_sf'].append(round(effective_sf, 3))
        results['effective_kf'].append(round(effective_kf, 3))
        results['effective_k1'].append(round(effective_k1, 3))
        results['solver_crashed'].append(solver_crashed)
        results['h2k_MAP'].append(round(h2k_MAP, 1))
        results['h2k_CO'].append(round(h2k_CO, 2))
        results['h2k_CVP'].append(round(h2k_Pven, 1))
        results['k2h_Vblood'].append(round(renal.V_blood, 0))
        results['k2h_SVR'].append(round(SVR_ratio, 3))
        results['stiffness_prescribed'].append(round(k1_prescribed, 3))
        results['stiffness_feedback'].append(round(stiffness_fb, 4))
        results['stiffness_total'].append(round(k1_eff, 3))
        results['kf_prescribed'].append(round(kf_prescribed, 3))
        results['kf_feedback_loss'].append(round(kf_fb_loss, 4))
        results['kf_total'].append(round(kf_eff, 3))
        results['cardiac_stress'].append(round(cardiac_stress, 4))
        results['renal_stress'].append(round(renal_stress, 4))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run coupled simulation with user-specified parameters."""
    try:
        data = request.get_json(force=True)

        n_steps = int(data.get('n_steps', 8))
        n_steps = max(2, min(n_steps, 16))
        dt_hours = float(data.get('dt_hours', 6.0))
        dt_hours = max(1.0, min(dt_hours, 24.0))
        coupling = float(data.get('coupling_intensity', 1.0))
        feedback_rate = float(data.get('feedback_rate', 1.0))

        scenario = data.get('scenario', 'custom')

        # Default schedules
        sf_sched = [1.0] * n_steps
        stiff_sched = [1.0] * n_steps
        kf_sched = [1.0] * n_steps
        infl_sched = [0.0] * n_steps
        diab_sched = [0.0] * n_steps

        if scenario == 'hfpef':
            stiff_sched = np.linspace(1.0, 3.0, n_steps).tolist()
        elif scenario == 'ckd':
            kf_sched = np.linspace(1.0, 0.20, n_steps).tolist()
        elif scenario == 'hfpef_ckd':
            sf_sched = [0.95] * n_steps
            kf_sched = np.linspace(1.0, 0.40, n_steps).tolist()
            stiff_sched = np.linspace(1.0, 2.0, n_steps).tolist()
        elif scenario == 'ckd_hfpef':
            kf_sched = np.linspace(1.0, 0.35, n_steps).tolist()
            half = n_steps // 2
            stiff_sched = ([1.0] * half +
                           np.linspace(1.0, 1.8, n_steps - half).tolist())
        elif scenario == 'diabetic_hfpef':
            kf_sched = np.linspace(1.0, 0.50, n_steps).tolist()
            stiff_sched = np.linspace(1.0, 1.5, n_steps).tolist()
            infl_sched = np.linspace(0.0, 0.3, n_steps).tolist()
            diab_sched = np.linspace(0.2, 0.85, n_steps).tolist()
        elif scenario == 'inflammatory_hfpef':
            sf_sched = [0.9] * n_steps
            kf_sched = np.linspace(1.0, 0.50, n_steps).tolist()
            stiff_sched = np.linspace(1.0, 2.0, n_steps).tolist()
            infl_sched = np.linspace(0.1, 0.8, n_steps).tolist()
        elif scenario == 'sepsis_aki':
            t_norm = np.linspace(0, 1, n_steps)
            infl_curve = np.exp(-((t_norm - 0.35) / 0.2) ** 2)
            infl_sched = (infl_curve * 0.95).tolist()
            sf_sched = (1.0 - 0.30 * infl_curve).tolist()
            kf_sched = (1.0 - 0.25 * infl_curve).tolist()
        elif scenario == 'custom':
            sf_sched = [float(x) for x in data.get('sf_schedule', sf_sched)]
            stiff_sched = [float(x) for x in data.get('stiffness_schedule', stiff_sched)]
            kf_sched = [float(x) for x in data.get('kf_schedule', kf_sched)]
            infl_sched = [float(x) for x in data.get('inflammation_schedule', infl_sched)]
            diab_sched = [float(x) for x in data.get('diabetes_schedule', diab_sched)]

        results = run_simulation({
            'n_steps': n_steps,
            'dt_hours': dt_hours,
            'coupling_intensity': coupling,
            'feedback_rate': feedback_rate,
            'stiffness_schedule': stiff_sched,
            'kf_schedule': kf_sched,
            'sf_schedule': sf_sched,
            'inflammation_schedule': infl_sched,
            'diabetes_schedule': diab_sched,
        })
        return jsonify({'status': 'ok', 'data': sanitize_for_json(results)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/single_beat', methods=['POST'])
def single_beat():
    """Run a single CircAdapt beat and return waveforms + hemodynamics."""
    try:
        data = request.get_json()
        sf_scale = float(np.clip(data.get('sf_scale', 1.0), 0.1, 1.5))
        stiffness_scale = float(np.clip(data.get('stiffness_scale', 1.0), 1.0, 3.5))
        V_blood = float(np.clip(data.get('V_blood', 5000.0), 3000.0, 8000.0))
        SVR_ratio = float(np.clip(data.get('SVR_ratio', 1.0), 0.5, 2.0))
        infl_scale = float(data.get('inflammation_scale', 0.0))
        diab_scale = float(data.get('diabetes_scale', 0.0))

        heart = CircAdaptHeartModel()
        ist = InflammatoryState()
        ist = update_inflammatory_state(ist, infl_scale, diab_scale)
        heart.apply_inflammatory_modifiers(ist)
        effective_k1 = stiffness_scale * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)
        effective_sf = max(sf_scale * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)
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
            'MAP': round(hemo['MAP'], 2), 'SBP': round(hemo['SBP'], 2),
            'DBP': round(hemo['DBP'], 2), 'CO': round(hemo['CO'], 3),
            'SV': round(hemo['SV'], 2), 'EF': round(hemo['EF'], 2),
            'HR': round(hemo['HR'], 1), 'EDV': round(hemo['EDV'], 2),
            'ESV': round(hemo['ESV'], 2), 'Pven': round(hemo['Pven'], 2),
            'V_blood_total': round(hemo['V_blood_total'], 1),
        }
        return jsonify({'status': 'ok', 'data': sanitize_for_json(result)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/kidney_step', methods=['POST'])
def kidney_step():
    """Run a single kidney evaluation given heart outputs."""
    try:
        data = request.get_json()
        MAP = float(np.clip(data.get('MAP', 93.0), 40.0, 200.0))
        CO = float(np.clip(data.get('CO', 5.0), 0.5, 10.0))
        Pven = float(np.clip(data.get('Pven', 3.0), 0.0, 30.0))
        Kf_scale = float(np.clip(data.get('Kf_scale', 1.0), 0.05, 1.0))
        infl_scale = float(data.get('inflammation_scale', 0.0))
        diab_scale = float(data.get('diabetes_scale', 0.0))

        renal = HallowRenalModel()
        renal.Kf_scale = Kf_scale
        ist = InflammatoryState()
        ist = update_inflammatory_state(ist, infl_scale, diab_scale)

        V_blood_before = renal.V_blood
        renal = update_renal_model(renal, MAP, CO, Pven, dt_hours=6.0,
                                   inflammatory_state=ist)
        k2h = kidney_to_heart(renal, MAP, CO, Pven)

        result = {
            'GFR': round(renal.GFR, 2), 'RBF': round(renal.RBF, 1),
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


@app.route('/api/emissions', methods=['POST'])
def emissions():
    """
    Run a single CircAdapt beat and extract all ARIC emission variables.
    Returns ~80+ echocardiographic, Doppler, strain, and renal variables.
    """
    try:
        data = request.get_json() or {}
        sf_scale = float(np.clip(data.get('sf_scale', 1.0), 0.1, 1.5))
        stiffness_scale = float(np.clip(data.get('stiffness_scale', 1.0), 1.0, 3.5))
        V_blood = float(np.clip(data.get('V_blood', 5000.0), 3000.0, 8000.0))
        SVR_ratio = float(np.clip(data.get('SVR_ratio', 1.0), 0.5, 2.0))
        Kf_scale = float(np.clip(data.get('Kf_scale', 1.0), 0.05, 1.0))
        infl_scale = float(data.get('inflammation_scale', 0.0))
        diab_scale = float(data.get('diabetes_scale', 0.0))

        # Build heart model
        heart = CircAdaptHeartModel()
        ist = InflammatoryState()
        ist = update_inflammatory_state(ist, infl_scale, diab_scale)
        heart.apply_inflammatory_modifiers(ist)
        effective_k1 = stiffness_scale * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)
        effective_sf = max(sf_scale * ist.Sf_act_factor, 0.20)
        heart.apply_deterioration(effective_sf)
        heart.apply_kidney_feedback(
            V_blood_m3=V_blood * ML_TO_M3,
            SVR_ratio=SVR_ratio * ist.p0_factor,
        )
        hemo = heart.run_to_steady_state()

        # Build renal state dict for emission functions
        renal = HallowRenalModel()
        renal.Kf_scale = Kf_scale
        renal = update_renal_model(
            renal, hemo['MAP'], hemo['CO'], hemo['Pven'],
            dt_hours=6.0, inflammatory_state=ist)
        renal_state = {
            'GFR': renal.GFR, 'RBF': renal.RBF,
            'P_glom': renal.P_glom,
            'Na_excretion': renal.Na_excretion,
            'water_excretion': renal.water_excretion,
            'V_blood': renal.V_blood, 'C_Na': renal.C_Na,
            'Kf_scale': renal.Kf_scale,
        }

        # Extract all ARIC variables
        all_vars = extract_all_aric_variables(
            heart.model, renal_state,
            BSA=float(data.get('BSA', 1.9)),
            height_m=float(data.get('height_m', 1.70)),
            age=float(data.get('age', 75.0)),
            sex=data.get('sex', 'M'),
        )

        # Group by category for frontend display
        categories = {
            'LV Structure': {k: v for k, v in all_vars.items()
                             if any(k.startswith(p) for p in
                                    ['LVIDd', 'LVIDs', 'IVSd', 'LVPWd',
                                     'LV_mass', 'RWT'])},
            'LV Systolic': {k: v for k, v in all_vars.items()
                            if any(k.startswith(p) for p in
                                   ['LVEDV', 'LVESV', 'SV_mL', 'LVEF',
                                    'CO_Lmin', 'HR_bpm', 'FS', 'GLS'])},
            'Mitral Doppler': {k: v for k, v in all_vars.items()
                               if any(k.startswith(p) for p in
                                      ['E_vel', 'A_vel', 'EA_ratio',
                                       'DT_ms', 'IVRT_ms'])},
            'Tissue Doppler': {k: v for k, v in all_vars.items()
                               if any(k.startswith(p) for p in
                                      ['e_prime', 's_prime', 'a_prime'])},
            'Filling Pressures': {k: v for k, v in all_vars.items()
                                  if any(k.startswith(p) for p in
                                         ['E_e_prime', 'LAP_est'])},
            'Left Atrium': {k: v for k, v in all_vars.items()
                            if any(k.startswith(p) for p in
                                   ['LAV', 'LA_diameter', 'LA_total',
                                    'LARS', 'LA_reservoir', 'LA_conduit',
                                    'LA_pump', 'LA_passive', 'LA_active'])},
            'Right Ventricle': {k: v for k, v in all_vars.items()
                                if any(k.startswith(p) for p in
                                       ['RVEDV', 'RVESV', 'RVEF', 'RVSV',
                                        'RV_basal', 'TAPSE', 'RV_FAC',
                                        'RV_s_prime', 'RV_free'])},
            'Aortic Doppler': {k: v for k, v in all_vars.items()
                               if any(k.startswith(p) for p in
                                      ['LVOT', 'AV_Vmax', 'AV_peak',
                                       'AV_mean', 'AVA'])},
            'Pulmonary Pressures': {k: v for k, v in all_vars.items()
                                    if any(k.startswith(p) for p in
                                           ['PASP', 'PADP', 'mPAP',
                                            'TR_Vmax', 'RAP'])},
            'Blood Pressure': {k: v for k, v in all_vars.items()
                               if any(k.startswith(p) for p in
                                      ['SBP', 'DBP', 'MAP', 'pulse_pressure',
                                       'HR_bpm'])},
            'Right Atrium': {k: v for k, v in all_vars.items()
                             if any(k.startswith(p) for p in ['RAV', 'RA_'])},
            'Timing / MPI': {k: v for k, v in all_vars.items()
                             if any(k.startswith(p) for p in
                                    ['IVCT', 'ET_ms', 'IVRT_lv', 'MPI'])},
            'Myocardial Work': {k: v for k, v in all_vars.items()
                                if any(k.startswith(p) for p in
                                       ['GWI', 'GCW', 'GWW', 'GWE'])},
            'Diastolic Grade': {k: v for k, v in all_vars.items()
                                if any(k.startswith(p) for p in
                                       ['diastolic'])},
            'Vascular': {k: v for k, v in all_vars.items()
                         if any(k.startswith(p) for p in
                                ['Ea_', 'Ees_', 'VA_coupling',
                                 'C_total', 'PWV'])},
            'Indexed': {k: v for k, v in all_vars.items()
                        if any(k.startswith(p) for p in
                               ['LVMi', 'LVEDVi', 'LVESVi', 'SVi',
                                'CI_', 'LAVi', 'RAVi', 'RVEDVi'])},
            'Renal / Lab': {k: v for k, v in all_vars.items()
                            if any(k.startswith(p) for p in
                                   ['eGFR', 'GFR_mL', 'RBF', 'serum_creatinine',
                                    'cystatin', 'BUN', 'UACR', 'serum_Na',
                                    'serum_K', 'blood_volume', 'plasma_volume',
                                    'NTproBNP', 'hsTnT', 'P_glom',
                                    'renal_resistive', 'Kf_scale',
                                    'Na_excretion'])},
        }

        # Round all float values
        for cat in categories.values():
            for k, v in cat.items():
                if isinstance(v, (float, np.floating)):
                    cat[k] = round(float(v), 3)

        return jsonify({
            'status': 'ok',
            'data': sanitize_for_json({
                'all_variables': {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
                                  for k, v in all_vars.items()},
                'categories': categories,
            }),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/validate', methods=['POST'])
def validate():
    """Run 3 canonical scenarios and return comparison with expectation checks."""
    try:
        n_steps = 8
        dt = 6.0

        def check(series, direction):
            first, last = series[0], series[-1]
            delta = last - first
            if direction == 'up':
                return delta > 0.5
            elif direction == 'down':
                return delta < -0.5
            elif direction == 'preserved':
                return abs(delta) / max(abs(first), 1) < 0.15
            return False

        # Scenario A: Heart-Only (HFpEF)
        heart_only = run_simulation({
            'n_steps': n_steps, 'dt_hours': dt,
            'coupling_intensity': 1.0, 'feedback_rate': 1.0,
            'stiffness_schedule': np.linspace(1.0, 3.0, n_steps).tolist(),
            'kf_schedule': [1.0] * n_steps,
            'sf_schedule': [1.0] * n_steps,
            'inflammation_schedule': [0.0] * n_steps,
            'diabetes_schedule': [0.0] * n_steps,
        })

        # Scenario B: Kidney-Only (CKD)
        kidney_only = run_simulation({
            'n_steps': n_steps, 'dt_hours': dt,
            'coupling_intensity': 1.0, 'feedback_rate': 1.0,
            'stiffness_schedule': [1.0] * n_steps,
            'kf_schedule': np.linspace(1.0, 0.20, n_steps).tolist(),
            'sf_schedule': [1.0] * n_steps,
            'inflammation_schedule': [0.0] * n_steps,
            'diabetes_schedule': [0.0] * n_steps,
        })

        # Scenario C: Combined
        combined = run_simulation({
            'n_steps': n_steps, 'dt_hours': dt,
            'coupling_intensity': 1.0, 'feedback_rate': 1.0,
            'stiffness_schedule': np.linspace(1.0, 3.0, n_steps).tolist(),
            'kf_schedule': np.linspace(1.0, 0.20, n_steps).tolist(),
            'sf_schedule': [1.0] * n_steps,
            'inflammation_schedule': [0.0] * n_steps,
            'diabetes_schedule': [0.0] * n_steps,
        })

        expectations = {
            'heart_only': [
                {'metric': 'EF', 'expected': 'Preserved (>45%)',
                 'passed': heart_only['ef'][-1] > 45,
                 'actual': f"{heart_only['ef'][0]:.0f} -> {heart_only['ef'][-1]:.0f}%"},
                {'metric': 'CO', 'expected': 'Decreases',
                 'passed': check(heart_only['co'], 'down'),
                 'actual': f"{heart_only['co'][0]:.1f} -> {heart_only['co'][-1]:.1f} L/min"},
                {'metric': 'GFR', 'expected': 'Decreases (secondary)',
                 'passed': check(heart_only['gfr'], 'down'),
                 'actual': f"{heart_only['gfr'][0]:.0f} -> {heart_only['gfr'][-1]:.0f} mL/min"},
                {'metric': 'V_blood', 'expected': 'Increases',
                 'passed': check(heart_only['v_blood'], 'up'),
                 'actual': f"{heart_only['v_blood'][0]:.0f} -> {heart_only['v_blood'][-1]:.0f} mL"},
            ],
            'kidney_only': [
                {'metric': 'GFR', 'expected': 'Decreases sharply',
                 'passed': check(kidney_only['gfr'], 'down'),
                 'actual': f"{kidney_only['gfr'][0]:.0f} -> {kidney_only['gfr'][-1]:.0f} mL/min"},
                {'metric': 'V_blood', 'expected': 'Increases',
                 'passed': check(kidney_only['v_blood'], 'up'),
                 'actual': f"{kidney_only['v_blood'][0]:.0f} -> {kidney_only['v_blood'][-1]:.0f} mL"},
                {'metric': 'MAP', 'expected': 'Increases',
                 'passed': check(kidney_only['map'], 'up'),
                 'actual': f"{kidney_only['map'][0]:.0f} -> {kidney_only['map'][-1]:.0f} mmHg"},
                {'metric': 'EF', 'expected': 'Preserved',
                 'passed': kidney_only['ef'][-1] > 50,
                 'actual': f"{kidney_only['ef'][0]:.0f} -> {kidney_only['ef'][-1]:.0f}%"},
            ],
            'combined': [
                {'metric': 'GFR', 'expected': 'Worse than either alone',
                 'passed': combined['gfr'][-1] <= min(heart_only['gfr'][-1], kidney_only['gfr'][-1]) + 0.5,
                 'actual': f"{combined['gfr'][-1]:.0f} vs H:{heart_only['gfr'][-1]:.0f} K:{kidney_only['gfr'][-1]:.0f}"},
                {'metric': 'V_blood', 'expected': 'Highest',
                 'passed': combined['v_blood'][-1] >= max(heart_only['v_blood'][-1], kidney_only['v_blood'][-1]) - 0.5,
                 'actual': f"{combined['v_blood'][-1]:.0f} vs H:{heart_only['v_blood'][-1]:.0f} K:{kidney_only['v_blood'][-1]:.0f}"},
                {'metric': 'Feedback', 'expected': 'Both channels active',
                 'passed': (combined['stiffness_feedback'][-1] > 0.005 and combined['kf_feedback_loss'][-1] > 0.005),
                 'actual': f"Stiff_fb={combined['stiffness_feedback'][-1]:.3f}, Kf_loss={combined['kf_feedback_loss'][-1]:.3f}"},
            ],
        }

        return jsonify(sanitize_for_json({
            'status': 'ok',
            'data': {
                'heart_only': heart_only,
                'kidney_only': kidney_only,
                'combined': combined,
                'expectations': expectations,
            }
        }))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/source')
def source_code():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'cardiorenal_coupling.py')
    try:
        with open(src, 'r') as f:
            return jsonify({'code': f.read()})
    except FileNotFoundError:
        return jsonify({'code': '# cardiorenal_coupling.py not found'})


# ═══════════════════════════════════════════════════════════════════════════
# HTML / JS FRONTEND
# ═══════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CircAdapt Cardiorenal Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root {
  --bg:#0b0b14; --surface:#12122a; --border:#1e1e3a; --text:#d0d0e0;
  --accent:#6c8cff; --accent2:#4ecdc4; --red:#ff6b6b; --gold:#ffd93d;
  --mint:#a8e6cf; --lilac:#c9b1ff; --peach:#ffb385;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
header{background:linear-gradient(135deg,#141432,#0e1628);border-bottom:1px solid var(--border);
  padding:16px 28px;display:flex;align-items:center;gap:16px;}
header h1{font-size:1.3rem;font-weight:700;color:var(--accent);}
header .sub{font-size:0.8rem;color:#888;margin-top:2px;}
.container{max-width:1600px;margin:0 auto;padding:18px 22px;}

/* Controls */
.controls{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px;margin-bottom:20px;}
.ctrl{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 18px;}
.ctrl h3{font-size:0.82rem;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);margin-bottom:10px;}
label{display:block;font-size:.78rem;color:#999;margin:8px 0 3px;}
label:first-of-type{margin-top:0;}
select,input[type=number]{width:100%;padding:6px 9px;background:#0a0a1a;border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:.86rem;}
input[type=range]{width:100%;accent-color:var(--accent);margin-top:2px;}
.rr{display:flex;align-items:center;gap:8px;}
.rr input[type=range]{flex:1;}
.rr .rv{font-size:.83rem;font-weight:600;color:var(--accent2);min-width:38px;text-align:right;}
button{display:inline-block;margin-top:12px;padding:9px 20px;border:none;border-radius:7px;
  font-size:.88rem;font-weight:600;cursor:pointer;transition:background .15s;}
.btn-p{background:var(--accent);color:#fff;}
.btn-p:hover{background:#5570e6;}
.btn-p:disabled{background:#333;cursor:wait;}
.btn-s{background:#1e1e3a;color:var(--accent2);border:1px solid var(--border);margin-left:6px;}

/* Status */
#status{font-size:.8rem;color:#888;margin-bottom:14px;min-height:1.2em;}
#status.running{color:var(--gold);}
#status.error{color:var(--red);}

/* Metrics */
.metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-bottom:18px;}
.mc{background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:10px 12px;text-align:center;}
.mc .v{font-size:1.2rem;font-weight:700;color:var(--accent2);}
.mc .l{font-size:.68rem;color:#777;margin-top:2px;text-transform:uppercase;letter-spacing:.04em;}

/* Plots */
.pg{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px;}
.pg.c3{grid-template-columns:1fr 1fr 1fr;}
.pb{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px;min-height:320px;}
.pb.w{grid-column:span 2;}

/* Tabs */
.tabs{display:flex;gap:0;margin-bottom:18px;border-bottom:2px solid var(--border);flex-wrap:wrap;}
.tab{padding:9px 20px;font-size:.87rem;font-weight:600;color:#666;cursor:pointer;
  border-bottom:2px solid transparent;margin-bottom:-2px;transition:color .15s,border-color .15s;}
.tab:hover{color:var(--text);}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.tc{display:none;}
.tc.active{display:block;}

/* Emission table */
.em-cat{margin-bottom:16px;}
.em-cat h4{font-size:.85rem;color:var(--accent);margin-bottom:6px;padding:6px 10px;
  background:var(--surface);border-radius:6px;border-left:3px solid var(--accent);}
.em-tbl{width:100%;border-collapse:collapse;font-size:.8rem;}
.em-tbl td{padding:4px 10px;border-bottom:1px solid var(--border);}
.em-tbl td:first-child{color:var(--accent2);font-family:monospace;width:45%;}
.em-tbl td:last-child{text-align:right;font-weight:600;}

/* Info */
.info{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:22px 26px;line-height:1.7;font-size:.86rem;}
.info h2{color:var(--accent);font-size:1.05rem;margin-bottom:10px;}
.info h3{color:var(--accent2);font-size:.92rem;margin-top:14px;margin-bottom:5px;}
.info ul{margin-left:18px;margin-bottom:6px;}

@media(max-width:900px){
  .pg,.pg.c3{grid-template-columns:1fr;}
  .pb.w{grid-column:span 1;}
  .controls{grid-template-columns:1fr;}
}
</style>
</head>
<body>

<header>
  <div>
    <h1>CircAdapt Cardiorenal Dashboard</h1>
    <div class="sub">Heart: CircAdapt VanOsta2024 | Kidney: Hallow et al. 2017 |
      Inflammation: mediator layer | Emissions: ARIC echo + renal</div>
  </div>
</header>

<div class="container">

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" data-tab="coupled">Coupled Simulation</div>
  <div class="tab" data-tab="interactive">Interactive</div>
  <div class="tab" data-tab="emissions">Emission Variables</div>
  <div class="tab" data-tab="feedback">Feedback Loop</div>
  <div class="tab" data-tab="validate">Validation</div>
  <div class="tab" data-tab="about">About</div>
</div>

<!-- ══════════════════════════════════════════════════════ -->
<!-- TAB: Coupled Simulation                                -->
<!-- ══════════════════════════════════════════════════════ -->
<div class="tc active" id="tab-coupled">

<div class="controls">
  <div class="ctrl">
    <h3>Scenario</h3>
    <label>Pre-built scenario</label>
    <select id="scenario" onchange="onScenarioChange()">
      <optgroup label="HFpEF (Diastolic)">
        <option value="hfpef">Isolated HFpEF</option>
        <option value="hfpef_ckd">HFpEF + CKD (Type 2 CRS)</option>
        <option value="ckd_hfpef">CKD → HFpEF (Type 4 CRS)</option>
      </optgroup>
      <option value="ckd">Progressive CKD</option>
      <optgroup label="Inflammatory / Metabolic">
        <option value="diabetic_hfpef">Diabetic HFpEF</option>
        <option value="inflammatory_hfpef">Inflammatory HFpEF</option>
        <option value="sepsis_aki">Septic Cardiorenal</option>
      </optgroup>
      <option value="custom">Custom schedule</option>
    </select>
  </div>
  <div class="ctrl">
    <h3>Simulation</h3>
    <label>Coupling steps</label>
    <input type="number" id="n_steps" value="8" min="2" max="16">
    <label>Renal dt (hours)</label>
    <input type="number" id="dt_hours" value="6" min="1" max="24" step="1">
  </div>
  <div class="ctrl">
    <h3>Coupling</h3>
    <label>Coupling intensity (0=off, 1=normal, 2=amplified)</label>
    <div class="rr">
      <input type="range" id="coupling" min="0" max="2" step="0.1" value="1.0"
             oninput="document.getElementById('cv').textContent=this.value">
      <span class="rv" id="cv">1.0</span>
    </div>
    <label>Feedback evolution rate</label>
    <div class="rr">
      <input type="range" id="feedback_rate" min="0" max="2" step="0.1" value="1.0"
             oninput="document.getElementById('fv').textContent=this.value">
      <span class="rv" id="fv">1.0</span>
    </div>
  </div>
  <div class="ctrl" id="custom-ctrl" style="display:none;">
    <h3>Custom Schedules</h3>
    <label>Sf_act (comma-sep)</label>
    <input type="text" id="csv_sf" value="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
           style="width:100%;padding:6px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:.82rem;">
    <label>k1 stiffness (comma-sep, ≥1)</label>
    <input type="text" id="csv_k1" value="1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.5"
           style="width:100%;padding:6px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:.82rem;">
    <label>Kf (comma-sep)</label>
    <input type="text" id="csv_kf" value="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
           style="width:100%;padding:6px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:.82rem;">
    <label>Inflammation (comma-sep)</label>
    <input type="text" id="csv_infl" value="0,0,0,0,0,0,0,0"
           style="width:100%;padding:6px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:.82rem;">
    <label>Diabetes (comma-sep)</label>
    <input type="text" id="csv_diab" value="0,0,0,0,0,0,0,0"
           style="width:100%;padding:6px;background:#0a0a1a;border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:.82rem;">
  </div>
  <div class="ctrl" style="display:flex;align-items:flex-end;">
    <div>
      <button class="btn-p" id="btn-run" onclick="runSim()">Run Simulation</button>
      <button class="btn-s" onclick="clearAll()">Clear</button>
    </div>
  </div>
</div>

<div id="status"></div>
<div class="metrics" id="mrow" style="display:none;"></div>

<div class="pg" id="plots" style="display:none;">
  <div class="pb w" id="p-pvlv"></div>
  <div class="pb" id="p-pvrv"></div>
  <div class="pb" id="p-wave"></div>
  <div class="pb" id="p-bp"></div>
  <div class="pb" id="p-co"></div>
  <div class="pb" id="p-svef"></div>
  <div class="pb" id="p-gfr"></div>
  <div class="pb" id="p-vbl"></div>
  <div class="pb" id="p-pglom"></div>
  <div class="pb" id="p-na"></div>
  <div class="pb" id="p-params"></div>
</div>
</div><!-- /tab-coupled -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- TAB: Interactive                                        -->
<!-- ══════════════════════════════════════════════════════ -->
<div class="tc" id="tab-interactive">
<p style="font-size:.83rem;color:#888;margin-bottom:14px;">
  Manually adjust parameters. Use transfer buttons to close the coupling loop.</p>

<div style="display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:start;">
  <!-- Heart Panel -->
  <div class="ctrl" style="min-height:380px;">
    <h3 style="color:var(--red);">Heart</h3>
    <label>Blood Volume (mL) [from kidney]</label>
    <div class="rr"><input type="range" id="ix-vb" min="3000" max="8000" step="50" value="5000"
      oninput="document.getElementById('ix-vb-v').textContent=this.value">
      <span class="rv" id="ix-vb-v">5000</span></div>
    <label>SVR Ratio [from kidney]</label>
    <div class="rr"><input type="range" id="ix-svr" min="0.5" max="2.0" step="0.01" value="1.0"
      oninput="document.getElementById('ix-svr-v').textContent=this.value">
      <span class="rv" id="ix-svr-v">1.0</span></div>
    <label>Contractility (Sf)</label>
    <div class="rr"><input type="range" id="ix-sf" min="0.5" max="1.2" step="0.01" value="1.0"
      oninput="document.getElementById('ix-sf-v').textContent=this.value">
      <span class="rv" id="ix-sf-v">1.0</span></div>
    <label>Stiffness (k1)</label>
    <div class="rr"><input type="range" id="ix-k1" min="1.0" max="3.5" step="0.05" value="1.0"
      oninput="document.getElementById('ix-k1-v').textContent=this.value">
      <span class="rv" id="ix-k1-v">1.0</span></div>
    <button class="btn-p" id="btn-hb" onclick="runHeart()" style="margin-top:8px;">Run Heart Beat</button>
    <div id="ix-hs" style="font-size:.78rem;color:#888;margin-top:4px;"></div>
    <div id="ix-ho" style="display:none;margin-top:10px;">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;">
        <div class="mc"><div class="v" id="ix-hmap">--</div><div class="l">MAP</div></div>
        <div class="mc"><div class="v" id="ix-hco">--</div><div class="l">CO</div></div>
        <div class="mc"><div class="v" id="ix-hef">--</div><div class="l">EF%</div></div>
        <div class="mc"><div class="v" id="ix-hsbp">--</div><div class="l">SBP</div></div>
        <div class="mc"><div class="v" id="ix-hdbp">--</div><div class="l">DBP</div></div>
        <div class="mc"><div class="v" id="ix-hpv">--</div><div class="l">CVP</div></div>
      </div>
      <button class="btn-s" onclick="h2k()" style="margin-top:6px;width:100%;margin-left:0;">
        Send MAP/CO/CVP → Kidney</button>
    </div>
  </div>

  <!-- Arrows -->
  <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;gap:20px;
    min-width:90px;padding-top:60px;">
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:7px;
      padding:8px 10px;text-align:center;font-size:.7rem;">
      <div style="color:var(--red);font-weight:600;">Heart → Kidney</div>
      <div style="color:#888;margin-top:3px;" id="ix-msg-h2k">MAP:-- CO:-- CVP:--</div>
    </div>
    <div style="font-size:1.4rem;color:#444;">↔</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:7px;
      padding:8px 10px;text-align:center;font-size:.7rem;">
      <div style="color:var(--accent2);font-weight:600;">Kidney → Heart</div>
      <div style="color:#888;margin-top:3px;" id="ix-msg-k2h">V_blood:-- SVR:--</div>
    </div>
  </div>

  <!-- Kidney Panel -->
  <div class="ctrl" style="min-height:380px;">
    <h3 style="color:var(--accent2);">Kidney</h3>
    <label>MAP (mmHg) [from heart]</label>
    <div class="rr"><input type="range" id="ix-map" min="40" max="180" step="1" value="93"
      oninput="document.getElementById('ix-map-v').textContent=this.value;runKidney();">
      <span class="rv" id="ix-map-v">93</span></div>
    <label>CO (L/min) [from heart]</label>
    <div class="rr"><input type="range" id="ix-co" min="0.5" max="10" step="0.1" value="5.0"
      oninput="document.getElementById('ix-co-v').textContent=this.value;runKidney();">
      <span class="rv" id="ix-co-v">5.0</span></div>
    <label>CVP (mmHg) [from heart]</label>
    <div class="rr"><input type="range" id="ix-pv" min="0" max="30" step="0.5" value="3.0"
      oninput="document.getElementById('ix-pv-v').textContent=this.value;runKidney();">
      <span class="rv" id="ix-pv-v">3.0</span></div>
    <label>Kf Scale (CKD)</label>
    <div class="rr"><input type="range" id="ix-kf" min="0.05" max="1.0" step="0.01" value="1.0"
      oninput="document.getElementById('ix-kf-v').textContent=this.value;runKidney();">
      <span class="rv" id="ix-kf-v">1.0</span></div>
    <div id="ix-ks" style="font-size:.78rem;color:#888;margin-top:8px;"></div>
    <div id="ix-ko" style="margin-top:10px;">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;">
        <div class="mc"><div class="v" id="ix-kgfr">--</div><div class="l">GFR</div></div>
        <div class="mc"><div class="v" id="ix-krbf">--</div><div class="l">RBF</div></div>
        <div class="mc"><div class="v" id="ix-kpg">--</div><div class="l">P_glom</div></div>
        <div class="mc"><div class="v" id="ix-kna">--</div><div class="l">Na excr</div></div>
        <div class="mc"><div class="v" id="ix-kvb">--</div><div class="l">V_blood</div></div>
        <div class="mc"><div class="v" id="ix-ksvr">--</div><div class="l">SVR</div></div>
      </div>
      <button class="btn-s" onclick="k2h()" style="margin-top:6px;width:100%;margin-left:0;">
        ← Send V_blood/SVR to Heart</button>
    </div>
  </div>
</div>

<div class="pg" style="margin-top:14px;">
  <div class="pb" id="ix-ppv"></div>
  <div class="pb" id="ix-pwv"></div>
</div>
</div><!-- /tab-interactive -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- TAB: Emission Variables                                -->
<!-- ══════════════════════════════════════════════════════ -->
<div class="tc" id="tab-emissions">
<p style="font-size:.83rem;color:#888;margin-bottom:14px;">
  Extract ~80+ ARIC echocardiographic, Doppler, strain, and renal variables from CircAdapt.</p>

<div class="controls">
  <div class="ctrl">
    <h3>Cardiac Parameters</h3>
    <label>Contractility (Sf)</label>
    <div class="rr"><input type="range" id="em-sf" min="0.5" max="1.2" step="0.01" value="1.0"
      oninput="document.getElementById('em-sf-v').textContent=this.value">
      <span class="rv" id="em-sf-v">1.0</span></div>
    <label>Stiffness (k1)</label>
    <div class="rr"><input type="range" id="em-k1" min="1.0" max="3.5" step="0.05" value="1.0"
      oninput="document.getElementById('em-k1-v').textContent=this.value">
      <span class="rv" id="em-k1-v">1.0</span></div>
  </div>
  <div class="ctrl">
    <h3>Renal / Volume</h3>
    <label>Kf Scale (CKD)</label>
    <div class="rr"><input type="range" id="em-kf" min="0.05" max="1.0" step="0.01" value="1.0"
      oninput="document.getElementById('em-kf-v').textContent=this.value">
      <span class="rv" id="em-kf-v">1.0</span></div>
    <label>Blood Volume (mL)</label>
    <div class="rr"><input type="range" id="em-vb" min="3000" max="8000" step="50" value="5000"
      oninput="document.getElementById('em-vb-v').textContent=this.value">
      <span class="rv" id="em-vb-v">5000</span></div>
  </div>
  <div class="ctrl">
    <h3>Pathology</h3>
    <label>Inflammation</label>
    <div class="rr"><input type="range" id="em-infl" min="0" max="1" step="0.05" value="0"
      oninput="document.getElementById('em-infl-v').textContent=this.value">
      <span class="rv" id="em-infl-v">0</span></div>
    <label>Diabetes</label>
    <div class="rr"><input type="range" id="em-diab" min="0" max="1" step="0.05" value="0"
      oninput="document.getElementById('em-diab-v').textContent=this.value">
      <span class="rv" id="em-diab-v">0</span></div>
  </div>
  <div class="ctrl" style="display:flex;align-items:flex-end;">
    <div>
      <button class="btn-p" id="btn-em" onclick="runEmissions()">Extract Variables</button>
      <div id="em-status" style="font-size:.78rem;color:#888;margin-top:6px;"></div>
    </div>
  </div>
</div>

<div id="em-results" style="display:none;"></div>
</div><!-- /tab-emissions -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- TAB: Feedback Loop                                     -->
<!-- ══════════════════════════════════════════════════════ -->
<div class="tc" id="tab-feedback">
<p style="font-size:.83rem;color:#888;margin-bottom:14px;">
  Cross-organ feedback: cardiac stress drives stiffness increase; renal stress drives nephron loss.
  Run a coupled simulation first to see feedback data.</p>
<div class="pg" id="fb-plots" style="display:none;">
  <div class="pb" id="p-stress"></div>
  <div class="pb" id="p-fb"></div>
  <div class="pb" id="p-eff"></div>
  <div class="pb" id="p-coupling"></div>
</div>
<div id="fb-empty" style="color:#555;font-style:italic;">No simulation data yet. Run a coupled simulation first.</div>
</div><!-- /tab-feedback -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- TAB: Validation                                        -->
<!-- ══════════════════════════════════════════════════════ -->
<div class="tc" id="tab-validate">
<p style="font-size:.83rem;color:#888;margin-bottom:14px;">
  Compare three canonical scenarios: Heart-Only (HFpEF), Kidney-Only (CKD), and Combined.</p>
<button class="btn-p" id="btn-val" onclick="runValidation()">Run 3-Scenario Comparison</button>
<div id="val-status" style="font-size:.78rem;color:#888;margin-top:6px;margin-bottom:14px;"></div>
<div id="val-results" style="display:none;"></div>
</div><!-- /tab-validate -->

<!-- ══════════════════════════════════════════════════════ -->
<!-- TAB: About                                             -->
<!-- ══════════════════════════════════════════════════════ -->
<div class="tc" id="tab-about">
<div class="info">
  <h2>Why Use the Published CircAdapt Heart Model?</h2>
  <p>This dashboard uses <strong>CircAdapt VanOsta2024</strong> — a published, peer-reviewed
    cardiovascular simulator — rather than a simplified time-varying elastance model.</p>
  <h3>1. Validated Multi-Scale Physiology</h3>
  <p>CircAdapt couples sarcomere-level mechanics with organ-level hemodynamics (PV loops,
    valve dynamics, TriSeg interventricular interaction). A simple E(t) model cannot reproduce
    load-dependent Frank-Starling behavior or realistic pressure waveforms.</p>
  <h3>2. Clinically Calibrated Parameters</h3>
  <p>Every parameter maps to a measurable quantity: <code>Sf_act</code> = active fiber stress,
    <code>k1</code> = passive myocardial stiffness. Increasing <code>k1</code> models the
    real biophysical mechanism of HFpEF diastolic dysfunction.</p>
  <h3>3. Strain Mapping &amp; LA Metrics</h3>
  <p>CircAdapt provides fiber strain (<code>Ef</code>), sarcomere length, and wall area signals
    that enable computation of GLS, LA reservoir strain (LARS), tissue Doppler velocities,
    and myocardial work indices — all exported via the Emission Variables tab.</p>
  <h3>4. Renal Model: Hallow et al. 2017</h3>
  <p>Glomerular filtration (Starling forces), TGF, RAAS, tubular Na handling, volume balance.</p>
  <h3>5. Emission Functions (ARIC Protocol)</h3>
  <p>~80+ synthetic echocardiographic, Doppler, tissue-Doppler, speckle-tracking, and renal
    variables matching ARIC visits 5 and 7. Includes LV structure, systolic function, diastolic
    grading (ASE/EACVI), LA strain, RV function, pulmonary pressures, vascular coupling, and
    renal biomarkers.</p>
  <h3>References</h3>
  <ul>
    <li>CircAdapt: <a href="https://framework.circadapt.org" style="color:var(--accent);">framework.circadapt.org</a></li>
    <li>VanOsta et al. (2024) — VanOsta2024 model</li>
    <li>Hallow &amp; Gebremichael, CPT:PSP 6:383-392, 2017</li>
    <li>Basu et al., PLoS Comput Biol 19(11):e1011598, 2023</li>
  </ul>
</div>
</div><!-- /tab-about -->

</div><!-- /container -->

<script>
// ── Tabs ──
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tc').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
  });
});

function onScenarioChange(){
  document.getElementById('custom-ctrl').style.display=
    document.getElementById('scenario').value==='custom'?'':'none';
}

// ── Plotly helpers ──
const DK={paper_bgcolor:'#12122a',plot_bgcolor:'#0a0a1a',font:{color:'#d0d0e0',size:11},
  margin:{l:55,r:20,t:40,b:45},
  xaxis:{gridcolor:'#1e1e3a',zerolinecolor:'#1e1e3a'},
  yaxis:{gridcolor:'#1e1e3a',zerolinecolor:'#1e1e3a'}};
function dl(title,xl,yl,ex){
  return Object.assign({},DK,{
    title:{text:title,font:{size:13,color:'#d0d0e0'}},
    xaxis:Object.assign({},DK.xaxis,{title:xl}),
    yaxis:Object.assign({},DK.yaxis,{title:yl})},ex||{});
}
const PC={responsive:true,displayModeBar:false};

function sc(n){
  const c=[];for(let i=0;i<n;i++){
    const t=n>1?i/(n-1):0;
    c.push(`rgb(${Math.round(46+t*209)},${Math.round(205-t*98)},${Math.round(196-t*89)})`);
  }return c;
}

function ss(el,msg,cls){const s=document.getElementById(el);s.textContent=msg;s.className=cls||'';}

function clearAll(){
  document.getElementById('plots').style.display='none';
  document.getElementById('mrow').style.display='none';
  ss('status','');
}

// ── Store last simulation data for feedback tab ──
let lastSimData=null;

// ════════════════════════════════════════════════════════
// COUPLED SIMULATION
// ════════════════════════════════════════════════════════
async function runSim(){
  const btn=document.getElementById('btn-run');btn.disabled=true;
  ss('status','Running coupled simulation (30-90s per step)...','running');
  const scenario=document.getElementById('scenario').value;
  const n=parseInt(document.getElementById('n_steps').value)||8;
  const dt=parseFloat(document.getElementById('dt_hours').value)||6;
  const ci=parseFloat(document.getElementById('coupling').value)||1.0;
  const fr=parseFloat(document.getElementById('feedback_rate').value)||1.0;
  const body={scenario,n_steps:n,dt_hours:dt,coupling_intensity:ci,feedback_rate:fr};
  if(scenario==='custom'){
    body.sf_schedule=document.getElementById('csv_sf').value.split(',').map(Number);
    body.stiffness_schedule=document.getElementById('csv_k1').value.split(',').map(Number);
    body.kf_schedule=document.getElementById('csv_kf').value.split(',').map(Number);
    body.inflammation_schedule=document.getElementById('csv_infl').value.split(',').map(Number);
    body.diabetes_schedule=document.getElementById('csv_diab').value.split(',').map(Number);
  }
  try{
    const r=await fetch('/api/simulate',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(j.status!=='ok')throw new Error(j.message||'Unknown error');
    lastSimData=j.data;
    renderSim(j.data);
    renderFeedback(j.data);
    ss('status','Simulation complete.');
  }catch(e){ss('status','Error: '+e.message,'error');}
  finally{btn.disabled=false;}
}

function renderSim(d){
  const n=d.steps.length,colors=sc(n),li=n-1;
  // Metrics
  const md=[
    {v:d.map[li].toFixed(0),l:'MAP'},{v:d.sbp[li].toFixed(0)+'/'+d.dbp[li].toFixed(0),l:'BP'},
    {v:d.co[li].toFixed(2),l:'CO (L/min)'},{v:d.ef[li].toFixed(0)+'%',l:'EF'},
    {v:d.sv[li].toFixed(0),l:'SV (mL)'},{v:d.gfr[li].toFixed(0),l:'GFR'},
    {v:d.v_blood[li].toFixed(0),l:'V_blood'},{v:d.pven[li].toFixed(1),l:'CVP'},
  ];
  const mr=document.getElementById('mrow');
  mr.innerHTML=md.map(m=>`<div class="mc"><div class="v">${m.v}</div><div class="l">${m.l}</div></div>`).join('');
  mr.style.display='';

  // PV LV
  Plotly.newPlot('p-pvlv',d.pv_lv.map((pv,i)=>({x:pv.V,y:pv.P,type:'scatter',mode:'lines',
    name:`Step ${i+1}`,line:{color:colors[i],width:2}})),
    dl('LV PV Loops (CircAdapt)','Volume [mL]','Pressure [mmHg]',
      {showlegend:true,legend:{font:{size:9},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  // PV RV
  Plotly.newPlot('p-pvrv',d.pv_rv.map((pv,i)=>({x:pv.V,y:pv.P,type:'scatter',mode:'lines',
    name:`Step ${i+1}`,line:{color:colors[i],width:2}})),
    dl('RV PV Loops','Volume [mL]','Pressure [mmHg]',
      {showlegend:true,legend:{font:{size:9},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  // Waveform
  const wf=d.pressure_waveforms[li];
  Plotly.newPlot('p-wave',[
    {x:wf.t,y:wf.p_SyArt,type:'scatter',mode:'lines',name:'Aortic',line:{color:'#ff6b6b',width:2}},
    {x:wf.t,y:wf.p_LV,type:'scatter',mode:'lines',name:'LV',line:{color:'#6c8cff',width:2}},
  ],dl('Pressure Waveforms (final)','Time [ms]','Pressure [mmHg]',
    {showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  // BP
  Plotly.newPlot('p-bp',[
    {x:d.steps,y:d.sbp,type:'scatter',mode:'lines+markers',name:'SBP',line:{color:'#ff6b6b'},marker:{size:6}},
    {x:d.steps,y:d.dbp,type:'scatter',mode:'lines+markers',name:'DBP',line:{color:'#4ecdc4'},marker:{size:6}},
    {x:d.steps,y:d.map,type:'scatter',mode:'lines+markers',name:'MAP',line:{color:'#ffd93d',dash:'dash'},marker:{size:4}},
  ],dl('Blood Pressure','Step','mmHg',
    {showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  // CO
  Plotly.newPlot('p-co',[{x:d.steps,y:d.co,type:'scatter',mode:'lines+markers',
    name:'CO',line:{color:'#ffd93d',width:2.5},marker:{size:6}}],
    dl('Cardiac Output','Step','L/min'),PC);

  // SV & EF
  Plotly.newPlot('p-svef',[
    {x:d.steps,y:d.sv,type:'scatter',mode:'lines+markers',name:'SV',line:{color:'#87ceeb',width:2.5},marker:{size:6}},
    {x:d.steps,y:d.ef,type:'scatter',mode:'lines+markers',name:'EF%',line:{color:'#ffb385',width:2,dash:'dash'},
      marker:{size:4},yaxis:'y2'},
  ],dl('SV & EF','Step','SV [mL]',{
    yaxis2:{title:'EF [%]',overlaying:'y',side:'right',gridcolor:'#1e1e3a',
      titlefont:{color:'#ffb385'},tickfont:{color:'#ffb385'}},
    showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  // GFR
  Plotly.newPlot('p-gfr',[{x:d.steps,y:d.gfr,type:'scatter',mode:'lines+markers',
    name:'GFR',line:{color:'#c9b1ff',width:2.5},marker:{size:6}}],
    dl('GFR (Hallow)','Step','mL/min'),PC);

  // V_blood
  Plotly.newPlot('p-vbl',[{x:d.steps,y:d.v_blood,type:'scatter',mode:'lines+markers',
    name:'V_blood',line:{color:'#a8e6cf',width:2.5},marker:{size:6}}],
    dl('Blood Volume','Step','mL'),PC);

  // P_glom
  Plotly.newPlot('p-pglom',[{x:d.steps,y:d.p_glom,type:'scatter',mode:'lines+markers',
    name:'P_gc',line:{color:'#ffb385',width:2.5},marker:{size:6}}],
    dl('Glomerular Pressure','Step','mmHg'),PC);

  // Na
  Plotly.newPlot('p-na',[{x:d.steps,y:d.na_excr,type:'scatter',mode:'lines+markers',
    name:'Na excr',line:{color:'#a8e6cf',width:2.5},marker:{size:6}}],
    dl('Sodium Excretion','Step','mEq/day'),PC);

  // Params
  const pt=[
    {x:d.steps,y:d.sf_scale,type:'scatter',mode:'lines+markers',name:'Sf',line:{color:'#ff6b6b',width:2.5},marker:{size:6}},
    {x:d.steps,y:d.kf_scale,type:'scatter',mode:'lines+markers',name:'Kf',line:{color:'#4ecdc4',width:2.5},marker:{size:6}},
  ];
  if(d.k1_scale&&d.k1_scale.some(v=>v>1.01))pt.push(
    {x:d.steps,y:d.k1_scale,type:'scatter',mode:'lines+markers',name:'k1',line:{color:'#ffb385',width:2.5},marker:{size:6}});
  if(d.effective_sf)pt.push(
    {x:d.steps,y:d.effective_sf,type:'scatter',mode:'lines+markers',name:'Eff Sf',line:{color:'#ff6b6b',width:1.5,dash:'dash'},marker:{size:3}});
  if(d.effective_kf)pt.push(
    {x:d.steps,y:d.effective_kf,type:'scatter',mode:'lines+markers',name:'Eff Kf',line:{color:'#4ecdc4',width:1.5,dash:'dash'},marker:{size:3}});
  const pmx=Math.max(...pt.flatMap(t=>t.y||[]).filter(v=>v!=null),1.15);
  Plotly.newPlot('p-params',pt,dl('Disease Parameters','Step','Scale',{
    yaxis:Object.assign({},DK.yaxis,{title:'Scale',range:[0,pmx*1.1]}),
    showlegend:true,legend:{font:{size:9},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  document.getElementById('plots').style.display='';
}

// ── Feedback tab ──
function renderFeedback(d){
  if(!d||!d.cardiac_stress)return;
  document.getElementById('fb-plots').style.display='';
  document.getElementById('fb-empty').style.display='none';

  Plotly.newPlot('p-stress',[
    {x:d.steps,y:d.cardiac_stress,type:'scatter',mode:'lines+markers',name:'Cardiac Stress',
      line:{color:'#ff6b6b',width:2.5},marker:{size:6}},
    {x:d.steps,y:d.renal_stress,type:'scatter',mode:'lines+markers',name:'Renal Stress',
      line:{color:'#4ecdc4',width:2.5},marker:{size:6}},
  ],dl('Cross-Organ Stress Indices','Step','Stress',
    {showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  Plotly.newPlot('p-fb',[
    {x:d.steps,y:d.stiffness_feedback,type:'scatter',mode:'lines+markers',name:'Stiffness fb',
      line:{color:'#ffb385',width:2.5},marker:{size:6}},
    {x:d.steps,y:d.kf_feedback_loss,type:'scatter',mode:'lines+markers',name:'Kf loss',
      line:{color:'#c9b1ff',width:2.5},marker:{size:6}},
  ],dl('Emergent Feedback Damage','Step','Accumulated',
    {showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  Plotly.newPlot('p-eff',[
    {x:d.steps,y:d.stiffness_total,type:'scatter',mode:'lines+markers',name:'Total k1',
      line:{color:'#ffb385',width:2.5},marker:{size:6}},
    {x:d.steps,y:d.stiffness_prescribed,type:'scatter',mode:'lines+markers',name:'Prescribed k1',
      line:{color:'#ffb385',width:1.5,dash:'dash'},marker:{size:3}},
    {x:d.steps,y:d.kf_total,type:'scatter',mode:'lines+markers',name:'Total Kf',
      line:{color:'#4ecdc4',width:2.5},marker:{size:6}},
    {x:d.steps,y:d.kf_prescribed,type:'scatter',mode:'lines+markers',name:'Prescribed Kf',
      line:{color:'#4ecdc4',width:1.5,dash:'dash'},marker:{size:3}},
  ],dl('Prescribed vs Effective Parameters','Step','Scale',
    {showlegend:true,legend:{font:{size:9},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);

  Plotly.newPlot('p-coupling',[
    {x:d.steps,y:d.h2k_MAP,type:'scatter',mode:'lines+markers',name:'H→K MAP',
      line:{color:'#ff6b6b',width:2},marker:{size:5}},
    {x:d.steps,y:d.h2k_CO.map(v=>v*20),type:'scatter',mode:'lines+markers',name:'H→K CO×20',
      line:{color:'#ffd93d',width:2},marker:{size:5}},
    {x:d.steps,y:d.k2h_Vblood.map(v=>v/50),type:'scatter',mode:'lines+markers',name:'K→H Vbl/50',
      line:{color:'#a8e6cf',width:2},marker:{size:5}},
    {x:d.steps,y:d.k2h_SVR.map(v=>v*50),type:'scatter',mode:'lines+markers',name:'K→H SVR×50',
      line:{color:'#c9b1ff',width:2},marker:{size:5}},
  ],dl('Coupling Messages (scaled)','Step','Scaled Value',
    {showlegend:true,legend:{font:{size:9},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);
}

// ════════════════════════════════════════════════════════
// INTERACTIVE
// ════════════════════════════════════════════════════════
let lastH=null,lastK=null;

async function runHeart(){
  const btn=document.getElementById('btn-hb');btn.disabled=true;
  document.getElementById('ix-hs').textContent='Running CircAdapt...';
  document.getElementById('ix-hs').style.color='#ffd93d';
  const body={
    sf_scale:parseFloat(document.getElementById('ix-sf').value),
    stiffness_scale:parseFloat(document.getElementById('ix-k1').value),
    V_blood:parseFloat(document.getElementById('ix-vb').value),
    SVR_ratio:parseFloat(document.getElementById('ix-svr').value),
  };
  try{
    const r=await fetch('/api/single_beat',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();if(j.status!=='ok')throw new Error(j.message);
    const d=j.data;lastH=d;
    document.getElementById('ix-hmap').textContent=d.MAP.toFixed(0);
    document.getElementById('ix-hco').textContent=d.CO.toFixed(2);
    document.getElementById('ix-hef').textContent=d.EF.toFixed(0);
    document.getElementById('ix-hsbp').textContent=d.SBP.toFixed(0);
    document.getElementById('ix-hdbp').textContent=d.DBP.toFixed(0);
    document.getElementById('ix-hpv').textContent=d.Pven.toFixed(1);
    document.getElementById('ix-ho').style.display='';
    document.getElementById('ix-msg-h2k').textContent=
      `MAP:${d.MAP.toFixed(0)} CO:${d.CO.toFixed(2)} CVP:${d.Pven.toFixed(1)}`;
    Plotly.newPlot('ix-ppv',[
      {x:d.pv_lv.V,y:d.pv_lv.P,type:'scatter',mode:'lines',name:'LV',line:{color:'#6c8cff',width:2.5}},
      {x:d.pv_rv.V,y:d.pv_rv.P,type:'scatter',mode:'lines',name:'RV',line:{color:'#4ecdc4',width:2}},
    ],dl('PV Loops','Volume [mL]','Pressure [mmHg]',
      {showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);
    Plotly.newPlot('ix-pwv',[
      {x:d.waveform.t,y:d.waveform.p_SyArt,type:'scatter',mode:'lines',name:'Aortic',line:{color:'#ff6b6b',width:2}},
      {x:d.waveform.t,y:d.waveform.p_LV,type:'scatter',mode:'lines',name:'LV',line:{color:'#6c8cff',width:2}},
    ],dl('Pressure Waveforms','Time [ms]','Pressure [mmHg]',
      {showlegend:true,legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);
    document.getElementById('ix-hs').textContent='';
  }catch(e){document.getElementById('ix-hs').textContent='Error: '+e.message;
    document.getElementById('ix-hs').style.color='#ff6b6b';}
  finally{btn.disabled=false;}
}

let kdb=null;
async function runKidney(){
  clearTimeout(kdb);kdb=setTimeout(async()=>{
    document.getElementById('ix-ks').textContent='Computing...';
    document.getElementById('ix-ks').style.color='#ffd93d';
    const body={MAP:parseFloat(document.getElementById('ix-map').value),
      CO:parseFloat(document.getElementById('ix-co').value),
      Pven:parseFloat(document.getElementById('ix-pv').value),
      Kf_scale:parseFloat(document.getElementById('ix-kf').value)};
    try{
      const r=await fetch('/api/kidney_step',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const j=await r.json();if(j.status!=='ok')throw new Error(j.message);
      const d=j.data;lastK=d;
      document.getElementById('ix-kgfr').textContent=d.GFR.toFixed(0);
      document.getElementById('ix-krbf').textContent=d.RBF.toFixed(0);
      document.getElementById('ix-kpg').textContent=d.P_glom.toFixed(0);
      document.getElementById('ix-kna').textContent=d.Na_excretion.toFixed(0);
      document.getElementById('ix-kvb').textContent=d.V_blood.toFixed(0);
      document.getElementById('ix-ksvr').textContent=d.SVR_ratio.toFixed(3);
      document.getElementById('ix-msg-k2h').textContent=
        `V_blood:${d.V_blood.toFixed(0)} SVR:${d.SVR_ratio.toFixed(3)}`;
      document.getElementById('ix-ks').textContent='';
    }catch(e){document.getElementById('ix-ks').textContent='Error: '+e.message;
      document.getElementById('ix-ks').style.color='#ff6b6b';}
  },200);
}

function h2k(){if(!lastH)return;
  document.getElementById('ix-map').value=Math.round(lastH.MAP);
  document.getElementById('ix-map-v').textContent=Math.round(lastH.MAP);
  document.getElementById('ix-co').value=lastH.CO.toFixed(1);
  document.getElementById('ix-co-v').textContent=lastH.CO.toFixed(1);
  document.getElementById('ix-pv').value=Math.min(lastH.Pven,30).toFixed(1);
  document.getElementById('ix-pv-v').textContent=Math.min(lastH.Pven,30).toFixed(1);
  runKidney();
}
function k2h(){if(!lastK)return;
  const vb=Math.max(3000,Math.min(8000,lastK.V_blood));
  document.getElementById('ix-vb').value=vb;
  document.getElementById('ix-vb-v').textContent=Math.round(vb);
  const svr=Math.max(0.5,Math.min(2.0,lastK.SVR_ratio));
  document.getElementById('ix-svr').value=svr.toFixed(2);
  document.getElementById('ix-svr-v').textContent=svr.toFixed(2);
  runHeart();
}

// ════════════════════════════════════════════════════════
// EMISSIONS
// ════════════════════════════════════════════════════════
async function runEmissions(){
  const btn=document.getElementById('btn-em');btn.disabled=true;
  document.getElementById('em-status').textContent='Running CircAdapt + emission extraction...';
  document.getElementById('em-status').style.color='#ffd93d';
  const body={
    sf_scale:parseFloat(document.getElementById('em-sf').value),
    stiffness_scale:parseFloat(document.getElementById('em-k1').value),
    Kf_scale:parseFloat(document.getElementById('em-kf').value),
    V_blood:parseFloat(document.getElementById('em-vb').value),
    inflammation_scale:parseFloat(document.getElementById('em-infl').value),
    diabetes_scale:parseFloat(document.getElementById('em-diab').value),
  };
  try{
    const r=await fetch('/api/emissions',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();if(j.status!=='ok')throw new Error(j.message);
    const cats=j.data.categories;
    let html='';
    for(const[cat,vars]of Object.entries(cats)){
      if(Object.keys(vars).length===0)continue;
      html+=`<div class="em-cat"><h4>${cat}</h4><table class="em-tbl">`;
      for(const[k,v]of Object.entries(vars)){
        const disp=typeof v==='number'?v.toFixed(3):String(v);
        html+=`<tr><td>${k}</td><td>${disp}</td></tr>`;
      }
      html+=`</table></div>`;
    }
    document.getElementById('em-results').innerHTML=html;
    document.getElementById('em-results').style.display='';
    document.getElementById('em-status').textContent=
      `Extracted ${Object.keys(j.data.all_variables).length} variables.`;
    document.getElementById('em-status').style.color='var(--mint)';
  }catch(e){document.getElementById('em-status').textContent='Error: '+e.message;
    document.getElementById('em-status').style.color='#ff6b6b';}
  finally{btn.disabled=false;}
}

// ════════════════════════════════════════════════════════
// VALIDATION
// ════════════════════════════════════════════════════════
async function runValidation(){
  const btn=document.getElementById('btn-val');btn.disabled=true;
  document.getElementById('val-status').textContent='Running 3 scenarios (this may take several minutes)...';
  document.getElementById('val-status').style.color='#ffd93d';
  try{
    const r=await fetch('/api/validate',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
    const j=await r.json();if(j.status!=='ok')throw new Error(j.message);
    const d=j.data;
    let html='<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:20px;">';
    for(const[name,label,color]of[
      ['heart_only','Heart-Only (HFpEF)','#ff6b6b'],
      ['kidney_only','Kidney-Only (CKD)','#4ecdc4'],
      ['combined','Combined','#ffd93d']
    ]){
      html+=`<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;">
        <h3 style="color:${color};font-size:.9rem;margin-bottom:8px;">${label}</h3>`;
      if(d.expectations[name]){
        for(const e of d.expectations[name]){
          const icon=e.passed?'✓':'✗';
          const c=e.passed?'var(--mint)':'var(--red)';
          html+=`<div style="font-size:.78rem;margin-bottom:4px;">
            <span style="color:${c};font-weight:700;">${icon}</span>
            <strong>${e.metric}</strong>: ${e.expected}<br>
            <span style="color:#888;margin-left:16px;">${e.actual}</span></div>`;
        }
      }
      html+=`</div>`;
    }
    html+=`</div>`;

    // Comparison plots
    html+=`<div class="pg"><div class="pb" id="val-gfr"></div><div class="pb" id="val-ef"></div>
      <div class="pb" id="val-vbl"></div><div class="pb" id="val-map"></div></div>`;
    document.getElementById('val-results').innerHTML=html;
    document.getElementById('val-results').style.display='';

    // Plot comparisons
    const ho=d.heart_only,ko=d.kidney_only,co=d.combined;
    for(const[el,key,title,yl]of[
      ['val-gfr','gfr','GFR Comparison','mL/min'],
      ['val-ef','ef','EF Comparison','%'],
      ['val-vbl','v_blood','Blood Volume','mL'],
      ['val-map','map','MAP Comparison','mmHg']
    ]){
      Plotly.newPlot(el,[
        {x:ho.steps,y:ho[key],type:'scatter',mode:'lines+markers',name:'Heart-Only',
          line:{color:'#ff6b6b',width:2},marker:{size:5}},
        {x:ko.steps,y:ko[key],type:'scatter',mode:'lines+markers',name:'Kidney-Only',
          line:{color:'#4ecdc4',width:2},marker:{size:5}},
        {x:co.steps,y:co[key],type:'scatter',mode:'lines+markers',name:'Combined',
          line:{color:'#ffd93d',width:2},marker:{size:5}},
      ],dl(title,'Step',yl,{showlegend:true,
        legend:{font:{size:10},bgcolor:'#14142a',bordercolor:'#1e1e3a'}}),PC);
    }

    document.getElementById('val-status').textContent='Validation complete.';
    document.getElementById('val-status').style.color='var(--mint)';
  }catch(e){document.getElementById('val-status').textContent='Error: '+e.message;
    document.getElementById('val-status').style.color='#ff6b6b';}
  finally{btn.disabled=false;}
}

// Init kidney on load
document.addEventListener('DOMContentLoaded',()=>{runKidney();});
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  CircAdapt Cardiorenal Dashboard")
    print("  Heart:      CircAdapt VanOsta2024")
    print("  Kidney:     Hallow et al. 2017")
    print("  Emissions:  ARIC echo + renal variables")
    print("  Open:       http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8050)