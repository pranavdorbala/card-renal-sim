#!/usr/bin/env python3
"""
Interactive Cardiorenal Syndrome Dashboard
==========================================
HFpEF + CKD Progressive Decline Simulator

Heart Model:  Simplified time-varying elastance with exponential EDPVR
              (captures HFpEF diastolic dysfunction without CircAdapt dependency)
Kidney Model: Hallow et al. 2017 CPT:PSP renal equations
Coupling:     Bidirectional message passing with adjustable intensity

Usage:
    pip install flask plotly numpy
    python dashboard.py
    # Open http://localhost:8050
"""

import json
import os
import numpy as np
from flask import Flask, request, jsonify

# ═══════════════════════════════════════════════════════════════════════════
# HEART MODEL — Time-Varying Elastance for HFpEF
# ═══════════════════════════════════════════════════════════════════════════

def compute_hemodynamics(stiffness_scale=1.0, arterial_stiffness=1.0,
                         R_sys_ratio=1.0, V_blood=5000.0,
                         E_max=2.3, V_0=10.0, A_ed=0.5, beta=0.028,
                         HR=72.0, R_sys=0.018):
    """
    Compute steady-state LV hemodynamics for HFpEF heart model.

    Key HFpEF parameter: stiffness_scale (1.0 = normal, >1 = diastolic dysfunction)
    Increases the EDPVR slope → elevated filling pressures with preserved EF.

    Returns dict with MAP, SBP, DBP, CO, SV, EF, EDV, ESV, LVEDP, CVP, etc.
    """
    R_eff = R_sys * R_sys_ratio * arterial_stiffness

    # Filling pressure from blood volume (simplified venous compliance model)
    P_LA = 10.0 + (V_blood - 5000.0) / 200.0
    P_LA = max(P_LA, 2.0)

    # EDV from EDPVR: stiffness_scale * A_ed * (exp(β*(EDV-V0)) - 1) = P_LA
    edpvr_ratio = P_LA / max(stiffness_scale * A_ed, 0.001) + 1.0
    if edpvr_ratio > 1.0:
        EDV = V_0 + np.log(edpvr_ratio) / beta
    else:
        EDV = V_0 + 30.0
    EDV = float(np.clip(EDV, V_0 + 20, 250.0))

    LVEDP = stiffness_scale * A_ed * (np.exp(min(beta * (EDV - V_0), 20)) - 1)

    # Iterate for MAP-CO consistency
    MAP = 93.0
    for _ in range(40):
        ESV = V_0 + MAP / E_max
        ESV = max(ESV, V_0 + 5)
        ESV = min(ESV, EDV - 1)
        SV = EDV - ESV
        CO_mL = SV * HR
        CVP = max(P_LA - 7.0, 1.0)
        MAP_new = CO_mL * R_eff + CVP
        MAP_new = float(np.clip(MAP_new, 40, 220))
        MAP = 0.7 * MAP + 0.3 * MAP_new

    ESV = V_0 + MAP / E_max
    ESV = min(ESV, EDV - 1)
    SV = max(EDV - ESV, 1.0)
    CO = SV * HR / 1000.0
    EF = SV / max(EDV, 1.0) * 100.0
    CVP = max(P_LA - 7.0, 1.0)

    C_art = 1.5 / max(arterial_stiffness, 0.3)
    PP = SV / max(C_art, 0.3)
    SBP = MAP + PP / 3.0
    DBP = MAP - 2 * PP / 3.0
    ESP = MAP

    return {
        'MAP': round(float(MAP), 2), 'SBP': round(float(SBP), 2),
        'DBP': round(float(DBP), 2), 'CO': round(float(CO), 2),
        'SV': round(float(SV), 2), 'EF': round(float(EF), 1),
        'EDV': round(float(EDV), 1), 'ESV': round(float(ESV), 1),
        'LVEDP': round(float(LVEDP), 1), 'HR': float(HR),
        'CVP': round(float(CVP), 1), 'P_LA': round(float(P_LA), 1),
        'ESP': round(float(ESP), 2),
    }


def generate_pv_loop(hemo, stiffness_scale, A_ed=0.5, beta=0.028,
                     V_0=10.0, n=60):
    """Construct analytical PV loop from hemodynamic parameters."""
    EDV, ESV = hemo['EDV'], hemo['ESV']
    LVEDP, SBP, DBP, ESP = hemo['LVEDP'], hemo['SBP'], hemo['DBP'], hemo['ESP']

    P_ed_ESV = stiffness_scale * A_ed * (np.exp(min(beta * max(ESV - V_0, 0), 20)) - 1)

    # Phase 1: Filling (ESV → EDV along EDPVR)
    V1 = np.linspace(ESV, EDV, n)
    P1 = stiffness_scale * A_ed * (np.exp(np.clip(beta * (V1 - V_0), 0, 20)) - 1)

    # Phase 2: Isovolumic contraction (V=EDV, P: LVEDP → DBP)
    n_iso = max(n // 4, 5)
    t2 = np.linspace(0, np.pi / 2, n_iso)
    P2 = LVEDP + (DBP - LVEDP) * np.sin(t2)
    V2 = np.full(n_iso, EDV)

    # Phase 3: Ejection (EDV → ESV, quadratic Bezier through SBP)
    t3 = np.linspace(0, 1, n)
    P_ctrl = 2 * SBP - (DBP + ESP) / 2
    P3 = (1 - t3)**2 * DBP + 2 * (1 - t3) * t3 * P_ctrl + t3**2 * ESP
    V3 = np.linspace(EDV, ESV, n)

    # Phase 4: Isovolumic relaxation (V=ESV, P: ESP → P_ed(ESV))
    t4 = np.linspace(0, np.pi / 2, n_iso)
    P4 = ESP + (P_ed_ESV - ESP) * np.sin(t4)
    V4 = np.full(n_iso, ESV)

    V = np.concatenate([V1, V2, V3, V4])
    P = np.concatenate([P1, P2, P3, P4])
    return [round(float(x), 2) for x in V], [round(float(x), 2) for x in P]


# ═══════════════════════════════════════════════════════════════════════════
# RENAL MODEL — Hallow et al. 2017
# ═══════════════════════════════════════════════════════════════════════════

def create_renal_state(na_intake=150.0, raas_gain=1.5, tgf_gain=2.0):
    """Create initial renal state dictionary."""
    return {
        'N_nephrons': 1e6, 'Kf': 8.0,
        'R_preAA': 12.0, 'R_AA0': 26.0, 'R_EA0': 43.0,
        'P_Bow': 18.0, 'P_renal_vein': 4.0,
        'pi_plasma': 25.0, 'Hct': 0.45,
        'eta_PT': 0.67, 'eta_LoH': 0.25, 'eta_DT': 0.05, 'eta_CD0': 0.024,
        'frac_water_reabs': 0.99,
        'Na_intake': na_intake, 'water_intake': 2.0,
        'TGF_gain': tgf_gain, 'TGF_setpoint': 0.0,
        'RAAS_gain': raas_gain, 'MAP_setpoint': 93.0,
        'V_blood': 5000.0, 'Na_total': 2100.0, 'C_Na': 140.0,
        'GFR': 120.0, 'RBF': 1100.0, 'P_glom': 60.0,
        'Na_excretion': 150.0, 'water_excretion': 1.5,
        'Kf_scale': 1.0,
        'RAAS_factor': 1.0,
    }


def update_renal(r, MAP, CO, Pven, dt_hours=6.0):
    """Update Hallow renal model given cardiac hemodynamic inputs. Modifies r in-place."""
    Kf_eff = r['Kf'] * r['Kf_scale']
    r['P_renal_vein'] = max(Pven, 2.0)  # venous congestion mechanism

    # 1. RAAS
    dMAP = MAP - r['MAP_setpoint']
    RAAS_factor = float(np.clip(1.0 - r['RAAS_gain'] * 0.005 * dMAP, 0.5, 2.0))
    R_EA = r['R_EA0'] * RAAS_factor
    eta_CD = r['eta_CD0'] * RAAS_factor

    # 2. TGF iteration
    R_AA = r['R_AA0']
    GFR = 120.0
    Na_filt = 0.0
    P_gc = 60.0
    RBF = 1100.0

    for _ in range(12):
        R_total = r['R_preAA'] + R_AA + R_EA
        RBF = max((MAP - r['P_renal_vein']) / R_total * 1000.0, 100.0)
        RPF = RBF * (1.0 - r['Hct'])
        P_gc = MAP - RBF / 1000.0 * (r['R_preAA'] + R_AA)
        P_gc = max(P_gc, 25.0)
        FF = float(np.clip(GFR / max(RPF, 1.0), 0.01, 0.45))
        pi_avg = r['pi_plasma'] * (1.0 + FF / (2.0 * (1.0 - FF)))
        NFP = max(P_gc - r['P_Bow'] - pi_avg, 0.0)
        SNGFR = Kf_eff * NFP
        GFR = max(2.0 * r['N_nephrons'] * SNGFR * 1e-6, 5.0)
        FF = float(np.clip(GFR / max(RPF, 1.0), 0.01, 0.45))
        Na_filt = GFR * r['C_Na'] * 1e-3
        MD_Na = Na_filt * (1.0 - r['eta_PT']) * (1.0 - r['eta_LoH'])
        if r['TGF_setpoint'] <= 0:
            r['TGF_setpoint'] = MD_Na
        TGF_err = (MD_Na - r['TGF_setpoint']) / max(r['TGF_setpoint'], 1e-6)
        R_AA_new = r['R_AA0'] * (1.0 + r['TGF_gain'] * TGF_err)
        R_AA_new = float(np.clip(R_AA_new, 0.5 * r['R_AA0'], 3.0 * r['R_AA0']))
        R_AA = 0.6 * R_AA + 0.4 * R_AA_new

    # 3. Tubular Na handling
    Na_after_PT = Na_filt * (1.0 - r['eta_PT'])
    Na_after_LoH = Na_after_PT * (1.0 - r['eta_LoH'])
    Na_after_DT = Na_after_LoH * (1.0 - r['eta_DT'])
    Na_after_CD = Na_after_DT * (1.0 - eta_CD)
    if MAP > r['MAP_setpoint']:
        pn = 1.0 + 0.03 * (MAP - r['MAP_setpoint'])
    else:
        pn = max(0.3, 1.0 + 0.015 * (MAP - r['MAP_setpoint']))
    Na_excr_min = Na_after_CD * pn
    Na_excr_day = Na_excr_min * 1440.0

    # 4. Water excretion
    water_excr_min = GFR * (1.0 - r['frac_water_reabs'])
    water_excr_day = water_excr_min * 1440.0 / 1000.0

    # 5. Volume / Na balance
    dt_min = dt_hours * 60.0
    Na_in_min = r['Na_intake'] / 1440.0
    r['Na_total'] = max(r['Na_total'] + (Na_in_min - Na_excr_min) * dt_min, 800.0)
    W_in_min = r['water_intake'] * 1000.0 / 1440.0
    dV = (W_in_min - water_excr_min) * dt_min
    r['V_blood'] = float(np.clip(r['V_blood'] + dV * 0.33, 3000.0, 8000.0))
    V_ECF = r['V_blood'] / 0.33
    r['C_Na'] = float(np.clip(r['Na_total'] / (V_ECF * 1e-3), 125.0, 155.0))

    # 6. Store outputs
    r['GFR'] = round(float(GFR), 1)
    r['RBF'] = round(float(RBF), 1)
    r['P_glom'] = round(float(P_gc), 1)
    r['Na_excretion'] = round(float(Na_excr_day), 1)
    r['water_excretion'] = round(float(water_excr_day), 2)

    # 7. Export RAAS activation for systemic SVR feedback
    #    RAAS_factor > 1 when MAP < setpoint → AngII ↑ → systemic vasoconstriction
    r['RAAS_factor'] = float(RAAS_factor)


# ═══════════════════════════════════════════════════════════════════════════
# COUPLED SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_schedule(schedule, n_steps):
    """Interpolate a schedule array to match n_steps."""
    if len(schedule) == n_steps:
        return list(schedule)
    x_old = np.linspace(0, 1, len(schedule))
    x_new = np.linspace(0, 1, n_steps)
    return np.interp(x_new, x_old, schedule).tolist()


def run_simulation(params):
    """Run coupled cardiorenal simulation with message passing and
    optional state-dependent feedback evolution of stiffness and Kf."""
    n_steps = params.get('n_steps', 8)
    dt_hours = params.get('dt_hours', 6.0)
    coupling = params.get('coupling_intensity', 1.0)
    feedback_rate = params.get('feedback_rate', 0.0)
    stiff_sched = interpolate_schedule(params.get('stiffness_schedule', [1.0] * n_steps), n_steps)
    kf_sched = interpolate_schedule(params.get('kf_schedule', [1.0] * n_steps), n_steps)
    art_stiff = params.get('arterial_stiffness', 1.0)
    raas_gain = params.get('raas_gain', 1.5)
    tgf_gain = params.get('tgf_gain', 2.0)
    na_intake = params.get('na_intake', 150.0)

    renal = create_renal_state(na_intake, raas_gain, tgf_gain)

    # Pre-equilibrate renal model so TGF setpoint is calibrated
    pre_hemo = compute_hemodynamics(stiffness_scale=1.0, V_blood=5000.0)
    for _ in range(5):
        update_renal(renal, pre_hemo['MAP'], pre_hemo['CO'], pre_hemo['CVP'], dt_hours)
    renal['V_blood'] = 5000.0  # reset volume after equilibration
    renal['Na_total'] = 2100.0

    results = {k: [] for k in [
        'steps', 'pv_loops', 'MAP', 'SBP', 'DBP', 'CO', 'SV', 'EF',
        'EDV', 'ESV', 'LVEDP', 'CVP', 'P_LA', 'GFR', 'V_blood',
        'Na_excr', 'P_glom', 'RBF', 'stiffness', 'kf',
        'h2k_MAP', 'h2k_CO', 'h2k_CVP', 'k2h_Vblood', 'k2h_SVR',
        # Feedback loop keys
        'stiffness_prescribed', 'stiffness_feedback', 'stiffness_total',
        'kf_prescribed', 'kf_feedback_loss', 'kf_total',
        'cardiac_stress', 'renal_stress',
        # Driver decomposition keys (cardiac LVEDP)
        'driver_cardiac_heart_state', 'driver_cardiac_renal_state',
        'driver_cardiac_renal_rate', 'driver_cardiac_total_delta',
    ]}

    MAP_base = CO_base = CVP_base = None
    V_blood_base = 5000.0
    R_sys_ratio = 1.0

    # Tracking variables for driver decomposition
    prev_EF = None
    prev_LVEDP = None
    prev_V_for_heart = 5000.0
    prev_R_sys_ratio = 1.0
    prev_stiff = 1.0

    # ── Feedback state variables ──
    stiffness_fb = 0.0
    kf_fb_loss = 0.0
    STIFF_RATE = 0.10    # stiffness-units / step / unit cardiac stress
    KF_LOSS_RATE = 0.02  # Kf-units / step / unit renal stress
    dt_factor = dt_hours / 6.0

    for step in range(n_steps):
        stiff_prescribed = stiff_sched[step]
        kf_prescribed = kf_sched[step]

        # ── Compose effective values: prescribed + emergent feedback ──
        stiff = stiff_prescribed + stiffness_fb
        kf = max(kf_prescribed - kf_fb_loss, 0.05)
        renal['Kf_scale'] = kf

        # Heart: use coupled V_blood
        V_for_heart = V_blood_base + coupling * (renal['V_blood'] - V_blood_base)
        hemo = compute_hemodynamics(
            stiffness_scale=stiff, arterial_stiffness=art_stiff,
            R_sys_ratio=R_sys_ratio, V_blood=V_for_heart,
        )

        if step == 0:
            MAP_base, CO_base, CVP_base = hemo['MAP'], hemo['CO'], hemo['CVP']

        # Heart → Kidney (scaled)
        h2k_MAP = MAP_base + coupling * (hemo['MAP'] - MAP_base)
        h2k_CO = max(CO_base + coupling * (hemo['CO'] - CO_base), 0.5)
        h2k_CVP = max(CVP_base + coupling * (hemo['CVP'] - CVP_base), 0.5)

        update_renal(renal, h2k_MAP, h2k_CO, h2k_CVP, dt_hours)

        # Kidney → Heart: SVR driven by renal pathophysiology
        raas_svr = renal.get('RAAS_factor', 1.0)
        ckd_svr = 1.0 + 0.4 * (1.0 - kf)
        svr_from_kidney = raas_svr * ckd_svr
        R_sys_ratio = 1.0 + coupling * (svr_from_kidney - 1.0)

        # ── Compute cross-organ stress indices ──
        #
        # Cardiac stress (kidney-originating → drives stiffness evolution):
        #   Volume overload, afterload excess, RAAS pro-fibrotic signaling
        volume_excess = max(renal['V_blood'] - V_blood_base, 0.0) / 1000.0
        svr_excess = max(R_sys_ratio - 1.0, 0.0)
        raas_excess = max(raas_svr - 1.0, 0.0)
        cardiac_stress = 0.5 * volume_excess + 0.25 * svr_excess + 0.25 * raas_excess

        # Renal stress (heart-originating → drives Kf loss):
        #   Perfusion deficit (low CO, low MAP) + venous congestion (high CVP)
        co_deficit = max((CO_base or 5.0) - hemo['CO'], 0.0) / 2.0
        map_deficit = max(93.0 - hemo['MAP'], 0.0) / 20.0
        perfusion_stress = 0.5 * co_deficit + 0.5 * map_deficit
        congestion_stress = max(hemo['CVP'] - 3.0, 0.0) / 10.0
        renal_stress = 0.5 * perfusion_stress + 0.5 * congestion_stress

        # ── Accumulate feedback damage ──
        stiffness_fb += STIFF_RATE * feedback_rate * cardiac_stress * dt_factor
        stiffness_fb = min(stiffness_fb, 3.0)
        kf_fb_loss += KF_LOSS_RATE * feedback_rate * renal_stress * dt_factor
        kf_fb_loss = min(kf_fb_loss, 0.90)

        V_pv, P_pv = generate_pv_loop(hemo, stiff)

        # ── Driver decomposition: cardiac ΔEF ──
        # (EF depends on stiffness, V_blood, and SVR — unlike LVEDP which
        #  equals P_LA in this model and depends only on V_blood)
        if step == 0 or prev_EF is None:
            results['driver_cardiac_heart_state'].append(0.0)
            results['driver_cardiac_renal_state'].append(0.0)
            results['driver_cardiac_renal_rate'].append(0.0)
            results['driver_cardiac_total_delta'].append(0.0)
        else:
            total_delta = hemo['EF'] - prev_EF

            # Component 1: heart state pure — stiffness change at baseline renal
            ef_curr_bl = compute_hemodynamics(
                stiffness_scale=stiff, V_blood=5000.0, R_sys_ratio=1.0,
                arterial_stiffness=art_stiff)['EF']
            ef_prev_bl = compute_hemodynamics(
                stiffness_scale=prev_stiff, V_blood=5000.0, R_sys_ratio=1.0,
                arterial_stiffness=art_stiff)['EF']
            heart_state_pure = ef_curr_bl - ef_prev_bl

            # Component 2: renal state amplification — pre-existing V_blood/SVR
            ef_at_prev_renal = compute_hemodynamics(
                stiffness_scale=stiff, V_blood=prev_V_for_heart,
                R_sys_ratio=prev_R_sys_ratio,
                arterial_stiffness=art_stiff)['EF']
            renal_state_amp = ef_at_prev_renal - prev_EF - heart_state_pure

            # Component 3: renal deterioration — ΔV_blood/ΔSVR this step
            renal_rate = total_delta - heart_state_pure - renal_state_amp

            results['driver_cardiac_heart_state'].append(round(heart_state_pure, 4))
            results['driver_cardiac_renal_state'].append(round(renal_state_amp, 4))
            results['driver_cardiac_renal_rate'].append(round(renal_rate, 4))
            results['driver_cardiac_total_delta'].append(round(total_delta, 4))

        # Update tracking for next step's decomposition
        prev_EF = hemo['EF']
        prev_LVEDP = hemo['LVEDP']
        prev_V_for_heart = V_for_heart
        prev_R_sys_ratio = R_sys_ratio
        prev_stiff = stiff

        # ── Record results ──
        results['steps'].append(step + 1)
        results['pv_loops'].append({'V': V_pv, 'P': P_pv})
        for k in ['MAP', 'SBP', 'DBP', 'CO', 'SV', 'EF', 'EDV', 'ESV', 'LVEDP', 'CVP', 'P_LA']:
            results[k].append(hemo[k])
        results['GFR'].append(renal['GFR'])
        results['V_blood'].append(round(renal['V_blood'], 0))
        results['Na_excr'].append(renal['Na_excretion'])
        results['P_glom'].append(renal['P_glom'])
        results['RBF'].append(renal['RBF'])
        results['stiffness'].append(round(stiff, 3))
        results['kf'].append(round(kf, 3))
        results['h2k_MAP'].append(round(h2k_MAP, 1))
        results['h2k_CO'].append(round(h2k_CO, 2))
        results['h2k_CVP'].append(round(h2k_CVP, 1))
        results['k2h_Vblood'].append(round(renal['V_blood'], 0))
        results['k2h_SVR'].append(round(R_sys_ratio, 3))
        # Feedback loop data
        results['stiffness_prescribed'].append(round(stiff_prescribed, 3))
        results['stiffness_feedback'].append(round(stiffness_fb, 4))
        results['stiffness_total'].append(round(stiff, 3))
        results['kf_prescribed'].append(round(kf_prescribed, 3))
        results['kf_feedback_loss'].append(round(kf_fb_loss, 4))
        results['kf_total'].append(round(kf, 3))
        results['cardiac_stress'].append(round(cardiac_stress, 4))
        results['renal_stress'].append(round(renal_stress, 4))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        params = request.get_json(force=True)
        results = run_simulation(params)
        # Also run uncoupled (α=0) for rate amplification comparison
        uncoupled_params = dict(params)
        uncoupled_params['coupling_intensity'] = 0.0
        uncoupled = run_simulation(uncoupled_params)
        # Compute rate amplification metrics
        n = len(results['steps'])
        rate_amp = {}
        for key in ['GFR', 'LVEDP', 'EF']:
            coupled_deltas = [results[key][i] - results[key][i-1] for i in range(1, n)]
            uncoupled_deltas = [uncoupled[key][i] - uncoupled[key][i-1] for i in range(1, n)]
            avg_coupled = sum(abs(d) for d in coupled_deltas) / max(len(coupled_deltas), 1)
            avg_uncoupled = sum(abs(d) for d in uncoupled_deltas) / max(len(uncoupled_deltas), 1)
            # Use a meaningful baseline to avoid division by near-zero
            baseline = max(avg_uncoupled, 0.5 * avg_coupled, 0.1)
            pct = ((avg_coupled - avg_uncoupled) / baseline) * 100
            rate_amp[key] = {
                'coupled_avg_rate': round(avg_coupled, 2),
                'uncoupled_avg_rate': round(avg_uncoupled, 2),
                'amplification_pct': round(pct, 1),
                'coupled_deltas': [round(d, 2) for d in coupled_deltas],
                'uncoupled_deltas': [round(d, 2) for d in uncoupled_deltas],
            }
        # Compute renal GFR driver decomposition
        driver_renal = {
            'kf_driven': [0.0],
            'heart_state': [0.0],
            'cardiac_rate': [0.0],
            'total_delta': [0.0],
        }
        for i in range(1, n):
            coupled_delta = results['GFR'][i] - results['GFR'][i-1]
            uncoupled_delta = uncoupled['GFR'][i] - uncoupled['GFR'][i-1]

            kf_driven = uncoupled_delta
            heart_total = coupled_delta - uncoupled_delta

            # Split heart_total: accumulated deficit vs this-step worsening
            map_deficit = abs(93.0 - results['MAP'][i-1])
            map_change = abs(results['MAP'][i] - results['MAP'][i-1])
            cvp_excess = max(results['CVP'][i-1] - 3.0, 0.0)
            cvp_change = abs(results['CVP'][i] - results['CVP'][i-1])

            state_mag = map_deficit + cvp_excess
            rate_mag = map_change + cvp_change
            total_mag = max(state_mag + rate_mag, 0.01)
            rate_frac = rate_mag / total_mag

            driver_renal['kf_driven'].append(round(kf_driven, 4))
            driver_renal['heart_state'].append(round(heart_total * (1 - rate_frac), 4))
            driver_renal['cardiac_rate'].append(round(heart_total * rate_frac, 4))
            driver_renal['total_delta'].append(round(coupled_delta, 4))

        return jsonify({
            'coupled': results,
            'uncoupled': uncoupled,
            'rate_amplification': rate_amp,
            'driver_analysis': {
                'cardiac': {
                    'heart_state': results['driver_cardiac_heart_state'],
                    'renal_state': results['driver_cardiac_renal_state'],
                    'renal_rate': results['driver_cardiac_renal_rate'],
                    'total_delta': results['driver_cardiac_total_delta'],
                },
                'renal': driver_renal,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/source')
def source_code():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cardiorenal_coupling.py')
    try:
        with open(src, 'r') as f:
            return jsonify({'code': f.read()})
    except FileNotFoundError:
        return jsonify({'code': '# cardiorenal_coupling.py not found in same directory'})


@app.route('/api/validate', methods=['POST'])
def validate():
    """Run all 3 tailored scenarios and return comparison data with expectation checks."""
    try:
        n_steps = 10
        dt = 6.0

        # ── Scenario A: Heart-Only (Isolated HFpEF) ──
        heart_only = run_simulation({
            'n_steps': n_steps, 'dt_hours': dt, 'coupling_intensity': 1.0,
            'feedback_rate': 1.0,
            'stiffness_schedule': [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            'kf_schedule': [1.0]*n_steps,
            'arterial_stiffness': 1.0, 'raas_gain': 1.5, 'tgf_gain': 2.0, 'na_intake': 150.0,
        })

        # ── Scenario B: Kidney-Only (CKD, no primary stiffness) ──
        kidney_only = run_simulation({
            'n_steps': n_steps, 'dt_hours': dt, 'coupling_intensity': 1.0,
            'feedback_rate': 1.0,
            'stiffness_schedule': [1.0]*n_steps,
            'kf_schedule': [1.0, 0.90, 0.78, 0.65, 0.52, 0.42, 0.34, 0.28, 0.23, 0.20],
            'arterial_stiffness': 1.0, 'raas_gain': 1.5, 'tgf_gain': 2.0, 'na_intake': 150.0,
        })

        # ── Scenario C: Combined (HFpEF + CKD simultaneous) ──
        combined = run_simulation({
            'n_steps': n_steps, 'dt_hours': dt, 'coupling_intensity': 1.0,
            'feedback_rate': 1.0,
            'stiffness_schedule': [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            'kf_schedule': [1.0, 0.90, 0.78, 0.65, 0.52, 0.42, 0.34, 0.28, 0.23, 0.20],
            'arterial_stiffness': 1.0, 'raas_gain': 1.5, 'tgf_gain': 2.0, 'na_intake': 150.0,
        })

        def check(series, direction, rel=False):
            """Check if a series moved in the expected direction. 'up'/'down'/'preserved'.
            rel=True uses relative (5%) threshold instead of absolute (0.5)."""
            first, last = series[0], series[-1]
            delta = last - first
            threshold = 0.05 * abs(first) if rel else 0.5
            if direction == 'up':
                return delta > threshold
            elif direction == 'down':
                return delta < -threshold
            elif direction == 'preserved':
                return abs(delta) / max(abs(first), 1) < 0.15
            return False

        def worst_check(combined_val, a_val, b_val, direction):
            """Check if combined is worse than both individual scenarios."""
            if direction == 'up':
                return combined_val >= max(a_val, b_val) - 0.5
            else:
                return combined_val <= min(a_val, b_val) + 0.5

        # ── Build expectation checks ──
        expectations = {
            'heart_only': [
                {'metric': 'LVEDP', 'expected': 'Increases (diastolic dysfunction)',
                 'direction': 'up', 'passed': check(heart_only['LVEDP'], 'up'),
                 'actual': f"{heart_only['LVEDP'][0]:.0f} → {heart_only['LVEDP'][-1]:.0f} mmHg"},
                {'metric': 'EF', 'expected': 'Preserved (>45%) or mildly reduced',
                 'direction': 'preserved', 'passed': heart_only['EF'][-1] > 45,
                 'actual': f"{heart_only['EF'][0]:.0f} → {heart_only['EF'][-1]:.0f}%"},
                {'metric': 'CO', 'expected': 'Decreases (impaired filling)',
                 'direction': 'down', 'passed': check(heart_only['CO'], 'down'),
                 'actual': f"{heart_only['CO'][0]:.1f} → {heart_only['CO'][-1]:.1f} L/min"},
                {'metric': 'CVP', 'expected': 'Increases (venous congestion)',
                 'direction': 'up', 'passed': check(heart_only['CVP'], 'up'),
                 'actual': f"{heart_only['CVP'][0]:.0f} → {heart_only['CVP'][-1]:.0f} mmHg"},
                {'metric': 'GFR', 'expected': 'Decreases (secondary renal impairment)',
                 'direction': 'down', 'passed': check(heart_only['GFR'], 'down'),
                 'actual': f"{heart_only['GFR'][0]:.0f} → {heart_only['GFR'][-1]:.0f} mL/min"},
                {'metric': 'V_blood', 'expected': 'Increases (renal Na/water retention)',
                 'direction': 'up', 'passed': check(heart_only['V_blood'], 'up'),
                 'actual': f"{heart_only['V_blood'][0]:.0f} → {heart_only['V_blood'][-1]:.0f} mL"},
                {'metric': 'Kf feedback', 'expected': 'Emergent Kf loss (cardiac stress damages kidney)',
                 'direction': 'up', 'passed': heart_only['kf_feedback_loss'][-1] > 0.005,
                 'actual': f"Kf loss = {heart_only['kf_feedback_loss'][-1]:.4f}"},
                {'metric': 'Kf total', 'expected': 'Effective Kf < 1.0 (secondary renal damage)',
                 'direction': 'down', 'passed': heart_only['kf_total'][-1] < 0.999,
                 'actual': f"Kf = {heart_only['kf_total'][-1]:.3f}"},
            ],
            'kidney_only': [
                {'metric': 'GFR', 'expected': 'Decreases sharply (nephron loss)',
                 'direction': 'down', 'passed': check(kidney_only['GFR'], 'down'),
                 'actual': f"{kidney_only['GFR'][0]:.0f} → {kidney_only['GFR'][-1]:.0f} mL/min"},
                {'metric': 'V_blood', 'expected': 'Increases (impaired excretion)',
                 'direction': 'up', 'passed': check(kidney_only['V_blood'], 'up'),
                 'actual': f"{kidney_only['V_blood'][0]:.0f} → {kidney_only['V_blood'][-1]:.0f} mL"},
                {'metric': 'MAP', 'expected': 'Increases (volume + SVR overload)',
                 'direction': 'up', 'passed': check(kidney_only['MAP'], 'up'),
                 'actual': f"{kidney_only['MAP'][0]:.0f} → {kidney_only['MAP'][-1]:.0f} mmHg"},
                {'metric': 'SVR', 'expected': 'Increases (RAAS + CKD mechanism)',
                 'direction': 'up', 'passed': check(kidney_only['k2h_SVR'], 'up', rel=True),
                 'actual': f"{kidney_only['k2h_SVR'][0]:.2f} → {kidney_only['k2h_SVR'][-1]:.2f}"},
                {'metric': 'LVEDP', 'expected': 'Increases (volume overload)',
                 'direction': 'up', 'passed': check(kidney_only['LVEDP'], 'up'),
                 'actual': f"{kidney_only['LVEDP'][0]:.0f} → {kidney_only['LVEDP'][-1]:.0f} mmHg"},
                {'metric': 'EF', 'expected': 'Preserved (no primary stiffness)',
                 'direction': 'preserved', 'passed': kidney_only['EF'][-1] > 50,
                 'actual': f"{kidney_only['EF'][0]:.0f} → {kidney_only['EF'][-1]:.0f}%"},
                {'metric': 'Stiff feedback', 'expected': 'Emergent stiffness rise (renal stress stiffens heart)',
                 'direction': 'up', 'passed': kidney_only['stiffness_feedback'][-1] > 0.005,
                 'actual': f"Stiff fb = {kidney_only['stiffness_feedback'][-1]:.4f}"},
                {'metric': 'Stiff total', 'expected': 'Effective stiffness > 1.0 (secondary HFpEF)',
                 'direction': 'up', 'passed': kidney_only['stiffness_total'][-1] > 1.005,
                 'actual': f"Stiff = {kidney_only['stiffness_total'][-1]:.3f}"},
            ],
            'combined': [
                {'metric': 'GFR', 'expected': 'Worse than either scenario alone',
                 'direction': 'down',
                 'passed': worst_check(combined['GFR'][-1], heart_only['GFR'][-1], kidney_only['GFR'][-1], 'down'),
                 'actual': f"{combined['GFR'][-1]:.0f} vs H:{heart_only['GFR'][-1]:.0f} K:{kidney_only['GFR'][-1]:.0f}"},
                {'metric': 'LVEDP', 'expected': 'Worse than either scenario alone',
                 'direction': 'up',
                 'passed': worst_check(combined['LVEDP'][-1], heart_only['LVEDP'][-1], kidney_only['LVEDP'][-1], 'up'),
                 'actual': f"{combined['LVEDP'][-1]:.0f} vs H:{heart_only['LVEDP'][-1]:.0f} K:{kidney_only['LVEDP'][-1]:.0f}"},
                {'metric': 'V_blood', 'expected': 'Worse than either scenario alone',
                 'direction': 'up',
                 'passed': worst_check(combined['V_blood'][-1], heart_only['V_blood'][-1], kidney_only['V_blood'][-1], 'up'),
                 'actual': f"{combined['V_blood'][-1]:.0f} vs H:{heart_only['V_blood'][-1]:.0f} K:{kidney_only['V_blood'][-1]:.0f}"},
                {'metric': 'CO', 'expected': 'Worse than either scenario alone',
                 'direction': 'down',
                 'passed': worst_check(combined['CO'][-1], heart_only['CO'][-1], kidney_only['CO'][-1], 'down'),
                 'actual': f"{combined['CO'][-1]:.1f} vs H:{heart_only['CO'][-1]:.1f} K:{kidney_only['CO'][-1]:.1f}"},
                {'metric': 'CVP', 'expected': 'Highest venous congestion',
                 'direction': 'up',
                 'passed': worst_check(combined['CVP'][-1], heart_only['CVP'][-1], kidney_only['CVP'][-1], 'up'),
                 'actual': f"{combined['CVP'][-1]:.0f} vs H:{heart_only['CVP'][-1]:.0f} K:{kidney_only['CVP'][-1]:.0f}"},
                {'metric': 'Feedback amp.', 'expected': 'Both feedback channels active',
                 'direction': 'up',
                 'passed': (combined['stiffness_feedback'][-1] > 0.005 and combined['kf_feedback_loss'][-1] > 0.005),
                 'actual': f"Stiff_fb={combined['stiffness_feedback'][-1]:.3f}, Kf_loss={combined['kf_feedback_loss'][-1]:.3f}"},
            ],
        }

        return jsonify({
            'heart_only': heart_only,
            'kidney_only': kidney_only,
            'combined': combined,
            'expectations': expectations,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE — loaded from dashboard_template.html
# ═══════════════════════════════════════════════════════════════════════════

_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_template.html')
with open(_template_path, 'r') as _f:
    HTML_TEMPLATE = _f.read()

_OLD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cardiorenal Syndrome Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root {
    --bg0: #0d1117; --bg1: #161b22; --bg2: #21262d; --border: #30363d;
    --t1: #e6edf3; --t2: #8b949e;
    --teal: #4ecdc4; --coral: #ff6b6b; --gold: #ffd93d; --mint: #a8e6cf;
    --lilac: #c9b1ff; --sky: #87ceeb; --peach: #ffb385;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg0); color: var(--t1); display:flex; flex-direction:column; height:100vh; }

.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
           padding: 16px 24px; border-bottom: 1px solid var(--border); flex-shrink:0; }
.header h1 { font-size: 20px; font-weight: 700; }
.header h1 span { color: var(--teal); }
.header p { font-size: 12px; color: var(--t2); margin-top: 2px; }

.main { display: flex; flex: 1; overflow: hidden; }

/* Sidebar */
.sidebar { width: 310px; min-width: 310px; background: var(--bg1); border-right: 1px solid var(--border);
           overflow-y: auto; padding: 16px; flex-shrink: 0; }
.section { margin-bottom: 18px; }
.section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
                 color: var(--teal); margin-bottom: 8px; }
.field { margin-bottom: 10px; }
.field label { display: flex; justify-content: space-between; font-size: 12px; color: var(--t2); margin-bottom: 3px; }
.field label .val { color: var(--teal); font-weight: 600; }
.field select, .field input[type=range] { width: 100%; }
.field select { background: var(--bg2); color: var(--t1); border: 1px solid var(--border);
                border-radius: 6px; padding: 6px 8px; font-size: 13px; }
input[type=range] { -webkit-appearance: none; height: 6px; border-radius: 3px;
                    background: var(--bg2); outline: none; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px;
    border-radius: 50%; background: var(--teal); cursor: pointer; }
.run-btn { width: 100%; padding: 10px; background: linear-gradient(135deg, var(--teal), #3ba89f);
           color: #0d1117; border: none; border-radius: 8px; font-size: 14px; font-weight: 700;
           cursor: pointer; margin-top: 8px; transition: opacity 0.2s; }
.run-btn:hover { opacity: 0.85; }
.run-btn:disabled { opacity: 0.4; cursor: wait; }
.scenario-desc { font-size: 11px; color: var(--t2); background: var(--bg2); border-radius: 6px;
                 padding: 8px; margin-top: 6px; line-height: 1.4; }

/* Content */
.content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* Summary bar */
.summary { display: flex; gap: 0; padding: 0 16px; background: var(--bg1);
           border-bottom: 1px solid var(--border); flex-shrink: 0; min-height: 48px; align-items: center;
           overflow-x: auto; }
.metric { padding: 8px 14px; border-right: 1px solid var(--border); text-align: center; min-width: 90px; }
.metric:last-child { border-right: none; }
.metric .label { font-size: 10px; color: var(--t2); text-transform: uppercase; }
.metric .value { font-size: 15px; font-weight: 700; margin-top: 1px; }
.metric .sub { font-size: 10px; color: var(--t2); }
.good { color: var(--mint); } .warn { color: var(--gold); } .bad { color: var(--coral); }

/* Tabs */
.tabs { display: flex; gap: 0; background: var(--bg1); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.tab-btn { padding: 10px 18px; font-size: 13px; font-weight: 600; color: var(--t2); background: none;
           border: none; border-bottom: 2px solid transparent; cursor: pointer; transition: all 0.2s; }
.tab-btn:hover { color: var(--t1); }
.tab-btn.active { color: var(--teal); border-bottom-color: var(--teal); }

.tab-content { display: none; flex: 1; overflow: auto; padding: 16px; }
.tab-content.active { display: block; }

/* Code viewer */
#source-code { background: var(--bg2); border-radius: 8px; padding: 16px; font-family: 'SF Mono',
    'Fira Code', 'Consolas', monospace; font-size: 12px; line-height: 1.5; overflow: auto;
    white-space: pre; color: var(--t1); max-height: calc(100vh - 200px); tab-size: 4; }

/* Loading */
.loading { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
           background: rgba(13,17,23,0.7); z-index: 100; justify-content: center; align-items: center; }
.loading.show { display: flex; }
.spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--teal);
           border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="header">
    <h1><span>&#9829;</span> Cardiorenal Syndrome Dashboard <span>&#9670;</span></h1>
    <p>HFpEF + CKD Progressive Decline &mdash; Coupled Heart-Kidney Simulation with Adjustable Message Passing</p>
</div>

<div class="main">
<!-- SIDEBAR -->
<div class="sidebar">

<div class="section">
    <div class="section-title">Scenario</div>
    <div class="field">
        <select id="scenario">
            <option value="heart_only" selected>Heart-Only: Isolated HFpEF</option>
            <option value="kidney_only">Kidney-Only: CKD &rarr; Volume Overload</option>
            <option value="combined">Combined: HFpEF + CKD</option>
            <option value="custom">Custom</option>
        </select>
    </div>
    <div class="scenario-desc" id="scenario-desc">Progressive diastolic stiffening with no primary kidney disease. Stiffness &#8593; &rarr; LVEDP &#8593; &rarr; CO &#8595; &rarr; renal hypoperfusion + venous congestion &rarr; secondary GFR decline &rarr; volume retention feeds back to heart.</div>
</div>

<div class="section">
    <div class="section-title">Simulation</div>
    <div class="field">
        <label>Steps <span class="val" id="v-steps">10</span></label>
        <input type="range" id="s-steps" min="4" max="16" value="10" step="1">
    </div>
    <div class="field">
        <label>Renal time-step (h) <span class="val" id="v-dt">6</span></label>
        <input type="range" id="s-dt" min="1" max="24" value="6" step="1">
    </div>
</div>

<div class="section">
    <div class="section-title">HFpEF Heart</div>
    <div class="field">
        <label>Final diastolic stiffness <span class="val" id="v-stiff">3.0</span></label>
        <input type="range" id="s-stiff" min="1.0" max="5.0" value="3.0" step="0.1">
    </div>
    <div class="field">
        <label>Arterial stiffness <span class="val" id="v-artstiff">1.0</span></label>
        <input type="range" id="s-artstiff" min="0.5" max="3.0" value="1.0" step="0.1">
    </div>
</div>

<div class="section">
    <div class="section-title">CKD Kidney</div>
    <div class="field">
        <label>Final nephron function (Kf) <span class="val" id="v-kf">0.50</span></label>
        <input type="range" id="s-kf" min="0.05" max="1.0" value="0.50" step="0.05">
    </div>
    <div class="field">
        <label>Na intake (mEq/day) <span class="val" id="v-na">150</span></label>
        <input type="range" id="s-na" min="50" max="300" value="150" step="10">
    </div>
</div>

<div class="section">
    <div class="section-title">Coupling</div>
    <div class="field">
        <label>Coupling intensity <span class="val" id="v-coupling">1.0</span></label>
        <input type="range" id="s-coupling" min="0.0" max="2.0" value="1.0" step="0.1">
    </div>
    <div class="field">
        <label>RAAS gain <span class="val" id="v-raas">1.5</span></label>
        <input type="range" id="s-raas" min="0.0" max="3.0" value="1.5" step="0.1">
    </div>
    <div class="field">
        <label>TGF gain <span class="val" id="v-tgf">2.0</span></label>
        <input type="range" id="s-tgf" min="0.0" max="4.0" value="2.0" step="0.1">
    </div>
    <div class="field">
        <label>Feedback evolution rate <span class="val" id="v-feedback">1.0</span></label>
        <input type="range" id="s-feedback" min="0.0" max="2.0" value="1.0" step="0.1">
    </div>
</div>

<button class="run-btn" id="run-btn" onclick="runSimulation()">Run Simulation</button>
</div>

<!-- CONTENT -->
<div class="content">

<div class="summary" id="summary"></div>

<div class="tabs">
    <button class="tab-btn active" data-tab="tab-pv">PV Loops</button>
    <button class="tab-btn" data-tab="tab-hemo">Hemodynamics</button>
    <button class="tab-btn" data-tab="tab-renal">Renal Function</button>
    <button class="tab-btn" data-tab="tab-coupling">Coupling</button>
    <button class="tab-btn" data-tab="tab-feedback" style="color:var(--lilac);">&#8635; Feedback Loop</button>
    <button class="tab-btn" data-tab="tab-validate" style="color:var(--gold);">&#9733; Scenario Analysis</button>
    <button class="tab-btn" data-tab="tab-code">Source Code</button>
</div>

<div class="tab-content active" id="tab-pv"><div id="chart-pv" style="width:100%;height:100%;"></div></div>
<div class="tab-content" id="tab-hemo"><div id="chart-hemo" style="width:100%;height:100%;"></div></div>
<div class="tab-content" id="tab-renal"><div id="chart-renal" style="width:100%;height:100%;"></div></div>
<div class="tab-content" id="tab-coupling"><div id="chart-coupling" style="width:100%;height:100%;"></div></div>
<div class="tab-content" id="tab-feedback"><div id="chart-feedback" style="width:100%;height:100%;"></div></div>
<div class="tab-content" id="tab-validate">
    <div style="padding:0 0 16px 0;">
        <button class="run-btn" id="validate-btn" onclick="runValidation()" style="width:auto;padding:10px 28px;display:inline-block;">
            &#9733; Compare All Three Scenarios</button>
        <span id="validate-status" style="margin-left:12px;font-size:12px;color:var(--t2);"></span>
    </div>
    <div id="validate-results" style="display:none;">
        <!-- Pathophysiology maps -->
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px;">
            <div class="scenario-card" style="background:var(--bg1);border:1px solid var(--border);border-radius:10px;padding:16px;">
                <h3 style="color:var(--coral);font-size:14px;margin-bottom:8px;">&#9829; Scenario A: Heart-Only (Isolated HFpEF)</h3>
                <div style="font-size:11px;color:var(--t2);line-height:1.7;">
                    <b style="color:var(--t1);">Primary insult:</b> &#8593; myocardial stiffness (1.0 &rarr; 3.0)<br>
                    <b style="color:var(--t1);">Chain of events:</b><br>
                    1. &#8593; Stiffness &rarr; &#8593; LVEDP (diastolic dysfunction)<br>
                    2. Impaired filling &rarr; &#8595; SV &rarr; &#8595; CO<br>
                    3. &#8595; CO &rarr; &#8595; MAP (reduced perfusion)<br>
                    4. &#8593; LVEDP &rarr; &#8593; CVP (venous congestion)<br>
                    <b style="color:var(--coral);">H&rarr;K:</b> &#8595; MAP, &#8595; CO, &#8593; CVP<br>
                    5. &#8595; MAP + &#8593; CVP &rarr; &#8595; GFR (secondary AKI)<br>
                    6. &#8595; GFR &rarr; Na/H&#8322;O retention &rarr; &#8593; V_blood<br>
                    7. RAAS activation &rarr; &#8593; SVR<br>
                    <b style="color:var(--teal);">K&rarr;H:</b> &#8593; V_blood, &#8593; SVR<br>
                    8. &#8593; V_blood &rarr; further &#8593; LVEDP (vicious cycle)<br>
                    9. &#8593; SVR &rarr; &#8593; afterload &rarr; further &#8595; SV
                </div>
            </div>
            <div class="scenario-card" style="background:var(--bg1);border:1px solid var(--border);border-radius:10px;padding:16px;">
                <h3 style="color:var(--teal);font-size:14px;margin-bottom:8px;">&#9670; Scenario B: Kidney-Only (CKD &rarr; Overload)</h3>
                <div style="font-size:11px;color:var(--t2);line-height:1.7;">
                    <b style="color:var(--t1);">Primary insult:</b> &#8595; nephron function (Kf: 1.0 &rarr; 0.2)<br>
                    <b style="color:var(--t1);">Chain of events:</b><br>
                    1. &#8595; Kf &rarr; &#8595; GFR (nephron loss)<br>
                    2. &#8595; GFR &rarr; &#8595; Na excretion &rarr; &#8593; V_blood<br>
                    3. CKD &rarr; &#8593; SVR (uremic, RAAS, sympathetic)<br>
                    <b style="color:var(--teal);">K&rarr;H:</b> &#8593; V_blood, &#8593; SVR<br>
                    4. &#8593; V_blood &rarr; &#8593; preload &rarr; &#8593; LVEDP<br>
                    5. &#8593; SVR + &#8593; Volume &rarr; &#8593; MAP (HTN)<br>
                    6. EF preserved (no primary stiffness)<br>
                    <b style="color:var(--coral);">H&rarr;K:</b> &#8593; MAP, &#8593; CVP<br>
                    7. &#8593; MAP partially maintains GFR (pressure natriuresis)<br>
                    8. But progressive Kf loss overwhelms compensation<br>
                    9. &#8593; CVP &rarr; venous congestion further impairs GFR
                </div>
            </div>
            <div class="scenario-card" style="background:var(--bg1);border:1px solid var(--border);border-radius:10px;padding:16px;">
                <h3 style="color:var(--gold);font-size:14px;margin-bottom:8px;">&#9733; Scenario C: Combined (HFpEF + CKD)</h3>
                <div style="font-size:11px;color:var(--t2);line-height:1.7;">
                    <b style="color:var(--t1);">Primary insult:</b> Both stiffness &#8593; AND Kf &#8595;<br>
                    <b style="color:var(--t1);">Bidirectional amplification:</b><br>
                    1. Heart: &#8593; stiffness &rarr; &#8593; LVEDP &rarr; &#8595; CO<br>
                    2. Kidney: &#8595; Kf &rarr; &#8595; GFR &rarr; &#8593; V_blood<br>
                    <b style="color:var(--coral);">H&rarr;K:</b> &#8595; MAP + &#8593; CVP compounds Kf-driven GFR loss<br>
                    <b style="color:var(--teal);">K&rarr;H:</b> &#8593; V_blood + &#8593; SVR compounds stiffness-driven LVEDP rise<br>
                    <br>
                    <b style="color:var(--gold);">Vicious cycle effect:</b><br>
                    3. Every metric should be <b>worse</b> than either alone<br>
                    4. GFR falls faster than A or B individually<br>
                    5. LVEDP rises faster than A or B individually<br>
                    6. Volume overload more severe than either alone<br>
                    7. This demonstrates <b>synergistic deterioration</b> &mdash;<br>
                    &nbsp;&nbsp;&nbsp;the hallmark of cardiorenal syndrome
                </div>
            </div>
        </div>
        <!-- Validation table -->
        <div id="validation-table" style="margin-bottom:20px;"></div>
        <!-- Comparison charts -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div id="chart-val-pv" style="height:400px;"></div>
            <div id="chart-val-gfr" style="height:400px;"></div>
            <div id="chart-val-lvedp" style="height:400px;"></div>
            <div id="chart-val-co" style="height:400px;"></div>
            <div id="chart-val-vol" style="height:400px;"></div>
            <div id="chart-val-cvp" style="height:400px;"></div>
            <div id="chart-val-feedback" style="height:400px;grid-column:1/-1;"></div>
        </div>
    </div>
</div>
<div class="tab-content" id="tab-code"><pre id="source-code">Loading source code...</pre></div>

</div>
</div>

<div class="loading" id="loading"><div class="spinner"></div></div>

<script>
// ─── Scenarios ───────────────────────────────────────────────────────────
const SCENARIOS = {
    heart_only: {
        desc: 'Progressive diastolic stiffening with no primary kidney disease. With feedback: cardiac stress (low CO, high CVP) causes emergent Kf loss even without prescribed CKD \u2014 the heart damages the kidney.',
        stiff: [1.0,1.2,1.5,1.8,2.0,2.2,2.4,2.6,2.8,3.0], kf: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        finalStiff: 3.0, finalKf: 1.0, artStiff: 1.0, coupling: 1.0, raas: 1.5, tgf: 2.0, na: 150, steps: 10, feedback: 1.0
    },
    kidney_only: {
        desc: 'Progressive nephron loss with no primary cardiac stiffening. With feedback: renal stress (volume overload, RAAS, SVR) causes emergent stiffness rise even without prescribed HFpEF \u2014 the kidney damages the heart.',
        stiff: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], kf: [1.0,0.90,0.78,0.65,0.52,0.42,0.34,0.28,0.23,0.20],
        finalStiff: 1.0, finalKf: 0.20, artStiff: 1.0, coupling: 1.0, raas: 1.5, tgf: 2.0, na: 150, steps: 10, feedback: 1.0
    },
    combined: {
        desc: 'Both stiffness \u2191 and Kf \u2193 simultaneously. Feedback amplifies both: emergent damage compounds prescribed insults, demonstrating the true cardiorenal vicious cycle. Every metric worse than either alone.',
        stiff: [1.0,1.2,1.5,1.8,2.0,2.2,2.4,2.6,2.8,3.0], kf: [1.0,0.90,0.78,0.65,0.52,0.42,0.34,0.28,0.23,0.20],
        finalStiff: 3.0, finalKf: 0.20, artStiff: 1.0, coupling: 1.0, raas: 1.5, tgf: 2.0, na: 150, steps: 10, feedback: 1.0
    },
    custom: {
        desc: 'Custom parameters. Adjust sliders to define your own deterioration schedule (linear from 1.0 to final value). Set feedback rate > 0 to enable emergent cross-organ damage.',
        stiff: null, kf: null,
        finalStiff: 2.0, finalKf: 0.50, artStiff: 1.0, coupling: 1.0, raas: 1.5, tgf: 2.0, na: 150, steps: 10, feedback: 0.0
    }
};

// ─── State ───────────────────────────────────────────────────────────────
let currentData = null;

// ─── Slider wiring ───────────────────────────────────────────────────────
const sliders = [
    {id:'s-steps',  vid:'v-steps',  fmt: v => v},
    {id:'s-dt',     vid:'v-dt',     fmt: v => v},
    {id:'s-stiff',  vid:'v-stiff',  fmt: v => parseFloat(v).toFixed(1)},
    {id:'s-artstiff', vid:'v-artstiff', fmt: v => parseFloat(v).toFixed(1)},
    {id:'s-kf',     vid:'v-kf',     fmt: v => parseFloat(v).toFixed(2)},
    {id:'s-na',     vid:'v-na',     fmt: v => v},
    {id:'s-coupling', vid:'v-coupling', fmt: v => parseFloat(v).toFixed(1)},
    {id:'s-raas',   vid:'v-raas',   fmt: v => parseFloat(v).toFixed(1)},
    {id:'s-tgf',    vid:'v-tgf',    fmt: v => parseFloat(v).toFixed(1)},
    {id:'s-feedback', vid:'v-feedback', fmt: v => parseFloat(v).toFixed(1)},
];
sliders.forEach(s => {
    const el = document.getElementById(s.id);
    el.addEventListener('input', () => { document.getElementById(s.vid).textContent = s.fmt(el.value); });
});

// ─── Scenario change ─────────────────────────────────────────────────────
document.getElementById('scenario').addEventListener('change', function() {
    const sc = SCENARIOS[this.value];
    if (!sc) return;
    document.getElementById('scenario-desc').textContent = sc.desc;
    if (sc.steps) { document.getElementById('s-steps').value = sc.steps; document.getElementById('v-steps').textContent = sc.steps; }
    document.getElementById('s-stiff').value = sc.finalStiff;
    document.getElementById('v-stiff').textContent = sc.finalStiff.toFixed(1);
    document.getElementById('s-kf').value = sc.finalKf;
    document.getElementById('v-kf').textContent = sc.finalKf.toFixed(2);
    document.getElementById('s-artstiff').value = sc.artStiff;
    document.getElementById('v-artstiff').textContent = sc.artStiff.toFixed(1);
    document.getElementById('s-coupling').value = sc.coupling;
    document.getElementById('v-coupling').textContent = sc.coupling.toFixed(1);
    document.getElementById('s-raas').value = sc.raas;
    document.getElementById('v-raas').textContent = sc.raas.toFixed(1);
    document.getElementById('s-tgf').value = sc.tgf;
    document.getElementById('v-tgf').textContent = sc.tgf.toFixed(1);
    document.getElementById('s-na').value = sc.na;
    document.getElementById('v-na').textContent = sc.na;
    if (sc.feedback !== undefined) {
        document.getElementById('s-feedback').value = sc.feedback;
        document.getElementById('v-feedback').textContent = sc.feedback.toFixed(1);
    }
});

// ─── Tab switching ───────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
        if (currentData) resizeCharts();
    });
});

function resizeCharts() {
    ['chart-pv','chart-hemo','chart-renal','chart-coupling','chart-feedback'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) Plotly.Plots.resize(el);
    });
}
window.addEventListener('resize', resizeCharts);

// ─── Colors ──────────────────────────────────────────────────────────────
const COLORS = {
    teal:'#4ecdc4', coral:'#ff6b6b', gold:'#ffd93d', mint:'#a8e6cf',
    lilac:'#c9b1ff', sky:'#87ceeb', peach:'#ffb385'
};
function stepColor(i, n) {
    // Green → Yellow → Red gradient
    const t = n > 1 ? i / (n - 1) : 0;
    const r = Math.round(80 + 175 * t);
    const g = Math.round(220 - 140 * t);
    const b = Math.round(120 - 60 * t);
    return `rgb(${r},${g},${b})`;
}

const DARK_AXIS = { gridcolor:'#30363d', zerolinecolor:'#30363d', tickfont:{color:'#8b949e'}, titlefont:{color:'#e6edf3',size:12} };
const DARK_LAYOUT = {
    paper_bgcolor:'#0d1117', plot_bgcolor:'#161b22',
    font:{color:'#e6edf3', family:'-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif'},
    margin:{l:55,r:25,t:40,b:45},
    xaxis:{...DARK_AXIS}, yaxis:{...DARK_AXIS},
};

// ─── CKD staging ─────────────────────────────────────────────────────────
function ckdStage(gfr) {
    if (gfr >= 90) return {stage:'G1',cls:'good'};
    if (gfr >= 60) return {stage:'G2',cls:'good'};
    if (gfr >= 45) return {stage:'G3a',cls:'warn'};
    if (gfr >= 30) return {stage:'G3b',cls:'warn'};
    if (gfr >= 15) return {stage:'G4',cls:'bad'};
    return {stage:'G5',cls:'bad'};
}
function ddGrade(lvedp) {
    if (lvedp <= 12) return {grade:'Normal',cls:'good'};
    if (lvedp <= 16) return {grade:'Grade I',cls:'warn'};
    if (lvedp <= 22) return {grade:'Grade II',cls:'bad'};
    return {grade:'Grade III',cls:'bad'};
}
function metricClass(val, good, warn) {
    if (typeof good === 'function') return good(val) ? 'good' : (warn(val) ? 'warn' : 'bad');
    return val >= good ? 'good' : (val >= warn ? 'warn' : 'bad');
}

// ─── Summary bar ─────────────────────────────────────────────────────────
function updateSummary(d) {
    const n = d.steps.length;
    const last = n - 1;
    const ckd = ckdStage(d.GFR[last]);
    const dd = ddGrade(d.LVEDP[last]);
    const html = [
        mkMetric('EF', d.EF[last].toFixed(0)+'%', d.EF[0].toFixed(0)+'%&rarr;', metricClass(d.EF[last], 50, 40)),
        mkMetric('CO', d.CO[last].toFixed(1)+' L/m', d.CO[0].toFixed(1)+'&rarr;', metricClass(d.CO[last], 4.0, 3.0)),
        mkMetric('MAP', d.MAP[last].toFixed(0)+' mmHg', d.MAP[0].toFixed(0)+'&rarr;', metricClass(d.MAP[last], 70, 60)),
        mkMetric('LVEDP', d.LVEDP[last].toFixed(0)+' mmHg', dd.grade, dd.cls),
        mkMetric('GFR', d.GFR[last].toFixed(0)+' mL/m', ckd.stage, ckd.cls),
        mkMetric('Volume', d.V_blood[last].toFixed(0)+' mL', d.V_blood[0].toFixed(0)+'&rarr;',
            d.V_blood[last] < 5500 ? 'good' : d.V_blood[last] < 6500 ? 'warn' : 'bad'),
        mkMetric('CVP', d.CVP[last].toFixed(0)+' mmHg', d.CVP[0].toFixed(0)+'&rarr;',
            d.CVP[last] < 8 ? 'good' : d.CVP[last] < 15 ? 'warn' : 'bad'),
    ].join('');
    document.getElementById('summary').innerHTML = html;
}
function mkMetric(label, value, sub, cls) {
    return `<div class="metric"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub}</div></div>`;
}

// ─── Run simulation ──────────────────────────────────────────────────────
async function runSimulation() {
    const btn = document.getElementById('run-btn');
    const loader = document.getElementById('loading');
    btn.disabled = true;
    loader.classList.add('show');

    const scenarioKey = document.getElementById('scenario').value;
    const nSteps = parseInt(document.getElementById('s-steps').value);
    const sc = SCENARIOS[scenarioKey];

    // Build schedules
    let stiffSched, kfSched;
    if (sc.stiff && scenarioKey !== 'custom') {
        stiffSched = sc.stiff;
        kfSched = sc.kf;
    } else {
        const finalStiff = parseFloat(document.getElementById('s-stiff').value);
        const finalKf = parseFloat(document.getElementById('s-kf').value);
        stiffSched = Array.from({length: nSteps}, (_, i) => 1.0 + (finalStiff - 1.0) * i / Math.max(nSteps - 1, 1));
        kfSched = Array.from({length: nSteps}, (_, i) => 1.0 + (finalKf - 1.0) * i / Math.max(nSteps - 1, 1));
    }

    const params = {
        n_steps: nSteps,
        dt_hours: parseFloat(document.getElementById('s-dt').value),
        stiffness_schedule: stiffSched,
        kf_schedule: kfSched,
        arterial_stiffness: parseFloat(document.getElementById('s-artstiff').value),
        coupling_intensity: parseFloat(document.getElementById('s-coupling').value),
        raas_gain: parseFloat(document.getElementById('s-raas').value),
        tgf_gain: parseFloat(document.getElementById('s-tgf').value),
        na_intake: parseFloat(document.getElementById('s-na').value),
        feedback_rate: parseFloat(document.getElementById('s-feedback').value),
    };

    try {
        const resp = await fetch('/api/simulate', {
            method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(params)
        });
        const data = await resp.json();
        if (data.error) { alert('Simulation error: ' + data.error); return; }
        currentData = data;
        const d = data.coupled;
        updateSummary(d);
        renderPV(d);
        renderHemo(d);
        renderRenal(d);
        renderCoupling(d);
        renderFeedback(d);
    } catch (e) {
        alert('Request failed: ' + e.message);
    } finally {
        btn.disabled = false;
        loader.classList.remove('show');
    }
}

// ─── PV Loops ────────────────────────────────────────────────────────────
function renderPV(d) {
    const n = d.steps.length;
    const traces = d.pv_loops.map((pv, i) => ({
        x: pv.V, y: pv.P, mode: 'lines', name: `Step ${i+1} (stiff=${d.stiffness[i].toFixed(1)})`,
        line: {color: stepColor(i, n), width: i===0?3:2, dash: i===0?'solid':'solid'},
        opacity: 0.85,
    }));

    // Add ESPVR line
    const E_max = 2.3, V_0 = 10;
    const vEsp = [V_0, 80];
    traces.push({x:vEsp, y:vEsp.map(v=>(v-V_0)*E_max), mode:'lines', name:'ESPVR',
        line:{color:'#ffffff',width:1,dash:'dash'}, opacity:0.3});

    // Add EDPVR curves for first and last stiffness
    const vEd = [];
    for (let v = V_0; v <= 160; v += 2) vEd.push(v);
    const stiffFirst = d.stiffness[0], stiffLast = d.stiffness[n-1];
    traces.push({x:vEd, y:vEd.map(v=>stiffFirst*0.5*(Math.exp(Math.min(0.028*(v-V_0),15))-1)),
        mode:'lines', name:`EDPVR (stiff=${stiffFirst.toFixed(1)})`,
        line:{color:stepColor(0,n), width:1, dash:'dot'}, opacity:0.4});
    if (stiffLast !== stiffFirst) {
        traces.push({x:vEd, y:vEd.map(v=>stiffLast*0.5*(Math.exp(Math.min(0.028*(v-V_0),15))-1)),
            mode:'lines', name:`EDPVR (stiff=${stiffLast.toFixed(1)})`,
            line:{color:stepColor(n-1,n), width:1, dash:'dot'}, opacity:0.4});
    }

    Plotly.react('chart-pv', traces, {
        ...DARK_LAYOUT,
        title:{text:'LV Pressure\u2013Volume Loops: HFpEF Progression', font:{size:15,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS, title:'Volume [mL]'},
        yaxis:{...DARK_AXIS, title:'Pressure [mmHg]'},
        legend:{font:{size:10,color:'#8b949e'}, bgcolor:'rgba(22,27,34,0.8)', bordercolor:'#30363d'},
        showlegend: true,
    }, {responsive:true});
}

// ─── Hemodynamics ────────────────────────────────────────────────────────
function renderHemo(d) {
    const steps = d.steps;
    const mk = (y,name,color,dash) => ({x:steps,y,mode:'lines+markers',name,
        line:{color,width:2,dash:dash||'solid'},marker:{size:6}});
    const ref = (val,name) => ({x:[steps[0],steps[steps.length-1]],y:[val,val],mode:'lines',name,
        line:{color:'#8b949e',width:1,dash:'dot'},showlegend:false});

    const traces = [
        // Row 1
        mk(d.SBP,'SBP',COLORS.coral), mk(d.DBP,'DBP',COLORS.teal), mk(d.MAP,'MAP',COLORS.gold,'dash'),
        ref(120,''), ref(80,''),
        // Row 2
        mk(d.CO,'CO',COLORS.gold), ref(5,''),
        mk(d.SV,'SV',COLORS.sky),
        // Row 3
        mk(d.EF,'EF',COLORS.peach), ref(50,''),
        mk(d.LVEDP,'LVEDP',COLORS.lilac), ref(12,''),
    ];

    const layout = {
        ...DARK_LAYOUT,
        grid:{rows:3,columns:2,pattern:'independent',roworder:'top to bottom'},
        height: 700,
        title:{text:'Cardiac Hemodynamic Trends',font:{size:15,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS, title:'Step'},  yaxis:{...DARK_AXIS, title:'mmHg'},
        xaxis2:{...DARK_AXIS, title:'Step'}, yaxis2:{...DARK_AXIS, title:'L/min'},
        xaxis3:{...DARK_AXIS, title:'Step'}, yaxis3:{...DARK_AXIS, title:'mL'},
        xaxis4:{...DARK_AXIS, title:'Step'}, yaxis4:{...DARK_AXIS, title:'%'},
        xaxis5:{...DARK_AXIS, title:'Step'}, yaxis5:{...DARK_AXIS, title:'mmHg'},
        xaxis6:{...DARK_AXIS, title:'Step'}, yaxis6:{...DARK_AXIS, title:'mmHg'},
        showlegend: false,
        annotations: [
            {text:'Blood Pressure',xref:'x axis',yref:'y axis',x:0.25,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Cardiac Output',x:0.75,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Stroke Volume',x:0.25,y:0.64,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Ejection Fraction',x:0.75,y:0.64,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'LVEDP (Diastolic Dysfunction)',x:0.25,y:0.30,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'CVP (Venous Congestion)',x:0.75,y:0.30,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
        ],
    };

    // Assign traces to subplots
    traces[0].xaxis='x'; traces[0].yaxis='y';  // SBP
    traces[1].xaxis='x'; traces[1].yaxis='y';  // DBP
    traces[2].xaxis='x'; traces[2].yaxis='y';  // MAP
    traces[3].xaxis='x'; traces[3].yaxis='y';  // ref 120
    traces[4].xaxis='x'; traces[4].yaxis='y';  // ref 80

    traces[5].xaxis='x2'; traces[5].yaxis='y2'; // CO
    traces[6].xaxis='x2'; traces[6].yaxis='y2'; // ref 5

    traces[7].xaxis='x3'; traces[7].yaxis='y3'; // SV

    traces[8].xaxis='x4'; traces[8].yaxis='y4'; // EF
    traces[9].xaxis='x4'; traces[9].yaxis='y4'; // ref 50

    traces[10].xaxis='x5'; traces[10].yaxis='y5'; // LVEDP
    traces[11].xaxis='x5'; traces[11].yaxis='y5'; // ref 12

    // CVP trace
    const cvpTrace = mk(d.CVP,'CVP',COLORS.mint);
    cvpTrace.xaxis='x6'; cvpTrace.yaxis='y6';
    traces.push(cvpTrace);
    const cvpRef = ref(8,'');
    cvpRef.xaxis='x6'; cvpRef.yaxis='y6';
    traces.push(cvpRef);

    Plotly.react('chart-hemo', traces, layout, {responsive:true});
}

// ─── Renal function ──────────────────────────────────────────────────────
function renderRenal(d) {
    const steps = d.steps;
    const mk = (y,name,color) => ({x:steps,y,mode:'lines+markers',name,
        line:{color,width:2.5},marker:{size:7}});
    const ref = (val,name) => ({x:[steps[0],steps[steps.length-1]],y:[val,val],mode:'lines',name,
        line:{color:'#8b949e',width:1,dash:'dot'},showlegend:true});

    // CKD stage shading for GFR plot
    const ckdShapes = [
        {type:'rect',xref:'x',yref:'y',x0:steps[0]-0.5,x1:steps[steps.length-1]+0.5,y0:90,y1:150,fillcolor:'rgba(168,230,207,0.08)',line:{width:0}},
        {type:'rect',xref:'x',yref:'y',x0:steps[0]-0.5,x1:steps[steps.length-1]+0.5,y0:60,y1:90,fillcolor:'rgba(255,217,61,0.06)',line:{width:0}},
        {type:'rect',xref:'x',yref:'y',x0:steps[0]-0.5,x1:steps[steps.length-1]+0.5,y0:30,y1:60,fillcolor:'rgba(255,179,133,0.06)',line:{width:0}},
        {type:'rect',xref:'x',yref:'y',x0:steps[0]-0.5,x1:steps[steps.length-1]+0.5,y0:0,y1:30,fillcolor:'rgba(255,107,107,0.08)',line:{width:0}},
    ];
    const ckdAnnotations = [
        {text:'G1',x:steps[steps.length-1]+0.3,y:120,xref:'x',yref:'y',showarrow:false,font:{size:9,color:'#a8e6cf'}},
        {text:'G2',x:steps[steps.length-1]+0.3,y:75,xref:'x',yref:'y',showarrow:false,font:{size:9,color:'#ffd93d'}},
        {text:'G3',x:steps[steps.length-1]+0.3,y:45,xref:'x',yref:'y',showarrow:false,font:{size:9,color:'#ffb385'}},
        {text:'G4-5',x:steps[steps.length-1]+0.3,y:15,xref:'x',yref:'y',showarrow:false,font:{size:9,color:'#ff6b6b'}},
    ];

    const traces = [
        mk(d.GFR,'GFR',COLORS.lilac),
        mk(d.V_blood,'Blood Volume',COLORS.mint),
        ref(5000,'Baseline'),
        mk(d.Na_excr,'Na Excretion',COLORS.peach),
        ref(150,'Na Intake'),
        mk(d.P_glom,'P_glom',COLORS.sky),
    ];

    traces[0].xaxis='x'; traces[0].yaxis='y';
    traces[1].xaxis='x2'; traces[1].yaxis='y2';
    traces[2].xaxis='x2'; traces[2].yaxis='y2';
    traces[3].xaxis='x3'; traces[3].yaxis='y3';
    traces[4].xaxis='x3'; traces[4].yaxis='y3';
    traces[5].xaxis='x4'; traces[5].yaxis='y4';

    // Deterioration params — show prescribed (dashed) vs total (solid) when feedback data exists
    if (d.stiffness_prescribed) {
        var sp = mk(d.stiffness_prescribed,'Stiffness (prescribed)',COLORS.coral);
        sp.line = {color:COLORS.coral,width:1.5,dash:'dash'}; sp.xaxis='x5'; sp.yaxis='y5';
        traces.push(sp);
        var st = mk(d.stiffness_total,'Stiffness (total)',COLORS.coral);
        st.xaxis='x5'; st.yaxis='y5'; traces.push(st);
        var kp = mk(d.kf_prescribed,'Kf (prescribed)',COLORS.teal);
        kp.line = {color:COLORS.teal,width:1.5,dash:'dash'}; kp.xaxis='x5'; kp.yaxis='y5';
        traces.push(kp);
        var kt = mk(d.kf_total,'Kf (total)',COLORS.teal);
        kt.xaxis='x5'; kt.yaxis='y5'; traces.push(kt);
    } else {
        var s6 = mk(d.stiffness,'Stiffness',COLORS.coral); s6.xaxis='x5'; s6.yaxis='y5'; traces.push(s6);
        var s7 = mk(d.kf,'Kf_scale',COLORS.teal); s7.xaxis='x5'; s7.yaxis='y5'; traces.push(s7);
    }

    // Add RBF trace
    const rbfTrace = mk(d.RBF,'RBF',COLORS.gold);
    rbfTrace.xaxis='x6'; rbfTrace.yaxis='y6';
    traces.push(rbfTrace);

    Plotly.react('chart-renal', traces, {
        ...DARK_LAYOUT,
        grid:{rows:3,columns:2,pattern:'independent'},
        height:700,
        title:{text:'Renal Function & Deterioration Parameters',font:{size:15,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'mL/min',range:[0,Math.max(150,...d.GFR)*1.1]},
        xaxis2:{...DARK_AXIS,title:'Step'}, yaxis2:{...DARK_AXIS,title:'mL'},
        xaxis3:{...DARK_AXIS,title:'Step'}, yaxis3:{...DARK_AXIS,title:'mEq/day'},
        xaxis4:{...DARK_AXIS,title:'Step'}, yaxis4:{...DARK_AXIS,title:'mmHg'},
        xaxis5:{...DARK_AXIS,title:'Step'}, yaxis5:{...DARK_AXIS,title:'Scale (1=healthy)',range:[0,Math.max(4,...d.stiffness)*1.1]},
        xaxis6:{...DARK_AXIS,title:'Step'}, yaxis6:{...DARK_AXIS,title:'mL/min'},
        showlegend:false,
        shapes: ckdShapes,
        annotations: [
            ...ckdAnnotations,
            {text:'GFR (with CKD staging)',x:0.25,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Blood Volume',x:0.75,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Na Excretion',x:0.25,y:0.64,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Glomerular Pressure',x:0.75,y:0.64,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Deterioration Parameters',x:0.25,y:0.30,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Renal Blood Flow',x:0.75,y:0.30,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
        ],
    }, {responsive:true});
}

// ─── Coupling analysis ──────────────────────────────────────────────────
function renderCoupling(d) {
    const steps = d.steps;
    const mk = (y,name,color,axis) => {
        const t = {x:steps,y,mode:'lines+markers',name,line:{color,width:2},marker:{size:6}};
        if (axis) { t.xaxis=axis[0]; t.yaxis=axis[1]; }
        return t;
    };

    const traces = [
        // H→K messages
        mk(d.h2k_MAP,'H\u2192K MAP',COLORS.coral,['x','y']),
        mk(d.MAP,'Actual MAP','rgba(255,107,107,0.4)',['x','y']),
        mk(d.h2k_CO,'H\u2192K CO',COLORS.gold,['x2','y2']),
        mk(d.CO,'Actual CO','rgba(255,217,61,0.4)',['x2','y2']),
        mk(d.h2k_CVP,'H\u2192K CVP',COLORS.mint,['x3','y3']),
        mk(d.CVP,'Actual CVP','rgba(168,230,207,0.4)',['x3','y3']),
        // K→H messages
        mk(d.k2h_Vblood,'K\u2192H V_blood',COLORS.lilac,['x4','y4']),
        mk(d.V_blood,'Actual V_blood','rgba(201,177,255,0.4)',['x4','y4']),
        mk(d.k2h_SVR,'K\u2192H SVR_ratio',COLORS.peach,['x5','y5']),
        // Coupling intensity annotation
        mk(d.steps.map(()=>parseFloat(document.getElementById('s-coupling').value)),'Coupling \u03B1',COLORS.sky,['x6','y6']),
    ];

    Plotly.react('chart-coupling', traces, {
        ...DARK_LAYOUT,
        grid:{rows:3,columns:2,pattern:'independent'},
        height:700,
        title:{text:'Message Passing: Heart \u2194 Kidney Coupling',font:{size:15,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'MAP [mmHg]'},
        xaxis2:{...DARK_AXIS,title:'Step'}, yaxis2:{...DARK_AXIS,title:'CO [L/min]'},
        xaxis3:{...DARK_AXIS,title:'Step'}, yaxis3:{...DARK_AXIS,title:'CVP [mmHg]'},
        xaxis4:{...DARK_AXIS,title:'Step'}, yaxis4:{...DARK_AXIS,title:'V_blood [mL]'},
        xaxis5:{...DARK_AXIS,title:'Step'}, yaxis5:{...DARK_AXIS,title:'SVR ratio'},
        xaxis6:{...DARK_AXIS,title:'Step'}, yaxis6:{...DARK_AXIS,title:'\u03B1',range:[0,2.5]},
        showlegend:true,
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)',bordercolor:'#30363d',x:1.02,y:1},
        annotations:[
            {text:'Heart \u2192 Kidney: MAP',x:0.25,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.coral}},
            {text:'Heart \u2192 Kidney: CO',x:0.75,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.gold}},
            {text:'Heart \u2192 Kidney: CVP',x:0.25,y:0.64,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.mint}},
            {text:'Kidney \u2192 Heart: Volume',x:0.75,y:0.64,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.lilac}},
            {text:'Kidney \u2192 Heart: SVR',x:0.25,y:0.30,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.peach}},
            {text:'Coupling Intensity (\u03B1)',x:0.75,y:0.30,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.sky}},
        ],
    }, {responsive:true});
}

// ─── Feedback Loop visualization ────────────────────────────────────────
function renderFeedback(d) {
    if (!d.stiffness_prescribed) return;
    const steps = d.steps;
    const mk = function(y,name,color,axis,dash) {
        var t = {x:steps,y:y,mode:'lines+markers',name:name,
            line:{color:color,width:2,dash:dash||'solid'},marker:{size:6}};
        if (axis) { t.xaxis=axis[0]; t.yaxis=axis[1]; }
        return t;
    };
    function round4(v) { return Math.round(v*10000)/10000; }

    // Subplot 1 (top-left): Stiffness decomposition
    var traces = [
        mk(d.stiffness_prescribed,'Prescribed',COLORS.coral,['x','y']),
        mk(d.stiffness_feedback,'Emergent (feedback)',COLORS.gold,['x','y'],'dash'),
        mk(d.stiffness_total,'Total effective',COLORS.peach,['x','y']),
    ];

    // Subplot 2 (top-right): Kf decomposition
    traces.push(mk(d.kf_prescribed,'Prescribed Kf',COLORS.teal,['x2','y2']));
    traces.push(mk(d.kf_feedback_loss,'Emergent loss',COLORS.coral,['x2','y2'],'dash'));
    traces.push(mk(d.kf_total,'Effective Kf',COLORS.mint,['x2','y2']));

    // Subplot 3 (bottom-left): Stress indices
    traces.push(mk(d.cardiac_stress,'Cardiac stress (K\u2192H)',COLORS.coral,['x3','y3']));
    traces.push(mk(d.renal_stress,'Renal stress (H\u2192K)',COLORS.teal,['x3','y3']));

    // Subplot 4 (bottom-right): Rate of damage per step (acceleration)
    var stiffRate = d.stiffness_feedback.map(function(v,i) {
        return i === 0 ? 0 : round4(v - d.stiffness_feedback[i-1]);
    });
    var kfRate = d.kf_feedback_loss.map(function(v,i) {
        return i === 0 ? 0 : round4(v - d.kf_feedback_loss[i-1]);
    });
    traces.push(mk(stiffRate,'Stiffness gain/step',COLORS.coral,['x4','y4']));
    traces.push(mk(kfRate,'Kf loss/step',COLORS.teal,['x4','y4']));

    Plotly.react('chart-feedback', traces, {
        ...DARK_LAYOUT,
        grid:{rows:2,columns:2,pattern:'independent'},
        height:700,
        title:{text:'Feedback Loop: Emergent Cross-Organ Damage',font:{size:15,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'Stiffness scale'},
        xaxis2:{...DARK_AXIS,title:'Step'}, yaxis2:{...DARK_AXIS,title:'Kf scale'},
        xaxis3:{...DARK_AXIS,title:'Step'}, yaxis3:{...DARK_AXIS,title:'Stress index'},
        xaxis4:{...DARK_AXIS,title:'Step'}, yaxis4:{...DARK_AXIS,title:'Rate / step'},
        showlegend:true,
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)',bordercolor:'#30363d'},
        annotations:[
            {text:'Stiffness: Prescribed vs Emergent',x:0.25,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.coral}},
            {text:'Kf: Prescribed vs Emergent Loss',x:0.75,y:1.02,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:COLORS.teal}},
            {text:'Cross-Organ Stress Indices',x:0.25,y:0.46,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
            {text:'Damage Acceleration (rate/step)',x:0.75,y:0.46,xref:'paper',yref:'paper',showarrow:false,font:{size:12,color:'#e6edf3'}},
        ],
    }, {responsive:true});
}

// ─── Validation / Scenario Analysis ─────────────────────────────────────
const VAL_COLORS = {heart:'#ff6b6b', kidney:'#4ecdc4', combined:'#ffd93d'};
async function runValidation() {
    const btn = document.getElementById('validate-btn');
    const status = document.getElementById('validate-status');
    btn.disabled = true;
    status.textContent = 'Running all 3 scenarios...';
    try {
        const resp = await fetch('/api/validate', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
        const data = await resp.json();
        if (data.error) { status.textContent = 'Error: ' + data.error; return; }
        document.getElementById('validate-results').style.display = 'block';
        renderValidationTable(data.expectations);
        renderValidationCharts(data);
        status.textContent = 'Done \u2014 scroll down for comparison charts.';
    } catch(e) {
        status.textContent = 'Request failed: ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

function renderValidationTable(exp) {
    const sections = [
        {key:'heart_only', title:'\u2764 Heart-Only (Isolated HFpEF)', color:VAL_COLORS.heart},
        {key:'kidney_only', title:'\u25C6 Kidney-Only (CKD \u2192 Overload)', color:VAL_COLORS.kidney},
        {key:'combined', title:'\u2605 Combined (HFpEF + CKD)', color:VAL_COLORS.combined},
    ];
    let html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">';
    for (const sec of sections) {
        const checks = exp[sec.key];
        const passed = checks.filter(c=>c.passed).length;
        const total = checks.length;
        const pct = Math.round(100*passed/total);
        html += '<div style="background:var(--bg1);border:1px solid var(--border);border-radius:10px;padding:16px;">';
        html += '<h4 style="color:' + sec.color + ';font-size:13px;margin-bottom:10px;">' + sec.title + '</h4>';
        html += '<div style="font-size:12px;color:var(--t2);margin-bottom:8px;">Conformity: <b style="color:' + (pct>=80?'var(--mint)':pct>=50?'var(--gold)':'var(--coral)') + '">' + passed + '/' + total + ' (' + pct + '%)</b></div>';
        html += '<table style="width:100%;font-size:11px;border-collapse:collapse;">';
        html += '<tr style="color:var(--t2);border-bottom:1px solid var(--border);"><th style="text-align:left;padding:4px;">Metric</th><th style="text-align:left;padding:4px;">Expected</th><th style="text-align:left;padding:4px;">Actual</th><th style="padding:4px;">\u2713/\u2717</th></tr>';
        for (const c of checks) {
            const icon = c.passed ? '<span style="color:var(--mint);">\u2713</span>' : '<span style="color:var(--coral);">\u2717</span>';
            html += '<tr style="border-bottom:1px solid rgba(48,54,61,0.5);">';
            html += '<td style="padding:4px;color:var(--t1);font-weight:600;">' + c.metric + '</td>';
            html += '<td style="padding:4px;color:var(--t2);">' + c.expected + '</td>';
            html += '<td style="padding:4px;color:var(--t1);">' + c.actual + '</td>';
            html += '<td style="padding:4px;text-align:center;">' + icon + '</td></tr>';
        }
        html += '</table></div>';
    }
    html += '</div>';
    document.getElementById('validation-table').innerHTML = html;
}

function renderValidationCharts(data) {
    const h = data.heart_only, k = data.kidney_only, co = data.combined;
    const steps = h.steps;
    const mk = function(d,name,color,dash) {
        return {x:steps,y:d,mode:'lines+markers',name:name,line:{color:color,width:2.5,dash:dash||'solid'},marker:{size:6}};
    };
    const refLine = function(y0,name,color) {
        return {x:[steps[0],steps[steps.length-1]],y:[y0,y0],mode:'lines',name:name,
            line:{color:color||'#8b949e',width:1,dash:'dot'},showlegend:true};
    };

    // PV loops: last step of each scenario + baseline
    Plotly.react('chart-val-pv', [
        {x:h.pv_loops[0].V, y:h.pv_loops[0].P, mode:'lines',name:'Baseline (healthy)',line:{color:'#8b949e',width:1.5,dash:'dot'}},
        {x:h.pv_loops[h.pv_loops.length-1].V, y:h.pv_loops[h.pv_loops.length-1].P, mode:'lines',name:'Heart-Only (final)',line:{color:VAL_COLORS.heart,width:2.5}},
        {x:k.pv_loops[k.pv_loops.length-1].V, y:k.pv_loops[k.pv_loops.length-1].P, mode:'lines',name:'Kidney-Only (final)',line:{color:VAL_COLORS.kidney,width:2.5}},
        {x:co.pv_loops[co.pv_loops.length-1].V, y:co.pv_loops[co.pv_loops.length-1].P, mode:'lines',name:'Combined (final)',line:{color:VAL_COLORS.combined,width:2.5}},
    ], {...DARK_LAYOUT,
        title:{text:'Final PV Loops: All Scenarios vs Baseline',font:{size:14,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Volume [mL]'}, yaxis:{...DARK_AXIS,title:'Pressure [mmHg]'},
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
    },{responsive:true});

    // GFR
    Plotly.react('chart-val-gfr', [
        mk(h.GFR,'Heart-Only',VAL_COLORS.heart), mk(k.GFR,'Kidney-Only',VAL_COLORS.kidney),
        mk(co.GFR,'Combined',VAL_COLORS.combined),
        refLine(90,'CKD G2'), refLine(60,'CKD G3','#ff6b6b'),
    ], {...DARK_LAYOUT,
        title:{text:'GFR Trajectories',font:{size:14,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'GFR [mL/min]',range:[0,150]},
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
        shapes:[{type:'rect',xref:'paper',yref:'y',x0:0,x1:1,y0:0,y1:60,fillcolor:'rgba(255,107,107,0.05)',line:{width:0}}],
    },{responsive:true});

    // LVEDP
    Plotly.react('chart-val-lvedp', [
        mk(h.LVEDP,'Heart-Only',VAL_COLORS.heart), mk(k.LVEDP,'Kidney-Only',VAL_COLORS.kidney),
        mk(co.LVEDP,'Combined',VAL_COLORS.combined),
        refLine(12,'Normal limit'), refLine(18,'Grade II DD','#ff6b6b'),
    ], {...DARK_LAYOUT,
        title:{text:'LVEDP (Diastolic Dysfunction)',font:{size:14,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'LVEDP [mmHg]'},
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
    },{responsive:true});

    // CO
    Plotly.react('chart-val-co', [
        mk(h.CO,'Heart-Only',VAL_COLORS.heart), mk(k.CO,'Kidney-Only',VAL_COLORS.kidney),
        mk(co.CO,'Combined',VAL_COLORS.combined),
        refLine(4.0,'Low CO threshold','#ff6b6b'),
    ], {...DARK_LAYOUT,
        title:{text:'Cardiac Output',font:{size:14,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'CO [L/min]'},
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
    },{responsive:true});

    // V_blood
    Plotly.react('chart-val-vol', [
        mk(h.V_blood,'Heart-Only',VAL_COLORS.heart), mk(k.V_blood,'Kidney-Only',VAL_COLORS.kidney),
        mk(co.V_blood,'Combined',VAL_COLORS.combined),
        refLine(5000,'Baseline'),
    ], {...DARK_LAYOUT,
        title:{text:'Blood Volume (Fluid Retention)',font:{size:14,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'V_blood [mL]'},
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
    },{responsive:true});

    // CVP
    Plotly.react('chart-val-cvp', [
        mk(h.CVP,'Heart-Only',VAL_COLORS.heart), mk(k.CVP,'Kidney-Only',VAL_COLORS.kidney),
        mk(co.CVP,'Combined',VAL_COLORS.combined),
        refLine(8,'Congestion threshold','#ffd93d'),
    ], {...DARK_LAYOUT,
        title:{text:'Central Venous Pressure (Congestion)',font:{size:14,color:'#e6edf3'}},
        xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'CVP [mmHg]'},
        legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
    },{responsive:true});

    // Emergent feedback comparison (7th chart, full-width)
    if (h.stiffness_feedback) {
        Plotly.react('chart-val-feedback', [
            mk(h.stiffness_feedback,'Heart-Only: stiff fb',VAL_COLORS.heart),
            mk(k.stiffness_feedback,'Kidney-Only: stiff fb',VAL_COLORS.kidney),
            mk(co.stiffness_feedback,'Combined: stiff fb',VAL_COLORS.combined),
            mk(h.kf_feedback_loss,'Heart-Only: Kf loss',VAL_COLORS.heart,'dash'),
            mk(k.kf_feedback_loss,'Kidney-Only: Kf loss',VAL_COLORS.kidney,'dash'),
            mk(co.kf_feedback_loss,'Combined: Kf loss',VAL_COLORS.combined,'dash'),
        ], {...DARK_LAYOUT,
            title:{text:'Emergent Cross-Organ Damage (solid = stiffness gain, dashed = Kf loss)',font:{size:14,color:'#e6edf3'}},
            xaxis:{...DARK_AXIS,title:'Step'}, yaxis:{...DARK_AXIS,title:'Accumulated feedback'},
            legend:{font:{size:10,color:'#8b949e'},bgcolor:'rgba(22,27,34,0.8)'},height:400,
        },{responsive:true});
    }
}

// ─── Source code ─────────────────────────────────────────────────────────
async function loadSource() {
    try {
        const resp = await fetch('/api/source');
        const data = await resp.json();
        const el = document.getElementById('source-code');
        el.textContent = data.code;
    } catch(e) {
        document.getElementById('source-code').textContent = '# Failed to load source code: ' + e.message;
    }
}
loadSource();

// ─── Auto-run default scenario on load ───────────────────────────────────
window.addEventListener('load', () => { setTimeout(runSimulation, 300); });
</script>
</body>
</html>
"""  # noqa: end _OLD

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n  Cardiorenal Syndrome Dashboard")
    print("  Open http://localhost:8050 in your browser\n")
    app.run(debug=False, port=8050)
