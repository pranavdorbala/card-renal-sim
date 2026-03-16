#!/usr/bin/env python3
"""
Synthetic Cohort Generator: ARIC Visit 5 → Visit 7 Paired Data
===============================================================
Generates synthetic patient pairs using the full CircAdapt + Hallow coupled
model with emission_functions.py for 80+ ARIC-compatible variables.

Usage:
    python synthetic_cohort.py --n_patients 10000 --n_workers 8
    python synthetic_cohort.py --n_patients 100 --n_workers 1  # quick test
"""

import os
import sys
import argparse
import time
import warnings
import copy
import numpy as np
from dataclasses import asdict
from multiprocessing import Pool
from typing import Dict, Tuple, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TUNABLE_PARAMS, NUMERIC_VAR_NAMES, NON_NUMERIC_VARS, COHORT_DEFAULTS

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# Stabilized Renal Model (adapted from dashboard.py with better TGF damping)
# ═══════════════════════════════════════════════════════════════════════════

def _update_renal_stable(r, MAP, CO, Pven, dt_hours=6.0):
    """
    Hallow renal model update with improved TGF loop damping.
    Adapted from dashboard.py's update_renal with damping factor 0.85/0.15
    (vs original 0.6/0.4) for stability at CircAdapt's MAP range (~86-90).
    Modifies r in-place.
    """
    Kf_eff = r['Kf'] * r['Kf_scale']
    r['P_renal_vein'] = max(Pven, 2.0)

    # 1. RAAS
    dMAP = MAP - r['MAP_setpoint']
    RAAS_factor = float(np.clip(1.0 - r['RAAS_gain'] * 0.005 * dMAP, 0.5, 2.0))
    R_EA = r['R_EA0'] * RAAS_factor
    eta_CD = r['eta_CD0'] * RAAS_factor

    # 2. TGF iteration with improved damping
    R_AA = r['R_AA0']
    GFR = r.get('GFR', 120.0)
    Na_filt = 0.0
    P_gc = 60.0
    RBF = 1100.0

    for _ in range(20):  # more iterations for convergence
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
        R_AA = 0.85 * R_AA + 0.15 * R_AA_new  # increased damping

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


def _create_renal_state_circadapt(na_intake=150.0, raas_gain=1.5, tgf_gain=2.0, kf_scale=1.0):
    """Create renal state dict calibrated for CircAdapt MAP (~86 mmHg)."""
    return {
        'N_nephrons': 1e6, 'Kf': 8.0,
        'R_preAA': 9.5, 'R_AA0': 20.5, 'R_EA0': 43.0,
        'P_Bow': 18.0, 'P_renal_vein': 4.0,
        'pi_plasma': 25.0, 'Hct': 0.45,
        'eta_PT': 0.67, 'eta_LoH': 0.25, 'eta_DT': 0.05, 'eta_CD0': 0.024,
        'frac_water_reabs': 0.99,
        'Na_intake': na_intake, 'water_intake': 2.0,
        'TGF_gain': tgf_gain, 'TGF_setpoint': 0.0,
        'RAAS_gain': raas_gain, 'MAP_setpoint': 86.0,
        'V_blood': 5000.0, 'Na_total': 2100.0, 'C_Na': 140.0,
        'GFR': 120.0, 'RBF': 1100.0, 'P_glom': 60.0,
        'Na_excretion': 150.0, 'water_excretion': 1.5,
        'Kf_scale': kf_scale,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Core Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_patient_state(
    params: Dict,
    demographics: Dict,
    n_coupling_steps: int = 2,
    dt_renal_hours: float = 6.0,
) -> Optional[Dict]:
    """
    Run CircAdapt heart + stabilized Hallow renal model coupled together.
    Returns 113 ARIC variables (cardiac from CircAdapt waveforms, renal from
    stabilized Hallow model calibrated for CircAdapt MAP range), or None on failure.

    Parameters
    ----------
    params : dict with keys matching TUNABLE_PARAMS
        Sf_act_scale, Kf_scale, inflammation_scale, diabetes_scale,
        RAAS_gain, TGF_gain, na_intake
    demographics : dict
        age, sex ('M'/'F'), BSA (m²), height_m
    n_coupling_steps : int
        Not used (kept for API compatibility); renal equilibrated in 5 passes
    dt_renal_hours : float
        Renal model time step (hours)
    """
    from cardiorenal_coupling import (
        CircAdaptHeartModel, InflammatoryState,
        update_inflammatory_state, ML_TO_M3,
    )
    from emission_functions import extract_all_aric_variables

    try:
        # 1. Create heart model (CircAdapt for full waveforms)
        heart = CircAdaptHeartModel()
        ist = InflammatoryState()

        # 2. Create renal model calibrated for CircAdapt MAP (~86 mmHg)
        renal = _create_renal_state_circadapt(
            na_intake=params.get('na_intake', 150.0),
            raas_gain=params.get('RAAS_gain', 1.5),
            tgf_gain=params.get('TGF_gain', 2.0),
            kf_scale=params.get('Kf_scale', 1.0),
        )

        # 3. Update inflammatory state
        ist = update_inflammatory_state(
            ist,
            inflammation_scale=params.get('inflammation_scale', 0.0),
            diabetes_scale=params.get('diabetes_scale', 0.0),
        )

        # 4. Apply inflammatory modifiers to heart
        heart.apply_inflammatory_modifiers(ist)

        # 5. Apply diastolic stiffness (k1_scale × inflammatory k1 factor)
        effective_k1 = params.get('k1_scale', 1.0) * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # 6. Apply Sf_act deterioration
        effective_sf = max(
            params.get('Sf_act_scale', 1.0) * ist.Sf_act_factor, 0.20
        )
        heart.apply_deterioration(effective_sf)

        # 7. Run CircAdapt to steady state (cardiac solve with waveforms)
        hemo = heart.run_to_steady_state()

        # 8. Equilibrate renal model at the cardiac hemodynamics
        # Hold volume/Na/C_Na steady to find hemodynamic equilibrium (GFR, P_glom, RBF)
        # Reset TGF_setpoint each pass so it calibrates at C_Na=140
        for _ in range(5):
            renal['C_Na'] = 140.0
            renal['TGF_setpoint'] = 0.0  # re-calibrate each pass
            _update_renal_stable(renal, hemo['MAP'], hemo['CO'], hemo['Pven'], dt_renal_hours)
            renal['V_blood'] = 5000.0
            renal['Na_total'] = 2100.0
            renal['C_Na'] = 140.0

        # 9. Build renal_state dict for emission_functions
        # Na_excretion = Na_intake at true steady state (volume balance)
        renal_state = {
            'GFR': renal['GFR'],
            'V_blood': 5000.0,
            'C_Na': 140.0,
            'Na_excretion': params.get('na_intake', 150.0),  # steady-state: excr = intake
            'P_glom': renal['P_glom'],
            'Kf_scale': renal['Kf_scale'],
            'RBF': renal['RBF'],
        }

        # 10. Extract ARIC variables (cardiac from CircAdapt waveforms + renal)
        age = demographics.get('age', 75.0)
        sex = demographics.get('sex', 'M')
        BSA = demographics.get('BSA', 1.9)
        height_m = demographics.get('height_m', 1.70)

        aric_vars = extract_all_aric_variables(
            heart.model, renal_state,
            BSA=BSA, height_m=height_m, age=age, sex=sex,
        )

        return aric_vars

    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Patient Sampling
# ═══════════════════════════════════════════════════════════════════════════

# Correlation matrix for disease progression deltas
# Order: delta_Sf_act, delta_Kf, delta_inflammation, delta_diabetes, delta_RAAS, delta_na
# Negative delta_Sf_act = worsening contractility; negative delta_Kf = worsening kidney
_DELTA_CORR = np.array([
    [ 1.0, 0.4, -0.2, -0.1,  0.0,  0.0],  # Sf_act
    [ 0.4, 1.0, -0.3, -0.2, -0.1,  0.0],  # Kf
    [-0.2,-0.3,  1.0,  0.3,  0.2,  0.0],  # inflammation
    [-0.1,-0.2,  0.3,  1.0,  0.1,  0.0],  # diabetes
    [ 0.0,-0.1,  0.2,  0.1,  1.0,  0.1],  # RAAS
    [ 0.0, 0.0,  0.0,  0.0,  0.1,  1.0],  # na_intake
])
_DELTA_CHOL = np.linalg.cholesky(_DELTA_CORR)


def sample_correlated_deltas(rng: np.random.Generator) -> Dict:
    """Sample correlated disease progression deltas for V5→V7 (6 years)."""
    z = rng.standard_normal(6)
    corr_z = _DELTA_CHOL @ z

    # Map to meaningful deltas (mostly worsening over 6 years)
    delta_Sf_act = -abs(corr_z[0]) * 0.08          # small contractility decline
    delta_Kf = -abs(corr_z[1]) * 0.10              # small Kf decline
    delta_inflammation = abs(corr_z[2]) * 0.08      # inflammation tends to increase
    delta_diabetes = abs(corr_z[3]) * 0.05          # diabetes progresses slowly
    delta_RAAS = corr_z[4] * 0.15                   # can go either way
    delta_na = corr_z[5] * 15.0                     # dietary changes

    return {
        'delta_Sf_act': delta_Sf_act,
        'delta_Kf': delta_Kf,
        'delta_inflammation': delta_inflammation,
        'delta_diabetes': delta_diabetes,
        'delta_RAAS': delta_RAAS,
        'delta_na': delta_na,
    }


def generate_patient_params(rng: np.random.Generator) -> Dict:
    """Generate one patient's V5 and V7 parameter sets + demographics."""

    # Demographics
    age_v5 = rng.uniform(65, 85)
    sex = 'M' if rng.random() < 0.5 else 'F'
    BSA = float(np.clip(rng.normal(1.9, 0.15), 1.4, 2.5))
    height_m = float(np.clip(rng.normal(1.70, 0.08), 1.50, 1.95))

    demographics_v5 = {'age': age_v5, 'sex': sex, 'BSA': BSA, 'height_m': height_m}
    demographics_v7 = {'age': age_v5 + 6.0, 'sex': sex, 'BSA': BSA, 'height_m': height_m}

    # V5 baseline disease parameters
    Sf_act_v5 = float(np.clip(rng.normal(0.92, 0.12), 0.35, 1.0))
    Kf_v5 = float(np.clip(rng.beta(5, 2), 0.3, 1.0))
    inflammation_v5 = float(np.clip(rng.exponential(0.10), 0.0, 0.6))
    diabetes_v5 = float(rng.choice(
        [0.0, 0.0, 0.0, np.clip(rng.uniform(0.1, 0.7), 0.0, 1.0)]
    ))
    RAAS_v5 = float(np.clip(rng.normal(1.5, 0.3), 0.8, 2.5))
    TGF_v5 = float(np.clip(rng.normal(2.0, 0.3), 1.0, 3.5))
    na_v5 = float(np.clip(rng.normal(150, 40), 50, 300))
    # k1_scale: mostly normal, some elevated (correlated with diabetes)
    k1_v5 = 1.0 if diabetes_v5 == 0.0 else float(np.clip(1.0 + diabetes_v5 * 0.5, 1.0, 2.0))

    v5_params = {
        'Sf_act_scale': Sf_act_v5,
        'Kf_scale': Kf_v5,
        'inflammation_scale': inflammation_v5,
        'diabetes_scale': diabetes_v5,
        'k1_scale': k1_v5,
        'RAAS_gain': RAAS_v5,
        'TGF_gain': TGF_v5,
        'na_intake': na_v5,
    }

    # Correlated disease progression
    deltas = sample_correlated_deltas(rng)

    # k1 progression: tied to diabetes progression
    delta_k1 = abs(deltas['delta_diabetes']) * 0.3  # diabetes worsens → k1 rises

    v7_params = {
        'Sf_act_scale': float(np.clip(Sf_act_v5 + deltas['delta_Sf_act'],
                                       *TUNABLE_PARAMS['Sf_act_scale']['range'])),
        'Kf_scale': float(np.clip(Kf_v5 + deltas['delta_Kf'],
                                   *TUNABLE_PARAMS['Kf_scale']['range'])),
        'inflammation_scale': float(np.clip(inflammation_v5 + deltas['delta_inflammation'],
                                            *TUNABLE_PARAMS['inflammation_scale']['range'])),
        'diabetes_scale': float(np.clip(diabetes_v5 + deltas['delta_diabetes'],
                                        *TUNABLE_PARAMS['diabetes_scale']['range'])),
        'k1_scale': float(np.clip(k1_v5 + delta_k1,
                                   *TUNABLE_PARAMS['k1_scale']['range'])),
        'RAAS_gain': float(np.clip(RAAS_v5 + deltas['delta_RAAS'],
                                    *TUNABLE_PARAMS['RAAS_gain']['range'])),
        'TGF_gain': TGF_v5,
        'na_intake': float(np.clip(na_v5 + deltas['delta_na'],
                                    *TUNABLE_PARAMS['na_intake']['range'])),
    }

    return {
        'v5_params': v5_params,
        'v7_params': v7_params,
        'demographics_v5': demographics_v5,
        'demographics_v7': demographics_v7,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Worker Function for Multiprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _process_patient(args: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Process one patient: generate V5 and V7 ARIC variable vectors."""
    patient_idx, seed = args
    rng = np.random.default_rng(seed)
    patient = generate_patient_params(rng)

    v5_vars = evaluate_patient_state(
        patient['v5_params'], patient['demographics_v5'],
        n_coupling_steps=COHORT_DEFAULTS['n_coupling_steps'],
        dt_renal_hours=COHORT_DEFAULTS['dt_renal_hours'],
    )
    if v5_vars is None:
        return None

    v7_vars = evaluate_patient_state(
        patient['v7_params'], patient['demographics_v7'],
        n_coupling_steps=COHORT_DEFAULTS['n_coupling_steps'],
        dt_renal_hours=COHORT_DEFAULTS['dt_renal_hours'],
    )
    if v7_vars is None:
        return None

    # Convert to ordered numeric vectors
    v5_vec = np.array([float(v5_vars.get(k, 0.0)) for k in NUMERIC_VAR_NAMES])
    v7_vec = np.array([float(v7_vars.get(k, 0.0)) for k in NUMERIC_VAR_NAMES])

    # Check for NaN/Inf
    if np.any(~np.isfinite(v5_vec)) or np.any(~np.isfinite(v7_vec)):
        return None

    return (v5_vec, v7_vec, patient)


# ═══════════════════════════════════════════════════════════════════════════
# Cohort Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_cohort(
    n_patients: int = 10000,
    seed: int = 42,
    n_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Generate full synthetic cohort of V5/V7 paired ARIC variables.

    Returns
    -------
    v5_array : ndarray of shape (N_valid, N_features)
    v7_array : ndarray of shape (N_valid, N_features)
    var_names : list of str (column names)
    patient_metadata : list of dicts (demographics + params for each valid patient)
    """
    print(f"Generating {n_patients} synthetic patients (seed={seed}, workers={n_workers})...")
    rng_master = np.random.default_rng(seed)
    seeds = rng_master.integers(0, 2**31, size=n_patients)
    args_list = [(i, int(seeds[i])) for i in range(n_patients)]

    t0 = time.time()

    if n_workers <= 1:
        results = []
        for i, args in enumerate(args_list):
            result = _process_patient(args)
            results.append(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_patients - i - 1) / rate
                print(f"  [{i+1}/{n_patients}] {rate:.1f} pts/s, ETA {eta:.0f}s")
    else:
        with Pool(n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(_process_patient, args_list, chunksize=10)):
                results.append(result)
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (n_patients - i - 1) / rate
                    print(f"  [{i+1}/{n_patients}] {rate:.1f} pts/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0

    # Filter successful results
    valid_results = [r for r in results if r is not None]
    n_valid = len(valid_results)
    n_failed = n_patients - n_valid
    print(f"Done in {elapsed:.1f}s. Valid: {n_valid}/{n_patients} ({n_failed} failed)")

    v5_list, v7_list, meta_list = [], [], []
    for v5_vec, v7_vec, patient in valid_results:
        v5_list.append(v5_vec)
        v7_list.append(v7_vec)
        meta_list.append(patient)

    v5_array = np.stack(v5_list)
    v7_array = np.stack(v7_list)

    return v5_array, v7_array, NUMERIC_VAR_NAMES, meta_list


def load_real_aric_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load real ARIC V5/V7 paired data from a CSV file.

    Expected format: CSV with columns matching NUMERIC_VAR_NAMES,
    prefixed with 'v5_' and 'v7_' (e.g., 'v5_LVEF_pct', 'v7_LVEF_pct').

    Returns same format as generate_cohort().
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    v5_cols = [f'v5_{k}' for k in NUMERIC_VAR_NAMES]
    v7_cols = [f'v7_{k}' for k in NUMERIC_VAR_NAMES]

    # Check which columns exist
    missing_v5 = [c for c in v5_cols if c not in df.columns]
    missing_v7 = [c for c in v7_cols if c not in df.columns]
    if missing_v5 or missing_v7:
        available = [c for c in v5_cols if c in df.columns]
        print(f"Warning: {len(missing_v5)} V5 cols and {len(missing_v7)} V7 cols missing. "
              f"Using {len(available)} available columns, filling rest with 0.")

    v5_array = np.zeros((len(df), len(NUMERIC_VAR_NAMES)))
    v7_array = np.zeros((len(df), len(NUMERIC_VAR_NAMES)))

    for i, var_name in enumerate(NUMERIC_VAR_NAMES):
        v5_col = f'v5_{var_name}'
        v7_col = f'v7_{var_name}'
        if v5_col in df.columns:
            v5_array[:, i] = df[v5_col].values
        if v7_col in df.columns:
            v7_array[:, i] = df[v7_col].values

    return v5_array, v7_array, NUMERIC_VAR_NAMES, []


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic ARIC V5/V7 cohort')
    parser.add_argument('--n_patients', type=int, default=COHORT_DEFAULTS['n_patients'])
    parser.add_argument('--seed', type=int, default=COHORT_DEFAULTS['seed'])
    parser.add_argument('--n_workers', type=int, default=COHORT_DEFAULTS['n_workers'])
    parser.add_argument('--output', type=str, default='cohort_data.npz',
                        help='Output file path (.npz)')
    args = parser.parse_args()

    v5, v7, var_names, metadata = generate_cohort(
        n_patients=args.n_patients,
        seed=args.seed,
        n_workers=args.n_workers,
    )

    # Save
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    np.savez(
        outpath,
        v5=v5, v7=v7,
        var_names=np.array(var_names),
    )
    print(f"Saved to {outpath}: v5 {v5.shape}, v7 {v7.shape}, {len(var_names)} variables")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"  Cohort Summary ({v5.shape[0]} patients, {v5.shape[1]} variables)")
    print(f"{'='*60}")
    for i, name in enumerate(var_names[:10]):
        print(f"  {name:30s}  V5: {v5[:,i].mean():.2f} ± {v5[:,i].std():.2f}  "
              f"V7: {v7[:,i].mean():.2f} ± {v7[:,i].std():.2f}")
    print(f"  ... and {len(var_names) - 10} more variables")


if __name__ == '__main__':
    main()
