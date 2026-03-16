#!/usr/bin/env python3
"""
Configuration for the Cardiorenal V5→V7 Prediction + Agentic Framework
======================================================================
Shared constants, tunable parameter ranges, ARIC variable metadata,
clinical thresholds, and LLM configuration.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Tunable Disease Progression Parameters
# ═══════════════════════════════════════════════════════════════════════════
# These map to CircAdapt + Hallow + InflammatoryState in cardiorenal_coupling.py

TUNABLE_PARAMS = {
    'Sf_act_scale': {
        'range': (0.2, 1.0), 'default': 1.0,
        'desc': 'Active fiber stress scale (HFrEF: <1 reduces contractility). '
                'Maps to Patch[Sf_act] on LV/SV in CircAdapt.',
    },
    'Kf_scale': {
        'range': (0.05, 1.0), 'default': 1.0,
        'desc': 'Glomerular ultrafiltration coefficient (CKD: <1 = nephron loss, '
                'podocyte injury, mesangial expansion).',
    },
    'inflammation_scale': {
        'range': (0.0, 1.0), 'default': 0.0,
        'desc': 'Systemic inflammation index (0=none, 1=severe). Drives via '
                'InflammatoryState: Sf_act_factor, p0_factor (SVR), stiffness_factor '
                '(arterial), Kf_factor, R_AA_factor, RAAS_gain_factor, eta_PT_offset, '
                'MAP_setpoint_offset.',
    },
    'diabetes_scale': {
        'range': (0.0, 1.0), 'default': 0.0,
        'desc': 'Diabetes metabolic burden (0=none, 1=severe). Drives via '
                'InflammatoryState: passive_k1_factor (diastolic stiffness → HFpEF), '
                'stiffness_factor (AGE arterial), Kf_factor (biphasic), R_EA_factor, '
                'eta_PT_offset (SGLT2), MAP_setpoint_offset.',
    },
    'k1_scale': {
        'range': (1.0, 3.0), 'default': 1.0,
        'desc': 'Passive myocardial stiffness scale (HFpEF: >1 increases diastolic '
                'dysfunction). Maps to Patch[k1] on LV/SV in CircAdapt.',
    },
    'RAAS_gain': {
        'range': (0.5, 3.0), 'default': 1.5,
        'desc': 'RAAS sensitivity on HallowRenalModel. Higher = more reactive '
                'to MAP drops (renin → AngII → R_EA + aldosterone → CD reabsorption).',
    },
    'TGF_gain': {
        'range': (1.0, 4.0), 'default': 2.0,
        'desc': 'Tubuloglomerular feedback gain. Senses macula densa Na delivery '
                '→ adjusts afferent arteriole resistance.',
    },
    'na_intake': {
        'range': (50.0, 300.0), 'default': 150.0,
        'desc': 'Dietary sodium intake (mEq/day). Affects volume balance and '
                'pressure-natriuresis.',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# ARIC Variable Metadata
# ═══════════════════════════════════════════════════════════════════════════
# Keys match extract_all_aric_variables() output from emission_functions.py.
# 'weight' is the importance weight for the NN loss function.
# Variables with is_numeric=False are excluded from NN training.

ARIC_VARIABLES = {
    # ── LV Structure ──
    'LVIDd_cm':           {'cat': 'LV_structure',  'units': 'cm',     'normal': (4.2, 5.9), 'weight': 0.5},
    'LVIDs_cm':           {'cat': 'LV_structure',  'units': 'cm',     'normal': (2.5, 4.0), 'weight': 0.5},
    'IVSd_cm':            {'cat': 'LV_structure',  'units': 'cm',     'normal': (0.6, 1.1), 'weight': 0.5},
    'LVPWd_cm':           {'cat': 'LV_structure',  'units': 'cm',     'normal': (0.6, 1.1), 'weight': 0.5},
    'LV_mass_g':          {'cat': 'LV_structure',  'units': 'g',      'normal': (66, 150),  'weight': 0.5},
    'LV_mass_cube_g':     {'cat': 'LV_structure',  'units': 'g',      'normal': (50, 120),  'weight': 0.3},
    'RWT':                {'cat': 'LV_structure',  'units': '',       'normal': (0.22, 0.42), 'weight': 0.5},
    # ── LV Systolic Function ──
    'LVEDV_mL':           {'cat': 'LV_systolic',   'units': 'mL',     'normal': (80, 150),  'weight': 1.0},
    'LVESV_mL':           {'cat': 'LV_systolic',   'units': 'mL',     'normal': (25, 60),   'weight': 1.0},
    'SV_mL':              {'cat': 'LV_systolic',   'units': 'mL',     'normal': (50, 100),  'weight': 1.0},
    'LVEF_pct':           {'cat': 'LV_systolic',   'units': '%',      'normal': (55, 70),   'weight': 2.0},
    'CO_Lmin':            {'cat': 'LV_systolic',   'units': 'L/min',  'normal': (4.0, 7.0), 'weight': 1.0},
    'HR_bpm':             {'cat': 'LV_systolic',   'units': 'bpm',    'normal': (60, 100),  'weight': 0.5},
    'FS_pct':             {'cat': 'LV_systolic',   'units': '%',      'normal': (25, 45),   'weight': 0.5},
    'GLS_pct':            {'cat': 'LV_systolic',   'units': '%',      'normal': (-22, -16), 'weight': 2.0},
    # ── Mitral Inflow Doppler ──
    'E_vel_cms':          {'cat': 'doppler',       'units': 'cm/s',   'normal': (50, 100),  'weight': 1.0},
    'A_vel_cms':          {'cat': 'doppler',       'units': 'cm/s',   'normal': (40, 80),   'weight': 0.5},
    'EA_ratio':           {'cat': 'doppler',       'units': '',       'normal': (0.8, 2.0), 'weight': 1.0},
    'DT_ms':              {'cat': 'doppler',       'units': 'ms',     'normal': (150, 220), 'weight': 0.5},
    'IVRT_ms':            {'cat': 'doppler',       'units': 'ms',     'normal': (60, 100),  'weight': 0.5},
    # ── Tissue Doppler ──
    'e_prime_sep_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (7, 15),    'weight': 1.0},
    'e_prime_lat_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (10, 20),   'weight': 1.0},
    'e_prime_avg_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (8, 17),    'weight': 1.0},
    's_prime_sep_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (6, 10),    'weight': 0.5},
    's_prime_lat_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (8, 14),    'weight': 0.5},
    'a_prime_sep_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (6, 12),    'weight': 0.3},
    'a_prime_lat_cms':    {'cat': 'tissue_doppler','units': 'cm/s',   'normal': (8, 14),    'weight': 0.3},
    # ── Filling Pressures ──
    'E_e_prime_sep':      {'cat': 'filling',       'units': '',       'normal': (4, 13),    'weight': 2.0},
    'E_e_prime_lat':      {'cat': 'filling',       'units': '',       'normal': (4, 13),    'weight': 1.5},
    'E_e_prime_avg':      {'cat': 'filling',       'units': '',       'normal': (4, 13),    'weight': 2.0},
    'LAP_est_mmHg':       {'cat': 'filling',       'units': 'mmHg',   'normal': (5, 12),    'weight': 1.5},
    # ── LA ──
    'LAV_max_mL':         {'cat': 'LA',            'units': 'mL',     'normal': (22, 58),   'weight': 0.5},
    'LAV_min_mL':         {'cat': 'LA',            'units': 'mL',     'normal': (8, 25),    'weight': 0.3},
    'LAV_preA_mL':        {'cat': 'LA',            'units': 'mL',     'normal': (18, 50),   'weight': 0.3},
    'LA_diameter_cm':     {'cat': 'LA',            'units': 'cm',     'normal': (3.0, 4.5), 'weight': 0.5},
    'LA_total_EF_pct':    {'cat': 'LA',            'units': '%',      'normal': (50, 75),   'weight': 0.5},
    'LA_passive_EF_pct':  {'cat': 'LA',            'units': '%',      'normal': (5, 30),    'weight': 0.3},
    'LA_active_EF_pct':   {'cat': 'LA',            'units': '%',      'normal': (30, 60),   'weight': 0.3},
    'LARS_pct':           {'cat': 'LA',            'units': '%',      'normal': (20, 50),   'weight': 0.5},
    'LA_reservoir_strain_pct': {'cat': 'LA',       'units': '%',      'normal': (15, 40),   'weight': 0.5},
    'LA_conduit_strain_pct':   {'cat': 'LA',       'units': '%',      'normal': (5, 20),    'weight': 0.3},
    'LA_pump_strain_pct':      {'cat': 'LA',       'units': '%',      'normal': (10, 30),   'weight': 0.3},
    # ── RV ──
    'RVEDV_mL':           {'cat': 'RV',            'units': 'mL',     'normal': (80, 150),  'weight': 0.5},
    'RVESV_mL':           {'cat': 'RV',            'units': 'mL',     'normal': (25, 60),   'weight': 0.5},
    'RVEF_pct':           {'cat': 'RV',            'units': '%',      'normal': (45, 70),   'weight': 0.5},
    'RVSV_mL':            {'cat': 'RV',            'units': 'mL',     'normal': (50, 100),  'weight': 0.3},
    'TAPSE_mm':           {'cat': 'RV',            'units': 'mm',     'normal': (16, 30),   'weight': 0.5},
    'RV_FAC_pct':         {'cat': 'RV',            'units': '%',      'normal': (35, 60),   'weight': 0.3},
    'RV_s_prime_cms':     {'cat': 'RV',            'units': 'cm/s',   'normal': (8, 15),    'weight': 0.5},
    'RV_free_wall_strain_pct': {'cat': 'RV',       'units': '%',      'normal': (-30, -18), 'weight': 0.5},
    'RV_basal_diam_cm':   {'cat': 'RV',            'units': 'cm',     'normal': (3.5, 5.5), 'weight': 0.3},
    # ── Aortic Doppler ──
    'LVOT_diam_cm':       {'cat': 'aortic',        'units': 'cm',     'normal': (1.8, 2.6), 'weight': 0.3},
    'LVOT_VTI_cm':        {'cat': 'aortic',        'units': 'cm',     'normal': (18, 28),   'weight': 0.5},
    'AV_Vmax_cms':        {'cat': 'aortic',        'units': 'cm/s',   'normal': (50, 150),  'weight': 0.5},
    'AV_peak_grad_mmHg':  {'cat': 'aortic',        'units': 'mmHg',   'normal': (0, 20),    'weight': 0.3},
    'AV_mean_grad_mmHg':  {'cat': 'aortic',        'units': 'mmHg',   'normal': (0, 10),    'weight': 0.3},
    'AVA_cm2':            {'cat': 'aortic',        'units': 'cm2',    'normal': (2.5, 5.5), 'weight': 0.3},
    # ── Pulmonary Pressures ──
    'PASP_mmHg':          {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (15, 30),   'weight': 1.0},
    'PASP_bernoulli_mmHg':{'cat': 'pulmonary',     'units': 'mmHg',   'normal': (15, 35),   'weight': 0.5},
    'PADP_mmHg':          {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (4, 12),    'weight': 0.5},
    'mPAP_mmHg':          {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (8, 20),    'weight': 1.0},
    'TR_Vmax_ms':         {'cat': 'pulmonary',     'units': 'm/s',    'normal': (1.5, 2.8), 'weight': 0.5},
    'RAP_est_mmHg':       {'cat': 'pulmonary',     'units': 'mmHg',   'normal': (0, 5),     'weight': 0.5},
    # ── Blood Pressure ──
    'SBP_mmHg':           {'cat': 'BP',            'units': 'mmHg',   'normal': (90, 140),  'weight': 1.0},
    'SBP_central_mmHg':   {'cat': 'BP',            'units': 'mmHg',   'normal': (85, 130),  'weight': 0.5},
    'DBP_mmHg':           {'cat': 'BP',            'units': 'mmHg',   'normal': (60, 90),   'weight': 1.0},
    'DBP_central_mmHg':   {'cat': 'BP',            'units': 'mmHg',   'normal': (55, 85),   'weight': 0.5},
    'MAP_mmHg':           {'cat': 'BP',            'units': 'mmHg',   'normal': (70, 105),  'weight': 1.0},
    'pulse_pressure_mmHg':{'cat': 'BP',            'units': 'mmHg',   'normal': (25, 60),   'weight': 0.5},
    # ── RA ──
    'RAV_max_mL':         {'cat': 'RA',            'units': 'mL',     'normal': (15, 40),   'weight': 0.3},
    'RAV_min_mL':         {'cat': 'RA',            'units': 'mL',     'normal': (5, 20),    'weight': 0.3},
    'RA_diameter_cm':     {'cat': 'RA',            'units': 'cm',     'normal': (3.0, 4.5), 'weight': 0.3},
    # ── Timing / MPI ──
    'IVCT_ms':            {'cat': 'timing',        'units': 'ms',     'normal': (30, 80),   'weight': 0.3},
    'ET_ms':              {'cat': 'timing',        'units': 'ms',     'normal': (200, 350), 'weight': 0.3},
    'IVRT_lv_ms':         {'cat': 'timing',        'units': 'ms',     'normal': (60, 100),  'weight': 0.3},
    'MPI_LV':             {'cat': 'timing',        'units': '',       'normal': (0.3, 0.5), 'weight': 0.3},
    # ── Myocardial Work ──
    'GWI_mmHgpct':        {'cat': 'myocardial_work','units': 'mmHg%', 'normal': (1500, 2500), 'weight': 0.5},
    'GCW_mmHgpct':        {'cat': 'myocardial_work','units': 'mmHg%', 'normal': (300, 700),   'weight': 0.3},
    'GWW_mmHgpct':        {'cat': 'myocardial_work','units': 'mmHg%', 'normal': (0, 100),     'weight': 0.3},
    'GWE_pct':            {'cat': 'myocardial_work','units': '%',      'normal': (85, 98),     'weight': 0.5},
    # ── Diastolic Grade ──
    'diastolic_grade':    {'cat': 'diastolic',     'units': '',       'normal': (0, 0),     'weight': 1.0},
    'n_abnormal_criteria':{'cat': 'diastolic',     'units': '',       'normal': (0, 1),     'weight': 0.5},
    # ── Vascular ──
    'Ea_mmHg_mL':         {'cat': 'vascular',      'units': 'mmHg/mL','normal': (1.0, 2.5), 'weight': 0.5},
    'Ees_mmHg_mL':        {'cat': 'vascular',      'units': 'mmHg/mL','normal': (1.5, 4.0), 'weight': 0.5},
    'VA_coupling':        {'cat': 'vascular',      'units': '',       'normal': (0.3, 0.8), 'weight': 0.5},
    'C_total_mL_mmHg':    {'cat': 'vascular',      'units': 'mL/mmHg','normal': (1.0, 2.5), 'weight': 0.5},
    'PWV_surrogate_ms':   {'cat': 'vascular',      'units': 'm/s',    'normal': (4, 8),     'weight': 0.5},
    # ── Indexed ──
    'LVMi_g_m2':          {'cat': 'indexed',       'units': 'g/m2',   'normal': (43, 95),   'weight': 0.5},
    'LVMi_height_g_m27':  {'cat': 'indexed',       'units': 'g/m2.7', 'normal': (18, 44),   'weight': 0.5},
    'LVEDVi_mL_m2':      {'cat': 'indexed',       'units': 'mL/m2',  'normal': (40, 80),   'weight': 0.5},
    'LVESVi_mL_m2':      {'cat': 'indexed',       'units': 'mL/m2',  'normal': (15, 35),   'weight': 0.5},
    'SVi_mL_m2':          {'cat': 'indexed',       'units': 'mL/m2',  'normal': (25, 50),   'weight': 0.5},
    'CI_L_min_m2':        {'cat': 'indexed',       'units': 'L/min/m2','normal': (2.5, 4.0), 'weight': 0.5},
    'LAVi_mL_m2':         {'cat': 'indexed',       'units': 'mL/m2',  'normal': (16, 34),   'weight': 0.5},
    'RAVi_mL_m2':         {'cat': 'indexed',       'units': 'mL/m2',  'normal': (8, 22),    'weight': 0.3},
    'RVEDVi_mL_m2':      {'cat': 'indexed',       'units': 'mL/m2',  'normal': (40, 80),   'weight': 0.3},
    # ── Renal / Lab ──
    'eGFR_mL_min_173m2':  {'cat': 'renal',         'units': 'mL/min/1.73m2', 'normal': (90, 120), 'weight': 2.0},
    'GFR_mL_min':         {'cat': 'renal',         'units': 'mL/min', 'normal': (90, 130),  'weight': 2.0},
    'RBF_mL_min':         {'cat': 'renal',         'units': 'mL/min', 'normal': (900, 1300),'weight': 0.5},
    'serum_creatinine_mg_dL': {'cat': 'renal',     'units': 'mg/dL',  'normal': (0.6, 1.2), 'weight': 1.0},
    'cystatin_C_mg_L':    {'cat': 'renal',         'units': 'mg/L',   'normal': (0.5, 1.0), 'weight': 0.5},
    'BUN_mg_dL':          {'cat': 'renal',         'units': 'mg/dL',  'normal': (7, 20),    'weight': 0.5},
    'UACR_mg_g':          {'cat': 'renal',         'units': 'mg/g',   'normal': (0, 30),    'weight': 1.0},
    'serum_Na_mEq_L':     {'cat': 'renal',         'units': 'mEq/L',  'normal': (135, 145), 'weight': 0.5},
    'serum_K_mEq_L':      {'cat': 'renal',         'units': 'mEq/L',  'normal': (3.5, 5.0), 'weight': 0.5},
    'blood_volume_mL':    {'cat': 'renal',         'units': 'mL',     'normal': (4500, 5500),'weight': 0.5},
    'plasma_volume_mL':   {'cat': 'renal',         'units': 'mL',     'normal': (2500, 3200),'weight': 0.3},
    'NTproBNP_pg_mL':     {'cat': 'renal',         'units': 'pg/mL',  'normal': (0, 125),   'weight': 2.0},
    'hsTnT_ng_L':         {'cat': 'renal',         'units': 'ng/L',   'normal': (0, 14),    'weight': 1.0},
    'P_glom_mmHg':        {'cat': 'renal',         'units': 'mmHg',   'normal': (45, 65),   'weight': 0.5},
    'renal_resistive_index': {'cat': 'renal',      'units': '',       'normal': (0.55, 0.70),'weight': 0.5},
    'Kf_scale':           {'cat': 'renal',         'units': '',       'normal': (0.8, 1.0), 'weight': 0.5},
    'Na_excretion_mEq_day': {'cat': 'renal',       'units': 'mEq/day','normal': (100, 250), 'weight': 0.5},
}

# Non-numeric variables excluded from NN
NON_NUMERIC_VARS = {'diastolic_label'}

# Ordered list of numeric variable names for NN input/output
NUMERIC_VAR_NAMES = sorted(k for k in ARIC_VARIABLES if k not in NON_NUMERIC_VARS)
N_FEATURES = len(NUMERIC_VAR_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# Clinical Thresholds
# ═══════════════════════════════════════════════════════════════════════════

CLINICAL_THRESHOLDS = {
    'LVEF_pct': [
        (50.0, 'HFpEF'),
        (40.0, 'HFmrEF'),
        (0.0,  'HFrEF'),
    ],
    'eGFR_mL_min_173m2': [
        (90.0, 'normal'),
        (60.0, 'CKD_G2'),
        (45.0, 'CKD_G3a'),
        (30.0, 'CKD_G3b'),
        (15.0, 'CKD_G4'),
        (0.0,  'CKD_G5'),
    ],
    'E_e_prime_avg': [
        (14.0, 'elevated_filling_pressure'),
        (8.0,  'indeterminate'),
        (0.0,  'normal'),
    ],
    'UACR_mg_g': [
        (300.0, 'macroalbuminuria'),
        (30.0,  'microalbuminuria'),
        (0.0,   'normal'),
    ],
    'NTproBNP_pg_mL': [
        (450.0, 'HF_likely'),
        (125.0, 'HF_possible'),
        (0.0,   'normal'),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# LLM Configuration
# ═══════════════════════════════════════════════════════════════════════════

LLM_CONFIG = {
    'model': 'gpt-4o',               # default LiteLLM model string
    'max_iterations': 15,
    'convergence_threshold': 0.05,    # 5% normalized error
    'temperature': 0.3,
    'max_tokens': 4096,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Generation Defaults
# ═══════════════════════════════════════════════════════════════════════════

COHORT_DEFAULTS = {
    'n_patients': 10000,
    'seed': 42,
    'n_workers': 8,
    'n_coupling_steps': 3,            # coupling steps for equilibration
    'dt_renal_hours': 6.0,
}

NN_DEFAULTS = {
    'hidden_dim': 256,
    'n_blocks': 3,
    'dropout': 0.1,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 200,
    'batch_size': 256,
    'patience': 20,
    'train_frac': 0.70,
    'val_frac': 0.15,
    'test_frac': 0.15,
}
