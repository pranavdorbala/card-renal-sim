#!/usr/bin/env python3
"""
Cardiorenal Coupling Simulator
==============================
Couples the CircAdapt VanOsta2024 heart model with a Hallow et al.
renal physiology module via asynchronous message passing.

Heart model:  circadapt.VanOsta2024  (real CircAdapt framework)
Kidney model: Hallow et al. 2017 CPT:PSP renal equations

Message protocol:
    Heart  → Kidney :  MAP, CO, central venous pressure
    Kidney → Heart  :  total blood volume, systemic vascular resistance

Deterioration knobs:
    Cardiac : Sf_act_scale — scales Patch['Sf_act'] on LV+SV
              (1.0 = healthy, <1 = reduced contractility / HFrEF)
    Kidney  : Kf_scale — scales glomerular ultrafiltration coefficient
              (1.0 = healthy, <1 = nephron loss / CKD)

Usage:
    pip install circadapt matplotlib numpy
    python cardiorenal_coupling.py

References:
    - CircAdapt: framework.circadapt.org  (VanOsta et al. 2024)
    - Renal model: Hallow & Gebremichael, CPT:PSP 6:383-392, 2017
    - Coupled model: Basu et al., PLoS Comput Biol 19(11):e1011598, 2023

Author: Generated for cardiorenal research coupling study
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Safe for headless; remove if running interactively
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings, copy

# ── CircAdapt import ────────────────────────────────────────────────────────
from circadapt import VanOsta2024

# ── Constants ───────────────────────────────────────────────────────────────
PA_TO_MMHG = 7.5e-3          # 1 Pa  = 7.5e-3 mmHg  (= 1/133.322)
MMHG_TO_PA = 133.322         # 1 mmHg = 133.322 Pa
M3_TO_ML   = 1e6             # 1 m³  = 1e6 mL
ML_TO_M3   = 1e-6
M3S_TO_LMIN = 6e4            # 1 m³/s = 60 000 mL/min → 60 L/min


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 ─ CircAdapt Heart Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CircAdaptHeartModel:
    """
    Wraps circadapt.VanOsta2024 for use in the coupling loop.

    Key operations:
      • apply_deterioration()  – scale Patch['Sf_act'] on LV / SV
      • apply_kidney_feedback() – set blood volume and vascular resistance
      • run_to_steady_state()  – run(stable=True) and extract hemodynamics
    """

    def __init__(self):
        self.model = VanOsta2024()
        # Store healthy reference values (before any changes)
        self._ref_Sf_act_lv = float(self.model['Patch']['Sf_act']['pLv0'])
        self._ref_Sf_act_sv = float(self.model['Patch']['Sf_act']['pSv0'])
        self._ref_ArtVen_p0 = float(self.model['ArtVen']['p0']['CiSy'])

        # Arterial stiffness reference (Tube0D)
        self._ref_Tube0D_k = float(self.model['Tube0D']['k']['SyArt'])

        # Passive myocardial stiffness reference (Patch k1)
        # Wrapped in try/except for CircAdapt version compatibility
        try:
            self._ref_k1_lv = float(self.model['Patch']['k1']['pLv0'])
            self._ref_k1_sv = float(self.model['Patch']['k1']['pSv0'])
        except Exception:
            self._ref_k1_lv = None
            self._ref_k1_sv = None

        # Inflammatory modifiers (set by apply_inflammatory_modifiers)
        self._pathology_p0_factor = 1.0
        self._inflammatory_k1_factor = 1.0

    # ── Cardiac deterioration ───────────────────────────────────────────
    def apply_deterioration(self, Sf_act_scale: float):
        """
        Scale active fiber stress for LV and septal patches.
        Sf_act_scale = 1.0 → healthy;  < 1.0 → reduced contractility (HFrEF)

        Maps directly to CircAdapt:  model['Patch']['Sf_act']['pLv0'] etc.
        This is the same parameter used in the VanOsta2024 Heart Failure tutorial.
        """
        self.model['Patch']['Sf_act']['pLv0'] = self._ref_Sf_act_lv * Sf_act_scale
        self.model['Patch']['Sf_act']['pSv0'] = self._ref_Sf_act_sv * Sf_act_scale

    # ── Diastolic stiffness (HFpEF) ──────────────────────────────────
    def apply_stiffness(self, k1_scale: float):
        """
        Scale passive myocardial stiffness for LV and septal patches.
        k1_scale = 1.0 → healthy;  >1.0 → stiffer ventricle (HFpEF)

        Maps to CircAdapt: model['Patch']['k1']['pLv0'] etc.
        Higher k1 → steeper EDPVR → elevated filling pressures at same EDV.
        """
        k1_scale = np.clip(k1_scale, 0.5, 4.0)  # stability guard
        if self._ref_k1_lv is not None:
            self.model['Patch']['k1']['pLv0'] = self._ref_k1_lv * k1_scale
            self.model['Patch']['k1']['pSv0'] = self._ref_k1_sv * k1_scale

    # ── Accept kidney feedback ──────────────────────────────────────────
    def apply_kidney_feedback(self, V_blood_m3: float, SVR_ratio: float):
        """
        Modify CircAdapt model based on renal outputs.

        V_blood_m3 : new total blood volume [m³]
            → fed to PFC volume control to update circulating volume.
        SVR_ratio  : ratio of new SVR / baseline SVR  (dimensionless)
            → scales ArtVen['p0'] for the systemic bed, which controls
              the pressure-flow relationship and thus peripheral resistance.
        """
        # --- Volume: use PFC volume control --------------------------------
        self.model['PFC']['is_volume_control'] = True
        self.model['PFC']['target_volume'] = V_blood_m3

        # --- Resistance: scale systemic ArtVen p0 --------------------------
        # In CircAdapt's ArtVen, q = sign(Δp) · q0 · (|Δp|/p0)^k
        # Raising p0 is equivalent to raising the resistance set-point.
        # _pathology_p0_factor incorporates inflammatory SVR effects
        # (endothelial dysfunction, microvascular rarefaction)
        self.model['ArtVen']['p0']['CiSy'] = (
            self._ref_ArtVen_p0 * SVR_ratio * self._pathology_p0_factor
        )

    def reset_volume_control(self):
        """Turn off volume control after PFC has adjusted the volume."""
        self.model['PFC']['is_volume_control'] = False

    # ── Inflammatory mediator effects ────────────────────────────────
    def apply_inflammatory_modifiers(self, state: 'InflammatoryState'):
        """
        Apply inflammatory / metabolic mediator effects to cardiac parameters.

        This is called BEFORE apply_stiffness / apply_deterioration so that
        the inflammatory effects compose multiplicatively with direct schedules.

        Modifies:
          - Tube0D['k']['SyArt']      : arterial stiffness (applied directly)
          - _inflammatory_k1_factor    : stored for composition with k1_scale
                                         in apply_stiffness (called separately)
          - _pathology_p0_factor       : SVR modifier (applied in apply_kidney_feedback)
        """
        # Arterial stiffness (PWV increase from inflammation + AGEs)
        self.model['Tube0D']['k']['SyArt'] = (
            self._ref_Tube0D_k * state.stiffness_factor
        )

        # Store k1 factor — composed with direct k1_scale in apply_stiffness
        # (not applied here; single point of k1 assignment is apply_stiffness)
        self._inflammatory_k1_factor = state.passive_k1_factor

        # Store p0 factor — applied multiplicatively with SVR_ratio
        # in apply_kidney_feedback
        self._pathology_p0_factor = state.p0_factor

    # ── Run to steady state ─────────────────────────────────────────────
    def run_to_steady_state(self, n_settle: int = 5) -> Dict:
        """
        Run VanOsta2024 until hemodynamically stable, then extract signals.

        Returns dict with:
            MAP, SBP, DBP      [mmHg]
            CO, SV              [L/min, mL]
            EF                  [%]
            Pven                [mmHg]
            V_blood_total       [mL]
            V_LV, p_LV, V_RV, p_RV, p_SyArt, t   (waveform arrays)
        """
        # Run with PFC active to converge to new steady state
        self.model['PFC']['is_active'] = True
        try:
            self.model.run(stable=True)
        except Exception:
            # Fallback: run a fixed number of beats
            try:
                self.model.run(n_settle)
            except Exception:
                pass  # model numerically crashed – continue with last state

        # Extra beats with PFC off if volume control was on
        self.reset_volume_control()
        try:
            self.model.run(stable=True)
        except Exception:
            try:
                self.model.run(n_settle)
            except Exception:
                pass  # model numerically crashed – continue with last state

        # Store one clean beat for plotting
        self.model['Solver']['store_beats'] = 1
        try:
            self.model.run(1)
        except Exception:
            pass  # model numerically crashed – use last available state

        return self._extract_hemodynamics()

    # ── Extract hemodynamic signals ─────────────────────────────────────
    def _extract_hemodynamics(self) -> Dict:
        """Pull pressures, volumes, and derived indices from CircAdapt."""
        # Time
        t = self.model['Solver']['t'] * 1e3  # ms

        # LV
        V_LV = self.model['Cavity']['V'][:, 'cLv'] * M3_TO_ML     # mL
        p_LV = self.model['Cavity']['p'][:, 'cLv'] * PA_TO_MMHG   # mmHg

        # RV
        V_RV = self.model['Cavity']['V'][:, 'cRv'] * M3_TO_ML
        p_RV = self.model['Cavity']['p'][:, 'cRv'] * PA_TO_MMHG

        # Aortic pressure
        p_SyArt = self.model['Cavity']['p'][:, 'SyArt'] * PA_TO_MMHG

        # Systemic venous pressure
        p_SyVen = self.model['Cavity']['p'][:, 'SyVen'] * PA_TO_MMHG

        # Derived
        SBP = float(np.max(p_SyArt))
        DBP = float(np.min(p_SyArt))
        MAP = (SBP + 2.0 * DBP) / 3.0

        EDV = float(np.max(V_LV))
        ESV = float(np.min(V_LV))
        SV  = EDV - ESV
        EF  = SV / max(EDV, 1.0) * 100.0

        t_cycle = float(self.model['General']['t_cycle'])
        HR = 60.0 / t_cycle
        CO = SV * HR / 1000.0   # L/min

        Pven = float(np.mean(p_SyVen))

        # Total circulating volume (sum over all cavities)
        V_all = self.model['Cavity']['V'][:, :]  # all cavities, all time
        V_blood_mL = float(np.mean(np.sum(V_all, axis=1))) * M3_TO_ML

        return {
            # Waveforms
            't': t, 'V_LV': V_LV, 'p_LV': p_LV,
            'V_RV': V_RV, 'p_RV': p_RV, 'p_SyArt': p_SyArt,
            # Scalars
            'MAP': MAP, 'SBP': SBP, 'DBP': DBP,
            'CO': CO, 'SV': SV, 'EF': EF, 'EDV': EDV, 'ESV': ESV,
            'Pven': Pven, 'HR': HR,
            'V_blood_total': V_blood_mL,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 ─ Hallow Renal Module
# ═══════════════════════════════════════════════════════════════════════════
#
# Implements the key equations from:
#   Hallow KM, Gebremichael Y. CPT:PSP (2017) 6:383–392
#   Hallow KM et al. AJP-Renal (2017) 312:F819–F835
#   Basu S et al. PLoS Comput Biol (2023) 19:e1011598
#
# Equations:
#   Renal hemodynamics:
#     RBF = (MAP − P_rv) / R_total ;  R_total = R_preAA + R_AA + R_EA
#     P_gc = MAP − RBF·(R_preAA + R_AA)
#     SNGFR = Kf · (P_gc − P_Bow − π_avg)
#     GFR = 2·N·SNGFR
#
#   Tubuloglomerular feedback (TGF):
#     Senses macula densa Na delivery → adjusts R_AA
#
#   RAAS:
#     Low MAP → ↑renin → ↑AngII → ↑R_EA + ↑aldosterone → ↑CD reabsorption
#
#   Tubular Na handling:
#     PT: glomerulotubular balance (reabsorbs η_PT of filtered load)
#     LoH, DT, CD: fractional reabsorption
#
#   Volume balance:
#     dV_blood/dt = water_intake − water_excretion − capillary filtration
#     dNa_total/dt = Na_intake − Na_excretion
#
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HallowRenalModel:
    """Steady-state renal module based on Hallow et al. equations."""

    # ── Glomerular ──────────────────────────────────────────────────────
    N_nephrons: float = 1.0e6        # per kidney (× 2)
    Kf: float = 8.0                  # ultrafiltration coeff [nL/min/mmHg/nephron]

    # ── Vascular resistances (normalized, 2-kidney) ─────────────────────
    # Calibrated to produce GFR ~120 mL/min at MAP ~93 mmHg
    R_preAA: float = 12.0            # pre-afferent [mmHg·min/mL × 1e-3]
    R_AA0:   float = 26.0            # afferent arteriolar baseline
    R_EA0:   float = 43.0            # efferent arteriolar baseline

    # ── Fixed pressures ────────────────────────────────────────────────
    P_Bow:        float = 18.0       # Bowman space [mmHg]
    P_renal_vein: float = 4.0        # renal vein    [mmHg]
    pi_plasma:    float = 25.0       # plasma oncotic [mmHg]
    Hct:          float = 0.45

    # ── Tubular reabsorption fractions ─────────────────────────────────
    eta_PT:  float = 0.67            # proximal tubule (glomerulotubular balance)
    eta_LoH: float = 0.25            # loop of Henle
    eta_DT:  float = 0.05            # distal tubule
    eta_CD0: float = 0.024           # collecting duct baseline (aldosterone-sensitive)

    # ── Water ──────────────────────────────────────────────────────────
    frac_water_reabs: float = 0.99

    # ── Intake ─────────────────────────────────────────────────────────
    Na_intake: float = 150.0         # mEq/day
    water_intake: float = 2.0        # L/day

    # ── Feedback gains ─────────────────────────────────────────────────
    TGF_gain:     float = 2.0
    TGF_setpoint: float = 0.0        # initialised on first call
    RAAS_gain:    float = 1.5
    MAP_setpoint: float = 93.0       # mmHg at which RAAS is quiescent

    # ── State ──────────────────────────────────────────────────────────
    V_blood: float = 5000.0          # mL
    Na_total: float = 2100.0         # mEq (≈ C_Na × V_ECF)
    C_Na:    float = 140.0           # plasma Na [mEq/L]

    # ── Outputs (written by update) ────────────────────────────────────
    GFR:            float = 120.0    # mL/min
    RBF:            float = 1100.0   # mL/min
    P_glom:         float = 60.0     # mmHg
    Na_excretion:   float = 150.0    # mEq/day
    water_excretion: float = 1.5     # L/day

    # ── Deterioration ──────────────────────────────────────────────────
    Kf_scale: float = 1.0            # 1.0 = healthy, <1 = CKD / nephron loss


# ═══════════════════════════════════════════════════════════════════════════
# PART 2b ─ Inflammatory Mediator Layer
# ═══════════════════════════════════════════════════════════════════════════
#
# Inflammation is not a standalone organ — it is the biochemical medium
# through which organ damage in one compartment propagates to others.
# It sits as a mediator layer on the edges of the heart ↔ kidney graph,
# operating on a slow timescale (weeks–months) alongside the fast
# hemodynamic coupling (hours).
#
# Architecture:
#
#                 ┌───────────────────────┐
#                 │   Diabetes /          │
#                 │   Metabolic Node      │
#                 └────┬────────┬─────────┘
#                      │        │
#                  AGE │        │ glucose
#                 RAGE │        │ hyperfiltration
#                      ▼        ▼
#         ┌──── INFLAMMATION LAYER ────┐
#         │  systemic_inflammatory_idx │
#         │  endothelial_dysfunction   │
#         │                            │
#         │  Sources: uremia,          │
#         │    congestion, AGE, aldo   │
#         │  Outputs: fibrosis rates,  │
#         │    vascular tone,          │
#         │    nephron loss rate        │
#         └────┬───────────────┬───────┘
#              │               │
#   fibrosis   │               │ nephron loss
#   stiffness  │               │ impaired autoregulation
#              ▼               ▼
#       ┌──────────┐    ┌──────────┐
#       │  Heart   │◄──►│  Kidney  │
#       │(CircAdapt│    │ (Hallow) │
#       └──────────┘    └──────────┘
#        hemodynamic coupling (existing)
#
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InflammatoryState:
    """
    Inflammatory mediator layer — sits between heart and kidney.

    In the full model, these are dynamic state variables driven by ODEs:
      - Sources: uremia (kidney damage), congestion (cardiac EDP),
        AGEs (diabetes), aldosterone excess
      - Sinks: hepatic/immune clearance
      - Outputs: fibrosis rates, vascular tone, nephron loss rate

    In the simple (active) version, these are computed from input
    schedules (inflammation_scale, diabetes_scale) via parametric scaling.
    """
    # ── Core state variables ─────────────────────────────────────────
    systemic_inflammatory_index: float = 0.0   # 0 = none, 1 = severe
    diabetes_metabolic_index: float = 0.0      # 0 = none, 1 = severe

    # ── Derived mediator effects (computed each step) ────────────────
    # Cardiac effects
    Sf_act_factor: float = 1.0        # contractility modifier
    p0_factor: float = 1.0            # SVR / resistance modifier
    stiffness_factor: float = 1.0     # arterial stiffness modifier
    passive_k1_factor: float = 1.0    # diastolic stiffness (diabetes)

    # Renal effects
    Kf_factor: float = 1.0            # glomerular filtration modifier
    R_AA_factor: float = 1.0          # afferent arteriole modifier
    R_EA_factor: float = 1.0          # efferent arteriole modifier
    RAAS_gain_factor: float = 1.0     # RAAS sensitivity modifier
    eta_PT_offset: float = 0.0        # proximal tubule Na reabsorption
    MAP_setpoint_offset: float = 0.0  # pressure-natriuresis shift

    # ── Future dynamic state variables (commented out in simple ver) ──
    # myocardial_fibrosis_volume: float = 0.0
    # endothelial_dysfunction_index: float = 0.0
    # renal_tubulointerstitial_fibrosis: float = 0.0
    # AGE_accumulation: float = 0.0


def update_inflammatory_state(
    state: InflammatoryState,
    inflammation_scale: float,
    diabetes_scale: float,
    # ── Inputs from heart/kidney for future ODE version ──────────────
    # GFR: float = 120.0,           # kidney damage → uremic inflammation
    # EDP: float = 10.0,            # cardiac congestion → inflammation
    # aldosterone_factor: float = 1.0,  # RAAS → pro-inflammatory
    # P_glom: float = 60.0,         # glomerular pressure → podocyte stress
    # CVP: float = 3.0,             # venous congestion → renal back-pressure
    # MAP: float = 93.0,            # mean arterial pressure → shear stress
    # dt_hours: float = 6.0,        # integration timestep
) -> InflammatoryState:
    """
    Update the inflammatory mediator layer.

    SIMPLE VERSION (active):
        Direct parametric scaling from input schedules.
        inflammation_scale and diabetes_scale (0 = none, 1 = severe)
        are mapped to modifier effects on cardiac and renal parameters
        via literature-derived coefficients.

    FULL ODE VERSION (commented out):
        Dynamic state variables driven by organ damage signals.
        Sources: uremia (1/GFR), congestion (EDP), AGEs (diabetes),
        aldosterone excess.  Sinks: hepatic/immune clearance.
        Outputs: fibrosis rates, vascular tone, nephron loss rate.
    """
    infl = np.clip(inflammation_scale, 0.0, 1.0)
    diab = np.clip(diabetes_scale, 0.0, 1.0)

    state.systemic_inflammatory_index = infl
    state.diabetes_metabolic_index = diab

    # ══════════════════════════════════════════════════════════════════
    # SIMPLE: Parametric scaling from input schedules
    # ══════════════════════════════════════════════════════════════════

    # ── Cardiac effects ──────────────────────────────────────────────

    # Contractility: TNF-α cardiodepressant (Feldman 2000; Finkel 1992)
    #   TNF-α reduces Sf_act by up to 25% in severe inflammation
    infl_Sf = 1.0 - 0.25 * infl
    # Diabetic cardiomyopathy: AGE cross-linking, lipotoxicity (Bugger 2014)
    #   Diabetic cardiomyopathy reduces contractility 15–20%
    diab_Sf = 1.0 - 0.20 * diab
    state.Sf_act_factor = infl_Sf * diab_Sf

    # Vascular resistance (SVR): endothelial dysfunction
    #   Chronic inflammation increases SVR 10–20% via NO impairment
    infl_p0 = 1.0 + 0.15 * infl
    # Microvascular rarefaction in diabetes → modest SVR increase
    diab_p0 = 1.0 + 0.10 * diab
    state.p0_factor = infl_p0 * diab_p0

    # Arterial stiffness: MMP-mediated elastin degradation
    #   CRP correlates with 20–30% increase in PWV (Vlachopoulos 2005)
    infl_k = 1.0 + 0.30 * infl
    # AGE-mediated arterial stiffening: diabetes increases PWV 30–50%
    #   (Prenner & Chirinos 2015)
    diab_k = 1.0 + 0.50 * diab
    state.stiffness_factor = infl_k * diab_k

    # Passive myocardial stiffness (diastolic dysfunction)
    #   AGE collagen cross-linking → 30–50% increase in LV passive
    #   stiffness, driving HFpEF phenotype (van Heerebeek 2008)
    state.passive_k1_factor = 1.0 + 0.40 * diab

    # ── Renal effects ────────────────────────────────────────────────

    # Glomerular filtration coefficient (Kf)
    #   Inflammation: mesangial expansion, podocyte injury → ↓Kf
    infl_Kf = 1.0 - 0.20 * infl
    #   Diabetes: BIPHASIC — early hyperfiltration (glomerular
    #   hypertrophy, AA dilation) then progressive decline from
    #   mesangial expansion and GBM thickening (Brenner 1996;
    #   Ruggenenti 1998).  Peaks at +8% around diab=0.33, then
    #   falls to 0.625 at diab=1.0.
    diab_Kf = 1.0 + 0.25 * diab * (1.0 - 1.5 * diab)
    state.Kf_factor = max(infl_Kf * diab_Kf, 0.05)

    # Afferent arteriole resistance
    #   Inflammation: endothelin-1 upregulation constricts AA
    infl_RAA = 1.0 + 0.20 * infl
    #   Diabetes: early AA dilation (prostaglandins, TGF blunting)
    #   is the primary mechanism of diabetic hyperfiltration (Vallon 2003)
    #   Returns toward normal at severe diabetes
    diab_RAA = 1.0 - 0.15 * diab * (1.0 - diab)
    state.R_AA_factor = infl_RAA * diab_RAA

    # Efferent arteriole resistance
    #   AngII-driven EA constriction — hallmark of diabetic nephropathy
    #   Raises P_gc and drives hyperfiltration (Brenner 1996)
    state.R_EA_factor = 1.0 + 0.25 * diab

    # RAAS gain
    #   IL-6 stimulates angiotensinogen production in proximal tubule
    state.RAAS_gain_factor = 1.0 + 0.30 * infl

    # Proximal tubule Na reabsorption
    #   Inflammation: NHE3 stimulation → increased PT Na reabsorption
    #   Diabetes: SGLT2 hyperactivity increases PT reabsorption by
    #   5–8% of filtered load (Thomson 2004)
    state.eta_PT_offset = 0.04 * infl + 0.06 * diab

    # MAP setpoint (pressure-natriuresis curve shift)
    #   Inflammation resets renal set-point ~5 mmHg higher
    #   Diabetes: loss of nocturnal dipping, chronically elevated → ~8 mmHg
    #   Use max (overlapping mechanisms, not fully additive)
    state.MAP_setpoint_offset = max(5.0 * infl, 8.0 * diab)

    # ══════════════════════════════════════════════════════════════════
    # FULL ODE VERSION — commented out for iterative verification
    # ══════════════════════════════════════════════════════════════════
    #
    # When ready to activate, uncomment this block, enable the state
    # variables in InflammatoryState, and pass GFR/EDP/aldosterone/etc
    # from the coupling loop.
    #
    # # ── Rate constants (to be calibrated via sensitivity analysis) ──
    # k_uremic = 0.01           # uremic toxin → inflammation
    # k_congestion = 0.005      # cardiac congestion → inflammation
    # k_AGE = 0.02              # AGE accumulation → inflammation via RAGE
    # k_aldo = 0.008            # aldosterone excess → pro-inflammatory
    # k_clearance = 0.05        # hepatic / immune clearance of inflammation
    # k_AGE_formation = 0.001   # diabetes → AGE accumulation rate
    # k_AGE_turnover = 0.0005   # AGE cross-link turnover (very slow)
    # k_fibrosis_inflam = 0.003 # inflammation → myocardial fibrosis
    # k_fibrosis_mech = 0.002   # mechanical stress → myocardial fibrosis
    # k_fibrosis_turnover = 0.001  # slow collagen remodeling
    # k_endoth_inflam = 0.01    # inflammation → endothelial dysfunction
    # k_endoth_shear = 0.002    # hypertension → endothelial damage
    # k_endoth_recovery = 0.008 # endothelial repair (NO restoration)
    # k_renal_inflam = 0.004    # inflammation → tubulointerstitial fibrosis
    # k_renal_pressure = 0.002  # glomerular hypertension → podocyte loss
    # k_renal_congestion = 0.003  # venous congestion → renal fibrosis
    #
    # # ── Sources of systemic inflammation ────────────────────────────
    # # Kidney → inflammation: uremic toxin accumulation
    # # (indoxyl sulfate, p-cresyl sulfate, TMAO)
    # uremic_source = k_uremic * max(0.0, 1.0 / max(GFR, 5.0) - 1.0 / 120.0)
    #
    # # Heart → inflammation: venous congestion, elevated EDP
    # congestion_source = k_congestion * max(0.0, EDP - 10.0)
    #
    # # Diabetes → inflammation: AGE-RAGE signaling
    # AGE_source = k_AGE * state.AGE_accumulation
    #
    # # RAAS → inflammation: aldosterone is independently pro-inflammatory
    # # (mechanistic basis for MRA therapy — TOPCAT, FINEARTS-HF)
    # aldo_source = k_aldo * max(0.0, aldosterone_factor - 1.0)
    #
    # # ── Inflammatory index ODE (logistic saturation ceiling) ────────
    # inflam_max = 1.0
    # d_inflam = (
    #     uremic_source + congestion_source + AGE_source + aldo_source
    #     - k_clearance * state.systemic_inflammatory_index
    # ) * (1.0 - state.systemic_inflammatory_index / inflam_max)
    # state.systemic_inflammatory_index += d_inflam * dt_hours
    # state.systemic_inflammatory_index = np.clip(
    #     state.systemic_inflammatory_index, 0.0, 1.0)
    #
    # # ── AGE accumulation (very slow timescale, diabetes-driven) ─────
    # d_AGE = (k_AGE_formation * diab
    #          - k_AGE_turnover * state.AGE_accumulation)
    # state.AGE_accumulation += d_AGE * dt_hours
    # state.AGE_accumulation = max(0.0, state.AGE_accumulation)
    #
    # # ── Myocardial fibrosis (slow, inflammation + mechanical) ───────
    # # Replaces fixed p[96]+p[97] from the original model.
    # # Inflammation: uremic toxins drive fibrosis through TGF-β
    # # Mechanical: elevated EDP / wall stress → fibroblast activation
    # d_fibrosis = (
    #     k_fibrosis_inflam * state.systemic_inflammatory_index
    #     + k_fibrosis_mech * max(0.0, EDP / 12.0 - 1.0)
    #     - k_fibrosis_turnover * state.myocardial_fibrosis_volume
    # )
    # state.myocardial_fibrosis_volume += d_fibrosis * dt_hours
    # state.myocardial_fibrosis_volume = np.clip(
    #     state.myocardial_fibrosis_volume, 0.0, 1.0)
    #
    # # ── Endothelial dysfunction ──────────────────────────────────────
    # # Captures NO bioavailability reduction.  Affects peripheral
    # # resistance, renal afferent arteriole tone, and coronary
    # # microvascular function.
    # d_endoth = (
    #     k_endoth_inflam * state.systemic_inflammatory_index
    #     + k_endoth_shear * max(0.0, MAP - 100.0)
    #     - k_endoth_recovery * state.endothelial_dysfunction_index
    # )
    # state.endothelial_dysfunction_index += d_endoth * dt_hours
    # state.endothelial_dysfunction_index = np.clip(
    #     state.endothelial_dysfunction_index, 0.0, 1.0)
    #
    # # ── Renal tubulointerstitial fibrosis ────────────────────────────
    # # Drives nephron loss.  No negative term: nephron loss is
    # # irreversible, matching clinical reality.
    # d_renal_fib = (
    #     k_renal_inflam * state.systemic_inflammatory_index
    #     + k_renal_pressure * max(0.0, P_glom - 65.0)
    #     + k_renal_congestion * max(0.0, CVP - 8.0)
    # )
    # state.renal_tubulointerstitial_fibrosis += d_renal_fib * dt_hours
    # state.renal_tubulointerstitial_fibrosis = min(
    #     state.renal_tubulointerstitial_fibrosis, 0.95)
    #
    # # ── Derive modifier effects from dynamic state ──────────────────
    # infl = state.systemic_inflammatory_index
    # fibrosis = state.myocardial_fibrosis_volume
    # endoth = state.endothelial_dysfunction_index
    # renal_fib = state.renal_tubulointerstitial_fibrosis
    # AGE = state.AGE_accumulation
    #
    # state.Sf_act_factor = (1.0 - 0.25 * infl) * (1.0 - 0.15 * fibrosis)
    # state.p0_factor = 1.0 + 0.15 * endoth + 0.10 * AGE
    # state.stiffness_factor = 1.0 + 0.30 * endoth + 0.50 * AGE
    # state.passive_k1_factor = 1.0 + 0.40 * fibrosis + 0.30 * AGE
    # state.Kf_factor = max((1.0 - 0.20 * infl) * (1.0 - renal_fib), 0.05)
    # state.R_AA_factor = 1.0 + 0.20 * endoth
    # state.R_EA_factor = 1.0 + 0.25 * AGE
    # state.RAAS_gain_factor = 1.0 + 0.30 * infl
    # state.eta_PT_offset = 0.04 * infl + 0.06 * AGE
    # state.MAP_setpoint_offset = 5.0 * infl + 3.0 * endoth

    return state


def update_renal_model(renal: HallowRenalModel,
                       MAP: float, CO: float, P_ven: float,
                       dt_hours: float = 6.0,
                       inflammatory_state: Optional['InflammatoryState'] = None,
                       ) -> HallowRenalModel:
    """
    Update the Hallow renal model given cardiac hemodynamic inputs.

    Parameters
    ----------
    renal    : current renal state
    MAP      : mean arterial pressure [mmHg]   (from CircAdapt)
    CO       : cardiac output [L/min]           (from CircAdapt)
    P_ven    : central venous pressure [mmHg]   (from CircAdapt)
    dt_hours : integration time-step for volume balance
    inflammatory_state : InflammatoryState or None
        If provided, inflammatory mediator effects are applied to
        renal parameters (Kf, R_AA, R_EA, RAAS_gain, eta_PT,
        MAP_setpoint).  If None, no inflammatory effects (backward
        compatible).
    """
    # ── Apply inflammatory mediator effects ──────────────────────────
    if inflammatory_state is not None:
        ist = inflammatory_state
    else:
        ist = InflammatoryState()   # no-op defaults (all factors = 1.0)

    Kf_eff = renal.Kf * renal.Kf_scale * ist.Kf_factor
    R_AA0_eff = renal.R_AA0 * ist.R_AA_factor
    R_EA0_eff = renal.R_EA0 * ist.R_EA_factor
    RAAS_gain_eff = renal.RAAS_gain * ist.RAAS_gain_factor
    eta_PT_eff = min(renal.eta_PT + ist.eta_PT_offset, 0.85)
    MAP_sp_eff = renal.MAP_setpoint + ist.MAP_setpoint_offset

    # ── 1. RAAS ────────────────────────────────────────────────────────
    dMAP = MAP - MAP_sp_eff
    RAAS_factor = np.clip(1.0 - RAAS_gain_eff * 0.005 * dMAP, 0.5, 2.0)
    R_EA = R_EA0_eff * RAAS_factor
    eta_CD = renal.eta_CD0 * RAAS_factor   # aldosterone → CD reabsorption

    # ── 2. TGF iteration ──────────────────────────────────────────────
    R_AA = R_AA0_eff
    GFR = 120.0  # initial guess
    Na_filt = 0.0
    P_gc = 60.0
    RBF = 1100.0

    for _ in range(30):
        R_total = renal.R_preAA + R_AA + R_EA
        RBF = max((MAP - renal.P_renal_vein) / R_total * 1000.0, 100.0)
        RPF = RBF * (1.0 - renal.Hct)

        # Glomerular capillary pressure (Hallow Eq. for P_gc)
        P_gc = MAP - RBF / 1000.0 * (renal.R_preAA + R_AA)
        P_gc = max(P_gc, 25.0)

        # Average oncotic pressure (rises along capillary with filtration)
        FF = np.clip(GFR / max(RPF, 1.0), 0.01, 0.45)
        pi_avg = renal.pi_plasma * (1.0 + FF / (2.0 * (1.0 - FF)))

        # Starling equation → SNGFR
        NFP = max(P_gc - renal.P_Bow - pi_avg, 0.0)
        SNGFR = Kf_eff * NFP   # nL/min per nephron
        GFR = max(2.0 * renal.N_nephrons * SNGFR * 1e-6, 5.0)   # mL/min

        FF = np.clip(GFR / max(RPF, 1.0), 0.01, 0.45)

        # Tubular Na for TGF sensing
        Na_filt = GFR * renal.C_Na * 1e-3   # mEq/min
        MD_Na = Na_filt * (1.0 - eta_PT_eff) * (1.0 - renal.eta_LoH)

        if renal.TGF_setpoint <= 0:
            renal.TGF_setpoint = MD_Na

        TGF_err = (MD_Na - renal.TGF_setpoint) / max(renal.TGF_setpoint, 1e-6)
        R_AA_new = R_AA0_eff * (1.0 + renal.TGF_gain * TGF_err)
        R_AA_new = np.clip(R_AA_new, 0.5 * R_AA0_eff, 3.0 * R_AA0_eff)
        R_AA = 0.8 * R_AA + 0.2 * R_AA_new   # slow relaxation for stability

    # ── 3. Tubular Na handling ────────────────────────────────────────
    Na_after_PT  = Na_filt * (1.0 - eta_PT_eff)
    Na_after_LoH = Na_after_PT * (1.0 - renal.eta_LoH)
    Na_after_DT  = Na_after_LoH * (1.0 - renal.eta_DT)
    Na_after_CD  = Na_after_DT * (1.0 - eta_CD)

    # Pressure-natriuresis
    if MAP > MAP_sp_eff:
        pn = 1.0 + 0.03 * (MAP - MAP_sp_eff)
    else:
        pn = max(0.3, 1.0 + 0.015 * (MAP - MAP_sp_eff))

    Na_excr_min = Na_after_CD * pn       # mEq/min
    Na_excr_day = Na_excr_min * 1440.0   # mEq/day

    # ── 4. Water excretion ────────────────────────────────────────────
    water_excr_min = GFR * (1.0 - renal.frac_water_reabs)  # mL/min
    water_excr_day = water_excr_min * 1440.0 / 1000.0       # L/day

    # ── 5. Volume / Na balance ────────────────────────────────────────
    dt_min = dt_hours * 60.0

    Na_in_min = renal.Na_intake / 1440.0
    renal.Na_total = max(renal.Na_total + (Na_in_min - Na_excr_min) * dt_min,
                         800.0)

    W_in_min = renal.water_intake * 1000.0 / 1440.0   # mL/min
    dV = (W_in_min - water_excr_min) * dt_min
    renal.V_blood = np.clip(renal.V_blood + dV * 0.33, 3000.0, 8000.0)
    #   0.33 factor: only ~1/3 of ECF change appears in blood volume

    # Update plasma Na
    V_ECF = renal.V_blood / 0.33
    renal.C_Na = np.clip(renal.Na_total / (V_ECF * 1e-3), 125.0, 155.0)

    # ── 6. Store outputs ──────────────────────────────────────────────
    renal.GFR = GFR
    renal.RBF = RBF
    renal.P_glom = P_gc
    renal.Na_excretion = Na_excr_day
    renal.water_excretion = water_excr_day

    return renal


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 ─ Message-Passing Protocol
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeartToKidneyMessage:
    MAP: float        # mmHg
    CO:  float        # L/min
    Pven: float       # mmHg
    SBP: float        # mmHg
    DBP: float        # mmHg
    # ── Inflammatory mediator fields (for future ODE version) ────────
    # EDP: float = 10.0    # end-diastolic pressure → congestion source
    #                       # for inflammatory ODE

@dataclass
class KidneyToHeartMessage:
    V_blood: float    # mL  → converted to m³ for CircAdapt
    SVR_ratio: float  # dimensionless (new SVR / baseline SVR)
    GFR: float        # mL/min (informational)
    # ── Inflammatory mediator fields (for future ODE version) ────────
    # aldosterone_factor: float = 1.0  # RAAS state → inflammation source
    #                                   # for inflammatory ODE


def heart_to_kidney(hemo: Dict) -> HeartToKidneyMessage:
    return HeartToKidneyMessage(
        MAP=hemo['MAP'], CO=hemo['CO'], Pven=hemo['Pven'],
        SBP=hemo['SBP'], DBP=hemo['DBP'],
    )

def kidney_to_heart(renal: HallowRenalModel, MAP: float, CO: float,
                    Pven: float) -> KidneyToHeartMessage:
    """Compute SVR ratio from current MAP / CO vs baseline."""
    CVP = max(Pven, 0.5)
    SVR_current = (MAP - CVP) / max(CO, 0.3)
    SVR_baseline = (93.0 - 3.0) / 5.0   # ~18 mmHg·min/L
    return KidneyToHeartMessage(
        V_blood=renal.V_blood,
        SVR_ratio=SVR_current / SVR_baseline,
        GFR=renal.GFR,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 ─ Coupled Simulation Driver
# ═══════════════════════════════════════════════════════════════════════════

def run_coupled_simulation(
    n_steps: int = 8,
    dt_renal_hours: float = 6.0,
    cardiac_schedule: Optional[List[float]] = None,
    kidney_schedule:  Optional[List[float]] = None,
    stiffness_schedule: Optional[List[float]] = None,
    inflammation_schedule: Optional[List[float]] = None,
    diabetes_schedule: Optional[List[float]] = None,
) -> Dict:
    """
    Run the coupled cardiorenal simulation.

    Each coupling step:
        0. Update inflammatory mediator layer
        1. Apply inflammatory modifiers to heart
        2. Apply stiffness (k1_scale × inflammatory k1 factor)
        3. Apply deterioration (Sf_act × inflammatory Sf factor)
        4. Heart → steady state  (CircAdapt run(stable=True))
        5. Heart → Kidney message
        6. Kidney update  (Hallow equations + inflammatory effects)
        7. Kidney → Heart message
        8. Apply kidney feedback to heart model
        9. Record everything

    Parameters
    ----------
    stiffness_schedule : list of float or None
        Passive myocardial stiffness k1 scale for each step.
        1.0 = healthy; >1.0 = HFpEF diastolic dysfunction.
        Default: [1.0]*n_steps.
    inflammation_schedule : list of float or None
        Inflammation scale (0–1) for each step.  Default: [0]*n_steps.
    diabetes_schedule : list of float or None
        Diabetes scale (0–1) for each step.  Default: [0]*n_steps.
    """
    if cardiac_schedule is None:
        cardiac_schedule = [1.0] * n_steps
    if kidney_schedule is None:
        kidney_schedule = [1.0] * n_steps
    if stiffness_schedule is None:
        stiffness_schedule = [1.0] * n_steps
    if inflammation_schedule is None:
        inflammation_schedule = [0.0] * n_steps
    if diabetes_schedule is None:
        diabetes_schedule = [0.0] * n_steps

    # ── Initialise models ──────────────────────────────────────────────
    heart = CircAdaptHeartModel()
    renal = HallowRenalModel()
    ist = InflammatoryState()

    hist = {k: [] for k in [
        'step', 'PV_LV', 'PV_RV',
        'SBP', 'DBP', 'MAP', 'CO', 'SV', 'EF',
        'V_blood', 'GFR', 'Na_excr', 'P_glom',
        'Sf_scale', 'Kf_scale', 'k1_scale',
        'inflammation_scale', 'diabetes_scale',
        'effective_Sf', 'effective_Kf', 'effective_k1',
    ]}

    has_inflammation = any(x > 0 for x in inflammation_schedule)
    has_diabetes = any(x > 0 for x in diabetes_schedule)

    print("=" * 70)
    print("  CARDIORENAL COUPLING SIMULATOR")
    print("  Heart : CircAdapt VanOsta2024")
    print("  Kidney: Hallow et al. 2017 renal module")
    if has_inflammation or has_diabetes:
        print("  Mediator: Inflammatory layer (parametric scaling)")
    print("=" * 70)

    for s in range(n_steps):
        sf = cardiac_schedule[s] if s < len(cardiac_schedule) else cardiac_schedule[-1]
        kf = kidney_schedule[s] if s < len(kidney_schedule) else kidney_schedule[-1]
        k1 = stiffness_schedule[s] if s < len(stiffness_schedule) else stiffness_schedule[-1]
        infl = inflammation_schedule[s] if s < len(inflammation_schedule) else inflammation_schedule[-1]
        diab = diabetes_schedule[s] if s < len(diabetes_schedule) else diabetes_schedule[-1]

        print(f"\n{'─'*60}")
        print(f"  Step {s+1}/{n_steps}   "
              f"Sf_act={sf:.2f}   k1={k1:.2f}   Kf={kf:.2f}")
        if has_inflammation or has_diabetes:
            print(f"  Inflammation={infl:.2f}   Diabetes={diab:.2f}")
        print(f"{'─'*60}")

        # 0 ── Update inflammatory mediator layer ─────────────────────
        ist = update_inflammatory_state(ist, infl, diab)

        # 1 ── Apply inflammatory modifiers to heart ──────────────────
        heart.apply_inflammatory_modifiers(ist)

        # 2 ── Apply stiffness (HFpEF diastolic dysfunction) ──────────
        effective_k1 = k1 * ist.passive_k1_factor
        heart.apply_stiffness(effective_k1)

        # 3 ── Apply deterioration (contractility) ────────────────────
        effective_sf = max(sf * ist.Sf_act_factor, 0.20)  # stability floor
        heart.apply_deterioration(effective_sf)
        renal.Kf_scale = kf
        # Note: ist.Kf_factor is applied inside update_renal_model

        effective_kf = kf * ist.Kf_factor  # for recording only

        if has_inflammation or has_diabetes:
            print(f"  [Inflam] Sf_eff={effective_sf:.3f}  k1_eff={effective_k1:.3f}  "
                  f"Kf_eff={effective_kf:.3f}  "
                  f"p0_factor={ist.p0_factor:.3f}  "
                  f"stiffness={ist.stiffness_factor:.3f}")

        # 3 ── Heart to steady state ──────────────────────────────────
        print("  [Heart]  Running CircAdapt to steady state …")
        hemo = heart.run_to_steady_state()

        print(f"  [Heart]  MAP={hemo['MAP']:.1f}  "
              f"SBP/DBP={hemo['SBP']:.0f}/{hemo['DBP']:.0f}  "
              f"CO={hemo['CO']:.2f} L/min  SV={hemo['SV']:.1f} mL  "
              f"EF={hemo['EF']:.0f}%")

        # 4 ── Heart → Kidney ─────────────────────────────────────────
        h2k = heart_to_kidney(hemo)
        print(f"  [H→K]  MAP={h2k.MAP:.1f}  CO={h2k.CO:.2f}  Pven={h2k.Pven:.1f}")

        # 5 ── Kidney update ──────────────────────────────────────────
        print(f"  [Kidney] Updating renal model (dt={dt_renal_hours}h) …")
        renal = update_renal_model(renal, h2k.MAP, h2k.CO,
                                   h2k.Pven, dt_renal_hours,
                                   inflammatory_state=ist)
        print(f"  [Kidney] GFR={renal.GFR:.1f} mL/min   "
              f"V_blood={renal.V_blood:.0f} mL   "
              f"Na_excr={renal.Na_excretion:.0f} mEq/day")

        # 6 ── Kidney → Heart ─────────────────────────────────────────
        k2h = kidney_to_heart(renal, h2k.MAP, h2k.CO, h2k.Pven)
        print(f"  [K→H]  V_blood={k2h.V_blood:.0f} mL   "
              f"SVR_ratio={k2h.SVR_ratio:.3f}   GFR={k2h.GFR:.1f}")

        # 7 ── Apply kidney feedback ──────────────────────────────────
        heart.apply_kidney_feedback(
            V_blood_m3=k2h.V_blood * ML_TO_M3,
            SVR_ratio=k2h.SVR_ratio,
        )

        # 8 ── Record ─────────────────────────────────────────────────
        hist['step'].append(s + 1)
        hist['PV_LV'].append((hemo['V_LV'].copy(), hemo['p_LV'].copy()))
        hist['PV_RV'].append((hemo['V_RV'].copy(), hemo['p_RV'].copy()))
        hist['SBP'].append(hemo['SBP'])
        hist['DBP'].append(hemo['DBP'])
        hist['MAP'].append(hemo['MAP'])
        hist['CO'].append(hemo['CO'])
        hist['SV'].append(hemo['SV'])
        hist['EF'].append(hemo['EF'])
        hist['V_blood'].append(renal.V_blood)
        hist['GFR'].append(renal.GFR)
        hist['Na_excr'].append(renal.Na_excretion)
        hist['P_glom'].append(renal.P_glom)
        hist['Sf_scale'].append(sf)
        hist['Kf_scale'].append(kf)
        hist['k1_scale'].append(k1)
        hist['inflammation_scale'].append(infl)
        hist['diabetes_scale'].append(diab)
        hist['effective_Sf'].append(effective_sf)
        hist['effective_Kf'].append(effective_kf)
        hist['effective_k1'].append(effective_k1)

    print(f"\n{'='*70}")
    print("  SIMULATION COMPLETE")
    print(f"{'='*70}\n")
    return hist


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 ─ Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(hist: Dict, title: str, save_path: str):
    """12-panel dark-theme dashboard of coupled simulation results."""
    n = len(hist['step'])
    steps = np.array(hist['step'])

    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor('#080810')
    gs = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.05, right=0.97, top=0.91, bottom=0.06)

    cmap = plt.cm.RdYlGn_r
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    C = dict(txt='#d8d8e8', grid='#222238', red='#ff6b6b', teal='#4ecdc4',
             gold='#ffd93d', mint='#a8e6cf', lilac='#c9b1ff', peach='#ffb385',
             sky='#87ceeb')

    def sty(ax, t, xl, yl):
        ax.set_facecolor('#0e0e18')
        ax.set_title(t, color=C['txt'], fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(xl, color=C['txt'], fontsize=9)
        ax.set_ylabel(yl, color=C['txt'], fontsize=9)
        ax.tick_params(colors=C['txt'], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(C['grid'])
        ax.grid(True, alpha=0.12, color=C['grid'])

    # ── Row 1 : PV loops ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    sty(ax1, 'LV Pressure–Volume Loop  (CircAdapt VanOsta2024)',
        'Volume [mL]', 'Pressure [mmHg]')
    for i, (V, P) in enumerate(hist['PV_LV']):
        ax1.plot(V, P, color=colors[i], lw=2, alpha=0.85,
                 label=f'Step {i+1}  Sf={hist["Sf_scale"][i]:.2f}')
    ax1.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'], loc='upper right', ncol=2)

    ax2 = fig.add_subplot(gs[0, 2:])
    sty(ax2, 'RV Pressure–Volume Loop', 'Volume [mL]', 'Pressure [mmHg]')
    for i, (V, P) in enumerate(hist['PV_RV']):
        ax2.plot(V, P, color=colors[i], lw=2, alpha=0.85,
                 label=f'Step {i+1}')
    ax2.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'], loc='upper right', ncol=2)

    # ── Row 2 ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sty(ax3, 'Systolic / Diastolic BP', 'Coupling Step', 'mmHg')
    ax3.fill_between(steps, hist['DBP'], hist['SBP'], alpha=0.2, color=C['red'])
    ax3.plot(steps, hist['SBP'], 'o-', color=C['red'], lw=2, ms=6, label='SBP')
    ax3.plot(steps, hist['DBP'], 's-', color=C['teal'], lw=2, ms=6, label='DBP')
    ax3.plot(steps, hist['MAP'], '^-', color=C['gold'], lw=1.5, ms=5, label='MAP')
    ax3.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'])

    ax4 = fig.add_subplot(gs[1, 1])
    sty(ax4, 'Blood Volume  (Kidney → Heart)', 'Coupling Step', 'mL')
    ax4.plot(steps, hist['V_blood'], 'o-', color=C['mint'], lw=2.5, ms=7)
    ax4.axhline(5000, color=C['txt'], ls='--', alpha=0.25, label='Baseline')
    ax4.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'])

    ax5 = fig.add_subplot(gs[1, 2])
    sty(ax5, 'GFR  (Hallow Kidney)', 'Coupling Step', 'mL/min')
    ax5.plot(steps, hist['GFR'], 'o-', color=C['lilac'], lw=2.5, ms=7)
    ax5.axhline(120, color=C['txt'], ls='--', alpha=0.25, label='Normal ≈120')
    ax5.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'])

    ax6 = fig.add_subplot(gs[1, 3])
    sty(ax6, 'Cardiac Output', 'Coupling Step', 'L/min')
    ax6.plot(steps, hist['CO'], 'o-', color=C['gold'], lw=2.5, ms=7)
    ax6.axhline(5.0, color=C['txt'], ls='--', alpha=0.25, label='Normal ≈5')
    ax6.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'])

    # ── Row 3 ─────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    sty(ax7, 'Stroke Volume & Ejection Fraction', 'Coupling Step', 'SV [mL]')
    ax7.plot(steps, hist['SV'], 'o-', color=C['sky'], lw=2.5, ms=7, label='SV')
    ax7b = ax7.twinx()
    ax7b.plot(steps, hist['EF'], 's--', color=C['peach'], lw=1.8, ms=5, label='EF%')
    ax7b.set_ylabel('EF [%]', color=C['peach'], fontsize=9)
    ax7b.tick_params(colors=C['peach'], labelsize=8)
    lines1, lab1 = ax7.get_legend_handles_labels()
    lines2, lab2 = ax7b.get_legend_handles_labels()
    ax7.legend(lines1+lines2, lab1+lab2, fontsize=7, facecolor='#14142a',
               edgecolor=C['grid'], labelcolor=C['txt'])

    ax8 = fig.add_subplot(gs[2, 1])
    sty(ax8, 'Glomerular Pressure', 'Coupling Step', 'P_gc [mmHg]')
    ax8.plot(steps, hist['P_glom'], 'o-', color=C['peach'], lw=2.5, ms=7)

    ax9 = fig.add_subplot(gs[2, 2])
    sty(ax9, 'Na Excretion', 'Coupling Step', 'mEq/day')
    ax9.plot(steps, hist['Na_excr'], 'o-', color=C['mint'], lw=2.5, ms=7)
    ax9.axhline(150, color=C['txt'], ls='--', alpha=0.25, label='Intake=150')
    ax9.legend(fontsize=7, facecolor='#14142a', edgecolor=C['grid'],
               labelcolor=C['txt'])

    ax10 = fig.add_subplot(gs[2, 3])
    sty(ax10, 'Deterioration Parameters', 'Coupling Step', 'Scale (1=healthy)')
    ax10.plot(steps, hist['Sf_scale'], 'o-', color=C['red'], lw=2.5, ms=7,
              label='Cardiac  (Sf_act_scale)')
    ax10.plot(steps, hist['Kf_scale'], 's-', color=C['teal'], lw=2.5, ms=7,
              label='Kidney  (Kf_scale)')
    ax10.set_ylim(0, 1.15)
    ax10.axhline(1.0, color=C['txt'], ls=':', alpha=0.2)
    ax10.legend(fontsize=8, facecolor='#14142a', edgecolor=C['grid'],
                labelcolor=C['txt'])

    fig.suptitle(title, color=C['txt'], fontsize=15, fontweight='bold', y=0.96)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ─ Run 3 Scenarios
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N = 8

    # ── Scenario 1: Progressive HFrEF ─────────────────────────────────
    print("\n" + "█"*70)
    print("  SCENARIO 1 : Progressive Heart Failure (HFrEF)")
    print("█"*70)
    h1 = run_coupled_simulation(
        n_steps=N, dt_renal_hours=6.0,
        cardiac_schedule=[1.0, 0.92, 0.82, 0.72, 0.62, 0.52, 0.45, 0.40],
        kidney_schedule=[1.0]*N,
    )
    plot_results(h1,
        'Scenario 1: Progressive HFrEF\n'
        'Cardiac Sf_act_scale ↓  •  Kidney healthy',
        'scenario1_hfref.png')

    # ── Scenario 2: Progressive CKD ───────────────────────────────────
    print("\n" + "█"*70)
    print("  SCENARIO 2 : Progressive Chronic Kidney Disease")
    print("█"*70)
    h2 = run_coupled_simulation(
        n_steps=N, dt_renal_hours=6.0,
        cardiac_schedule=[1.0]*N,
        kidney_schedule=[1.0, 0.88, 0.76, 0.64, 0.52, 0.42, 0.35, 0.30],
    )
    plot_results(h2,
        'Scenario 2: Progressive CKD\n'
        'Heart healthy  •  Kidney Kf_scale ↓',
        'scenario2_ckd.png')

    # ── Scenario 3: Cardiorenal syndrome ──────────────────────────────
    print("\n" + "█"*70)
    print("  SCENARIO 3 : Combined Cardiorenal Syndrome")
    print("█"*70)
    h3 = run_coupled_simulation(
        n_steps=N, dt_renal_hours=6.0,
        cardiac_schedule=[1.0, 0.94, 0.86, 0.78, 0.70, 0.62, 0.55, 0.48],
        kidney_schedule= [1.0, 0.94, 0.86, 0.76, 0.66, 0.56, 0.46, 0.38],
    )
    plot_results(h3,
        'Scenario 3: Combined Cardiorenal Syndrome\n'
        'Simultaneous cardiac + kidney deterioration',
        'scenario3_crs.png')

    print("\nAll scenarios complete.")
    print("Output: scenario1_hfref.png, scenario2_ckd.png, scenario3_crs.png")
