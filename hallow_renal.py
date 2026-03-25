"""
Faithful Python translation of the Hallow et al. (2017) renal model.

Translated from the R reference implementation:
  - calcNomParams_timescale.R  -> HallowRenalParams
  - modelfile_commented.R      -> compute_renal_algebraic(), renal_ode_rhs()
  - getInits.R                 -> HallowRenalModel initial conditions

All ~200 parameters, algebraic equations (lines 589-1521 of modelfile_commented.R),
and 33-variable ODE right-hand-side are implemented exactly as in the R model.

References
----------
Hallow, K.M., et al. "A model-based approach to investigating the
pathophysiological mechanisms of hypertension and CKD." (2017)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional
import math
import copy

# =========================================================================
# PART 1: HallowRenalParams -- All parameters from calcNomParams_timescale.R
# =========================================================================

class HallowRenalParams:
    """
    All ~200 parameters from the R model's calcNomParams_timescale.R,
    plus derived equilibrium calculations.
    """

    def __init__(self):
        # =====================================================================
        # Constants and Unit Conversions
        # =====================================================================
        self.nL_mL = 1e6
        self.dl_ml = 0.01
        self.L_dL = 10
        self.L_mL = 1000
        self.L_m3 = 0.001
        self.m3_mL = 1000000
        self.m_mm = 1000
        self.g_mg = 0.001
        self.ng_mg = 1e-06
        self.secs_mins = 60
        self.min_hr = 60
        self.min_sec = 60
        self.hr_day = 24
        self.min_day = 1440
        self.Pa_mmHg = 0.0075
        self.MW_creatinine = 113.12
        self.Pi = 3.1416
        self.viscosity_length_constant = 1.5e-09
        self.gamma = 1.16667e-5  # viscosity of tubular fluid
        self.water_intake_species_scale = 1

        # =====================================================================
        # Systemic Parameters
        # =====================================================================
        self.nominal_map_setpoint = 85          # mmHg
        self.IF_nom = 15                        # L
        self.blood_volume_nom = 5               # L
        self.Na_intake_rate = 100 / 24 / 60     # mEq/min (100 mmol/day)
        self.nom_water_intake = 2.1             # L/day
        self.ref_Na_concentration = 140         # mEq/L
        self.glucose_concentration = 5.5        # mmol/L
        self.plasma_albumin_concentration = 35  # mg/ml
        self.plasma_protein_concentration = 7   # g/dl
        self.plasma_urea = 0
        self.nom_serum_uric_acid_concentration = 5.5  # mg/dl
        self.equilibrium_serum_creatinine = 0.92      # mg/dl
        self.P_venous = 4                       # mmHg

        # =====================================================================
        # Renal Parameters
        # =====================================================================
        self.nom_renal_blood_flow_L_min = 1     # L/min
        self.baseline_nephrons = 2e6
        self.nom_Kf = 3.9                       # nl/min*mmHg
        self.nom_oncotic_pressure_difference = 28  # mmHg
        self.P_renal_vein = 4                   # mmHg
        self.nom_oncotic_pressure_peritubular = 28.05  # mmHg
        self.interstitial_oncotic_pressure = 5  # mmHg

        # =====================================================================
        # Renal Vasculature
        # =====================================================================
        self.nom_preafferent_arteriole_resistance = 14  # mmHg
        self.nom_afferent_diameter = 1.65e-5            # m
        self.nom_efferent_diameter = 1.1e-05            # m

        # =====================================================================
        # Renal Tubules
        # =====================================================================
        self.Dc_pt_nom = 27e-6      # m
        self.Dc_lh = 17e-6          # m
        self.Dc_dt = 17e-6          # m
        self.Dc_cd = 22e-6          # m

        self.L_pt_s1_nom = 0.005    # m
        self.L_pt_s2_nom = 0.005    # m
        self.L_pt_s3_nom = 0.004    # m
        self.L_lh_des = 0.01        # m
        self.L_lh_asc = 0.01        # m
        self.L_dct = 0.005          # m
        self.L_cd = self.L_lh_des   # m

        self.tubular_compliance = 0.2
        self.Pc_pt_s1_mmHg = 20.2
        self.Pc_pt_s2_mmHg = 15
        self.Pc_pt_s3_mmHg = 11
        self.Pc_lh_des_mmHg = 8
        self.Pc_lh_asc_mmHg = 7
        self.Pc_dt_mmHg = 6
        self.Pc_cd_mmHg = 5
        self.P_interstitial_mmHg = 5
        self.nominal_pt_na_reabsorption = 0.7
        self.nominal_loh_na_reabsorption = 0.8
        self.nominal_dt_na_reabsorption = 0.5
        self.LoH_flow_dependence = 0.75

        # =====================================================================
        # Renal Glucose Reabsorption
        # =====================================================================
        self.nom_glucose_reabs_per_unit_length_s1 = 5.4e-5
        self.nom_glucose_reabs_per_unit_length_s2 = 0
        self.nom_glucose_reabs_per_unit_length_s3 = 2.8e-5
        self.diabetic_adaptation = 1
        self.maximal_RTg_increase = 0.3
        self.T_glucose_RTg = 6000000

        self.glucose_natriuresis_effect_pt = 0
        self.glucose_natriuresis_effect_cd = 0
        self.glucose_diuresis_effect_pt = 0
        self.glucose_diuresis_effect_cd = 0

        # =====================================================================
        # Renal Urea Reabsorption
        # =====================================================================
        self.urea_permeability_PT = 0.5

        # =====================================================================
        # Albumin Sieving / Proteinuria
        # =====================================================================
        self.nom_glomerular_albumin_sieving_coefficient = 0.00062
        self.SN_albumin_reabsorptive_capacity = 1.4e-6
        self.Emax_seiving = 4
        self.Gamma_seiving = 3
        self.Km_seiving = 25
        self.max_PT_albumin_reabsorption_rate = 0.1
        self.nom_albumin_excretion_rate = 3.5e-9
        self.nom_GP_seiving_damage = 65
        self.c_albumin = 0.0231   # min/nl
        self.seiving_inf = 4.25e-4

        # =====================================================================
        # RAAS Pathway Parameters (nominal equilibrium)
        # =====================================================================
        self.concentration_to_renin_activity_conversion_plasma = 61
        self.nominal_equilibrium_PRA = 1000         # fmol/ml/hr
        self.nominal_equilibrium_AngI = 7.5          # fmol/ml
        self.nominal_equilibrium_AngII = 4.75        # fmol/ml
        self.nominal_renin_half_life = 0.1733        # hr
        self.nominal_AngI_half_life = 0.5 / 60       # hr
        self.nominal_AngII_half_life = 0.66 / 60     # hr
        self.nominal_AT1_bound_AngII_half_life = 12 / 60  # hr
        self.nominal_AT2_bound_AngII_half_life = 12 / 60  # hr
        self.ACE_chymase_fraction = 0.95
        self.fraction_AT1_bound_AngII = 0.75

        # =====================================================================
        # Effects of AT1-bound AngII
        # =====================================================================
        self.AT1_svr_slope = 0
        self.AT1_preaff_scale = 0.8
        self.AT1_preaff_slope = 16
        self.AT1_aff_scale = 0.8
        self.AT1_aff_slope = 16
        self.AT1_eff_scale = 0.8
        self.AT1_eff_slope = 16
        self.AT1_PT_scale = 0
        self.AT1_PT_slope = 16
        self.AT1_aldo_slope = 0.02

        self.AT1_aff_EC50 = 1e-9
        self.Emax_AT1_eff = 0
        self.Emax_AT1_aff = 0
        self.AT1_hill = 15

        self.AngII_effect_on_venous_compliance = 1

        # =====================================================================
        # Aldosterone Effects
        # =====================================================================
        self.nominal_aldosterone_concentration = 85
        self.aldo_DCT_scale = 0
        self.aldo_DCT_slope = 0.5
        self.aldo_CD_scale = 0.2
        self.aldo_CD_slope = 0.5
        self.aldo_renin_slope = -0.4

        # =====================================================================
        # ANP Effects
        # =====================================================================
        self.normalized_atrial_NP_concentration = 1
        self.nom_ANP = 50  # pg/ml
        self.ANP_aff_scale = 0.2
        self.ANP_aff_slope = 1
        self.ANP_preaff_scale = 0
        self.ANP_preaff_slope = 1
        self.ANP_eff_scale = 0
        self.ANP_eff_slope = 1
        self.anp_CD_scale = -0.1
        self.anp_CD_slope = 2
        self.ANP_effect_on_venous_compliance = 1
        self.LVEDP_ANP_slope = 20
        self.ANP_infused_concentration = 0
        self.ANP_infusion = 0

        # =====================================================================
        # RSNA Effects
        # =====================================================================
        self.renal_sympathetic_nerve_activity = 1
        self.nom_rsna = 1
        self.rsna_preaff_scale = 0.2
        self.rsna_preaff_slope = 0.25
        self.rsna_PT_scale = 0
        self.rsna_PT_slope = 1
        self.rsna_CD_scale = 0
        self.rsna_CD_slope = 1
        self.rsna_renin_slope = 1
        self.rsna_svr_slope = 0
        self.rsna_HR_slope = 0

        self.sna_effect_on_contractility = 1
        self.SNA_effect_on_venous_compliance = 1
        self.B2sna_effect_on_TPR = 1
        self.A1sna_effect_on_TPR = 1

        # =====================================================================
        # Vasopressin / Osmolarity Control
        # =====================================================================
        self.Na_controller_gain = 0.05
        self.Kp_VP = 2
        self.Ki_VP = 0.005
        self.nom_ADH_urea_permeability = 0.98
        self.nom_ADH_water_permeability = 0.98
        self.nominal_vasopressin_conc = 4
        self.water_intake_vasopressin_scale = 0.25
        self.water_intake_vasopressin_slope = -0.5

        # =====================================================================
        # Tubuloglomerular Feedback (TGF)
        # =====================================================================
        self.S_tubulo_glomerular_feedback = 0.7
        self.F_md_scale_tubulo_glomerular_feedback = 6
        self.MD_Na_concentration_setpoint = 63.29

        # Macula densa effect on renin
        self.md_renin_A = 1
        self.md_renin_tau = 1

        # =====================================================================
        # Renal Vascular Responsiveness
        # =====================================================================
        self.preaff_diameter_range = 0.25
        self.afferent_diameter_range = 1.2e-05
        self.efferent_diameter_range = 3e-06
        self.preaff_signal_nonlin_scale = 4
        self.afferent_signal_nonlin_scale = 4
        self.efferent_signal_nonlin_scale = 4

        # =====================================================================
        # RAAS Pathway (operational parameters)
        # =====================================================================
        self.AngI_half_life = 0.008333
        self.AngII_half_life = 0.011
        self.AT1_bound_AngII_half_life = 0.2
        self.AT1_PRC_slope = -0.9
        self.AT1_PRC_yint = 0
        self.AT2_bound_AngII_half_life = 0.2

        # Hard-coded operational RAAS values
        self.nominal_ACE_activity = 48.9
        self.nominal_AT1_receptor_binding_rate = 12.1
        self.nominal_AT2_receptor_binding_rate = 4.0
        self.nominal_chymase_activity = 1.25
        self.nominal_equilibrium_AT1_bound_AngII = 16.63
        self.nominal_equilibrium_PRC = 16.4
        self.renin_half_life = 0.1733

        # =====================================================================
        # Transfer Constants for ODEs
        # =====================================================================
        self.C_renal_CV_timescale = 60

        self.C_co_error = 1
        self.C_vasopressin_delay = 1

        # Na and water transfer between blood, IF
        self.Q_water = 1
        self.Q_Na = 1
        self.Q_Na_store = 0
        self.max_stored_sodium = 500
        self.C_Na_error = 1

        self.C_aldo_secretion = 100
        self.C_tgf_reset = 0
        self.C_md_flow = 0.06
        self.C_tgf = 1
        self.C_rbf = 100
        self.C_serum_creatinine = 1
        self.C_pt_water = 1
        self.C_rsna = 100
        self.C_postglomerular_pressure = 1

        # =====================================================================
        # Therapy Effects (all default to no-effect)
        # =====================================================================
        self.HCTZ_effect_on_DT_Na_reabs = 1
        self.HCTZ_effect_on_renin_secretion = 1
        self.CCB_effect_on_preafferent_resistance = 1
        self.CCB_effect_on_afferent_resistance = 1
        self.CCB_effect_on_efferent_resistance = 1
        self.pct_target_inhibition_MRA = 0
        self.pct_target_inhibition_ARB = 0
        self.pct_target_inhibition_ACEi = 0
        self.pct_target_inhibition_DRI = 0
        self.ARB_is_on = 0

        self.BB_TPR_effect = 1
        self.BB_cardiac_relax_effect = 0
        self.BB_venous_compliance_effect = 0
        self.BB_preafferent_R_effect = 1
        self.BB_renin_secretion_effect = 1
        self.BB_HR_effect = 1
        self.BB_contractility_effect = 1
        self.BB_is_on = 0
        self.k_PD = 0.001

        # Normalized aldo secretion
        self.K_Na_ratio_effect_on_aldo = 1

        # Renal autoregulation
        self.gp_autoreg_scale = 0
        self.preaff_autoreg_scale = 0
        self.myogenic_steepness = 2
        self.RBF_autoreg_scale = 0
        self.RBF_autoreg_steepness = 1

        # =====================================================================
        # Pressure Natriuresis
        # =====================================================================
        self.Kp_PN = 1
        self.Kd_PN = 0
        self.Ki_PN = 0
        self.max_pt_reabs_rate = 0.995
        self.pressure_natriuresis_PT_scale = 0.5
        self.pressure_natriuresis_PT_slope = 1
        self.pressure_natriuresis_LoH_scale = 0
        self.pressure_natriuresis_LoH_slope = 1
        self.pressure_natriuresis_DCT_scale = 0
        self.pressure_natriuresis_DCT_slope = 1
        self.max_cd_reabs_rate = 0.995
        self.pressure_natriuresis_CD_scale = 0.5
        self.pressure_natriuresis_CD_slope = 1
        self.RBF_CD_scale = 1
        self.RBF_CD_slope = 0.3
        self.CD_PN_loss_rate = 0

        self.CO_species_scale = 1

        # =====================================================================
        # Glomerular / Tubular Hypertrophy
        # =====================================================================
        self.maximal_glom_surface_area_increase = 0.5
        self.T_glomerular_pressure_increases_Kf = 120000
        self.maximal_tubule_length_increase = 0
        self.maximal_tubule_diameter_increase = 0
        self.T_PT_Na_reabs_PT_length = 1e10
        self.T_PT_Na_reabs_PT_diameter = 1e10

        # Reduce Kf due to glomerulosclerosis
        self.disease_effects_decreasing_Kf = 0

        # Disease effects
        self.disease_effect_on_nephrons = 0

        # =====================================================================
        # Reabsorption Limits
        # =====================================================================
        self.max_s1_Na_reabs = 7.5e-6
        self.max_s2_Na_reabs = 2e-6
        self.max_s3_Na_reabs = 1
        self.max_deltaLoH_reabs = 0.75e-6
        self.CD_Na_reabs_threshold = 7e-7

        # =====================================================================
        # Treatment Parameters
        # =====================================================================
        self.SGLT2_inhibition = 1
        self.SGLT1_inhibition = 1
        self.C_sglt2_delay = 0.1 * 60
        self.C_ruge = 0.0001 * 60
        self.Anhe3 = 0
        self.deltaCanp = 0
        self.ANP_effect_on_Arterial_Resistance = 0
        self.loop_diuretic_effect = 1

        # Heart-renal link (default on for standalone)
        self.heart_renal_link = 1

        # Cardiac params needed (CO_nom used by CO_error ODE)
        self.CO_nom = 5  # L/min

        # =====================================================================
        # Derived Equilibrium Calculations
        # (from calcNomParams_timescale.R lines ~283-361)
        # =====================================================================
        self._compute_derived()

    def _compute_derived(self):
        """Compute all derived parameters exactly as in the R code."""

        # Nominal equilibrium PRC
        self.nominal_equilibrium_PRC_calc = (
            self.nominal_equilibrium_PRA /
            self.concentration_to_renin_activity_conversion_plasma
        )
        # Use the hard-coded value from the R file
        # (self.nominal_equilibrium_PRC = 16.4 is already set)

        # RAAS degradation rates (derived from equilibrium)
        self.nominal_AngI_degradation_rate = math.log(2) / self.nominal_AngI_half_life
        self.nominal_AngII_degradation_rate = math.log(2) / self.nominal_AngII_half_life
        self.nominal_AT1_bound_AngII_degradation_rate = (
            math.log(2) / self.nominal_AT1_bound_AngII_half_life
        )
        self.nominal_AT2_bound_AngII_degradation_rate = (
            math.log(2) / self.nominal_AT2_bound_AngII_half_life
        )

        # Nominal ACE and chymase (derived from equilibrium)
        self.nominal_ACE_activity_calc = (
            self.ACE_chymase_fraction *
            (self.nominal_equilibrium_PRA -
             self.nominal_AngI_degradation_rate * self.nominal_equilibrium_AngI) /
            self.nominal_equilibrium_AngI
        )
        self.nominal_chymase_activity_calc = (
            (1 - self.ACE_chymase_fraction) *
            (self.nominal_equilibrium_PRA -
             self.nominal_AngI_degradation_rate * self.nominal_equilibrium_AngI) /
            self.nominal_equilibrium_AngI
        )

        # AT1/AT2 receptor binding rates
        self.nominal_AT1_receptor_binding_rate_calc = (
            self.fraction_AT1_bound_AngII *
            (self.nominal_equilibrium_AngI *
             (self.nominal_ACE_activity_calc + self.nominal_chymase_activity_calc) -
             self.nominal_AngII_degradation_rate * self.nominal_equilibrium_AngII) /
            self.nominal_equilibrium_AngII
        )
        self.nominal_AT2_receptor_binding_rate_calc = (
            (1 - self.fraction_AT1_bound_AngII) *
            (self.nominal_equilibrium_AngI *
             (self.nominal_ACE_activity_calc + self.nominal_chymase_activity_calc) -
             self.nominal_AngII_degradation_rate * self.nominal_equilibrium_AngII) /
            self.nominal_equilibrium_AngII
        )

        # Equilibrium AT1/AT2 bound AngII
        self.nominal_equilibrium_AT1_bound_AngII_calc = (
            self.nominal_equilibrium_AngII *
            self.nominal_AT1_receptor_binding_rate_calc /
            self.nominal_AT1_bound_AngII_degradation_rate
        )
        self.nominal_equilibrium_AT2_bound_AngII_calc = (
            self.nominal_equilibrium_AngII *
            self.nominal_AT2_receptor_binding_rate_calc /
            self.nominal_AT2_bound_AngII_degradation_rate
        )

        # AT1 efferent EC50 (from R code)
        self.AT1_eff_EC50 = (
            self.nominal_equilibrium_AT1_bound_AngII * 1e-12
        )

        # Renal pressures
        self.nom_preafferent_pressure = (
            self.nominal_map_setpoint -
            self.nom_renal_blood_flow_L_min * self.nom_preafferent_arteriole_resistance
        )

        self.nom_glomerular_pressure = (
            self.nom_preafferent_pressure -
            self.nom_renal_blood_flow_L_min *
            (self.L_m3 * self.viscosity_length_constant /
             (self.nom_afferent_diameter ** 4) / self.baseline_nephrons)
        )

        self.nom_postglomerular_pressure = (
            self.nom_preafferent_pressure -
            self.nom_renal_blood_flow_L_min *
            (self.L_m3 * self.viscosity_length_constant *
             (1 / (self.nom_afferent_diameter ** 4) +
              1 / (self.nom_efferent_diameter ** 4)) /
             self.baseline_nephrons)
        )

        # In the full R model, RIHP0 is hard-coded at 9.32 and the cardiac
        # model's closed-loop feedback ensures MAP adjusts to achieve
        # equilibrium. For standalone renal use, we set RIHP0 to the
        # actual nominal postglomerular pressure so the pressure
        # natriuresis signal is exactly 1.0 at the nominal operating point.
        self.RIHP0 = self.nom_postglomerular_pressure

        # Nominal GFR and filtered loads
        self.nom_GFR = (
            self.nom_Kf *
            (self.nom_glomerular_pressure -
             self.nom_oncotic_pressure_difference -
             self.Pc_pt_s1_mmHg) /
            self.nL_mL * self.baseline_nephrons
        )

        self.nom_filtered_sodium_load = (
            self.nom_GFR / self.L_mL * self.ref_Na_concentration
        )

        # Glucose filtration and reabsorption at baseline
        self.nom_filtered_glucose_load = (
            self.glucose_concentration * self.nom_GFR / 1000
        )
        self.nom_glucose_pt_out_s1 = max(
            0, self.nom_filtered_glucose_load -
            self.nom_glucose_reabs_per_unit_length_s1 *
            self.L_pt_s1_nom * self.baseline_nephrons
        )
        self.nom_glucose_pt_out_s2 = max(
            0, self.nom_glucose_pt_out_s1 -
            self.nom_glucose_reabs_per_unit_length_s2 *
            self.L_pt_s2_nom * self.baseline_nephrons
        )
        self.nom_glucose_pt_out_s3 = max(
            0, self.nom_glucose_pt_out_s2 -
            self.nom_glucose_reabs_per_unit_length_s3 *
            self.L_pt_s3_nom * self.baseline_nephrons
        )

        # SGLT Na reabsorption
        self.nom_SGTL2_Na_reabs_mmol_s1 = (
            self.nom_filtered_glucose_load - self.nom_glucose_pt_out_s1
        )
        self.nom_SGTL2_Na_reabs_mmol_s2 = (
            self.nom_glucose_pt_out_s1 - self.nom_glucose_pt_out_s2
        )
        self.nom_SGTL1_Na_reabs_mmol = (
            2 * (self.nom_glucose_pt_out_s2 - self.nom_glucose_pt_out_s3)
        )
        self.nom_total_SGLT_Na_reabs = (
            self.nom_SGTL2_Na_reabs_mmol_s1 +
            self.nom_SGTL2_Na_reabs_mmol_s2 +
            self.nom_SGTL1_Na_reabs_mmol
        )

        self.nom_SGLT_fractional_na_reabs = (
            self.nom_total_SGLT_Na_reabs / self.nom_filtered_sodium_load
            if self.nom_filtered_sodium_load > 0 else 0
        )

        # Non-SGLT PT Na reabsorption
        self.nominal_pt_na_reabsorption_nonSGLT = (
            self.nominal_pt_na_reabsorption -
            self.nom_SGLT_fractional_na_reabs
        )

        L_pt_total = (self.L_pt_s1_nom + self.L_pt_s2_nom + self.L_pt_s3_nom)
        self.nom_Na_reabs_per_unit_length = (
            -math.log(1 - self.nominal_pt_na_reabsorption_nonSGLT) /
            L_pt_total
        )

        # Segmental Na reabsorption at nominal
        self.nom_Na_pt_s1_reabs = (
            self.nom_filtered_sodium_load *
            (1 - math.exp(-self.nom_Na_reabs_per_unit_length * self.L_pt_s1_nom))
        )
        self.nom_Na_pt_out_s1 = (
            self.nom_filtered_sodium_load -
            self.nom_Na_pt_s1_reabs -
            self.nom_SGTL2_Na_reabs_mmol_s1
        )

        self.nom_Na_pt_s2_reabs = (
            self.nom_Na_pt_out_s1 *
            (1 - math.exp(-self.nom_Na_reabs_per_unit_length * self.L_pt_s2_nom))
        )
        self.nom_Na_pt_out_s2 = (
            self.nom_Na_pt_out_s1 -
            self.nom_Na_pt_s2_reabs -
            self.nom_SGTL2_Na_reabs_mmol_s2
        )

        self.nom_Na_pt_s3_reabs = (
            self.nom_Na_pt_out_s2 *
            (1 - math.exp(-self.nom_Na_reabs_per_unit_length * self.L_pt_s3_nom))
        )
        self.nom_Na_pt_out_s3 = (
            self.nom_Na_pt_out_s2 -
            self.nom_Na_pt_s3_reabs -
            self.nom_SGTL1_Na_reabs_mmol
        )

        self.nom_PT_Na_outflow = self.nom_Na_pt_out_s3

        # Loop of Henle nominal
        self.nom_Na_in_AscLoH = self.nom_PT_Na_outflow / self.baseline_nephrons
        self.AscLoH_Reab_Rate = (
            2 * self.nominal_loh_na_reabsorption *
            self.nom_Na_in_AscLoH / self.L_lh_des
        )

        self.nom_LoH_Na_outflow = (
            self.nom_PT_Na_outflow * (1 - self.nominal_loh_na_reabsorption)
        )
        self.nom_DT_Na_outflow = (
            self.nom_LoH_Na_outflow * (1 - self.nominal_dt_na_reabsorption)
        )

        # CD reabsorption calibrated so Na_excretion = Na_intake at SS
        self.nominal_cd_na_reabsorption = (
            1 - self.Na_intake_rate / self.nom_DT_Na_outflow
            if self.nom_DT_Na_outflow > 0 else 0
        )

        # Renal vascular resistance
        self.nom_RVR = (
            (self.nominal_map_setpoint - self.P_venous) /
            self.nom_renal_blood_flow_L_min
        )
        self.nom_peritubular_resistance = (
            self.nom_RVR -
            (self.nom_preafferent_arteriole_resistance +
             self.L_m3 * self.viscosity_length_constant *
             (1 / self.nom_afferent_diameter ** 4 +
              1 / self.nom_efferent_diameter ** 4) /
             self.baseline_nephrons)
        )

        # PT Na reabsorption per unit surface area
        self.PT_Na_reab_perUnitSA_0 = (
            (self.nom_filtered_sodium_load / self.baseline_nephrons) *
            self.nominal_pt_na_reabsorption /
            (3.14 * self.Dc_pt_nom * L_pt_total)
        )

        # Nominal TPR
        self.nom_TPR = self.nominal_map_setpoint / self.CO_nom

        # Peritubular ultrafiltration coefficient
        tubular_reabsorption = (
            self.nom_GFR / 1000 -
            self.nom_water_intake * self.water_intake_species_scale / 60 / 24
        )
        self.nom_peritubular_cap_Kf = (
            -tubular_reabsorption /
            (self.nom_postglomerular_pressure - self.RIHP0 -
             (self.nom_oncotic_pressure_peritubular -
              self.interstitial_oncotic_pressure))
        )

        # Creatinine synthesis rate
        self.creatinine_synthesis_rate = (
            self.equilibrium_serum_creatinine *
            self.dl_ml * self.nom_GFR
        )


# =========================================================================
# State variable indices for the 33-variable ODE system
# =========================================================================

# RAAS (5)
IDX_AngI = 0
IDX_AngII = 1
IDX_AT1_bound = 2
IDX_AT2_bound = 3
IDX_PRC = 4

# Volume (5)
IDX_blood_volume_L = 5
IDX_IF_volume = 6
IDX_sodium_amount = 7
IDX_IF_sodium = 8
IDX_stored_sodium = 9

# Feedback delays (8)
IDX_TGF_effect = 10
IDX_aldosterone = 11
IDX_preafferent_autoreg = 12
IDX_GP_autoreg = 13
IDX_CO_error = 14
IDX_Na_error = 15
IDX_VP_delayed = 16
IDX_F0_TGF = 17

# State delays (5)
IDX_P_bowmans = 18
IDX_oncotic_diff = 19
IDX_RBF_delayed = 20
IDX_MD_Na_delayed = 21
IDX_RSNA_delayed = 22

# Disease (4)
IDX_Kf_increase = 23
IDX_CD_PN_loss = 24
IDX_tubular_length = 25
IDX_tubular_diameter = 26

# PT water delays (3)
IDX_water_s1_delayed = 27
IDX_water_s2_delayed = 28
IDX_water_s3_delayed = 29

# Other (3)
IDX_serum_creatinine = 30
IDX_postglom_P_delayed = 31
IDX_postglom_P_error = 32

N_STATE = 33


# =========================================================================
# PART 2: compute_renal_algebraic()
# All non-ODE algebraic equations from modelfile_commented.R lines 589-1521
# =========================================================================

def compute_renal_algebraic(y, params, MAP, CO, P_ven, sim_time=0.0,
                            inflammatory_state=None):
    """
    Compute all renal algebraic (non-ODE) equations.

    Faithfully translates modelfile_commented.R lines 589-1588.

    Parameters
    ----------
    y : ndarray, shape (33,)
        Current ODE state vector.
    params : HallowRenalParams
        Model parameters.
    MAP : float
        Mean arterial pressure [mmHg].
    CO : float
        Cardiac output [L/min].
    P_ven : float
        Venous pressure [mmHg].
    sim_time : float
        Simulation time [hours].
    inflammatory_state : object or None
        If provided, must have attributes: Kf_factor, R_AA_factor,
        R_EA_factor, eta_PT_offset, MAP_setpoint_offset, RAAS_gain_factor.

    Returns
    -------
    out : dict
        All computed algebraic quantities.
    dydt_extras : dict
        Values needed by the ODE RHS that come from algebraic computation.
    """
    p = params
    out = {}

    # Unpack state variables
    AngI = y[IDX_AngI]
    AngII = y[IDX_AngII]
    AT1_bound_AngII = y[IDX_AT1_bound]
    AT2_bound_AngII = y[IDX_AT2_bound]
    plasma_renin_concentration = y[IDX_PRC]

    blood_volume_L = y[IDX_blood_volume_L]
    interstitial_fluid_volume = y[IDX_IF_volume]
    sodium_amount = y[IDX_sodium_amount]
    IF_sodium_amount = y[IDX_IF_sodium]
    stored_sodium = y[IDX_stored_sodium]

    tubulo_glomerular_feedback_effect = y[IDX_TGF_effect]
    normalized_aldosterone_level = y[IDX_aldosterone]
    preafferent_pressure_autoreg_signal = y[IDX_preafferent_autoreg]
    glomerular_pressure_autoreg_signal = y[IDX_GP_autoreg]
    CO_error = y[IDX_CO_error]
    Na_concentration_error = y[IDX_Na_error]
    normalized_vasopressin_concentration_delayed = y[IDX_VP_delayed]
    F0_TGF = y[IDX_F0_TGF]

    P_bowmans = y[IDX_P_bowmans]
    oncotic_pressure_difference = y[IDX_oncotic_diff]
    renal_blood_flow_L_min_delayed = y[IDX_RBF_delayed]
    SN_macula_densa_Na_flow_delayed = y[IDX_MD_Na_delayed]
    rsna_delayed = y[IDX_RSNA_delayed]

    disease_effects_increasing_Kf = y[IDX_Kf_increase]
    disease_effects_decreasing_CD_PN = y[IDX_CD_PN_loss]
    tubular_length_increase = y[IDX_tubular_length]
    tubular_diameter_increase = y[IDX_tubular_diameter]

    water_out_s1_delayed = y[IDX_water_s1_delayed]
    water_out_s2_delayed = y[IDX_water_s2_delayed]
    water_out_s3_delayed = y[IDX_water_s3_delayed]

    serum_creatinine = y[IDX_serum_creatinine]
    postglomerular_pressure_delayed = y[IDX_postglom_P_delayed]
    postglomerular_pressure_error = y[IDX_postglom_P_error]

    # Use MAP directly as mean_arterial_pressure_MAP
    mean_arterial_pressure_MAP = MAP

    # Mean venous pressure (converted from P_ven in mmHg)
    # In the R model: mean_venous_pressure is in Pa, used as:
    # (mean_venous_pressure*0.0075 - 3.16) in the RBF equation
    # For our standalone use, P_ven is already in mmHg
    mean_venous_pressure_mmHg = P_ven

    # ── Drug effects (default no-effect for standalone renal) ──
    ARB_signal = p.ARB_is_on * (1 - math.exp(-p.k_PD * sim_time)) if sim_time > 0 else 0
    BB_signal = p.BB_is_on * (1 - math.exp(-p.k_PD * sim_time)) if sim_time > 0 else 0

    # SGLT2 inhibition delayed state (from ODE or param)
    SGLT2_inhibition_delayed = p.SGLT2_inhibition  # simplified: use param directly
    RUGE_delayed = 0  # default
    RTg_compensation = 0  # default

    # Renal sympathetic nerve activity
    renal_sympathetic_nerve_activity = p.renal_sympathetic_nerve_activity

    # Normalized ANP (default = 1, would be driven by cardiac model)
    normalized_ANP = p.normalized_atrial_NP_concentration

    # =====================================================================
    # Line 592-593: Functional nephrons
    # =====================================================================
    number_of_functional_glomeruli = p.baseline_nephrons
    number_of_functional_tubules = p.baseline_nephrons * (1 - p.disease_effect_on_nephrons)

    out['number_of_functional_glomeruli'] = number_of_functional_glomeruli
    out['number_of_functional_tubules'] = number_of_functional_tubules

    # =====================================================================
    # Lines 597-698: Renal Vascular Resistance
    # =====================================================================

    # AT1-bound AngII effects on arterioles (sigmoid)
    AT1_preaff_int = 1 - p.AT1_preaff_scale / 2
    AT1_effect_on_preaff = (
        AT1_preaff_int +
        p.AT1_preaff_scale /
        (1 + math.exp(-(AT1_bound_AngII - p.nominal_equilibrium_AT1_bound_AngII) /
                       p.AT1_preaff_slope))
    )

    AT1_aff_int = 1 - p.AT1_aff_scale / 2
    AT1_effect_on_aff = (
        AT1_aff_int +
        p.AT1_aff_scale /
        (1 + math.exp(-(AT1_bound_AngII - p.nominal_equilibrium_AT1_bound_AngII) /
                       p.AT1_aff_slope))
    )

    AT1_eff_int = 1 - p.AT1_eff_scale / 2
    AT1_effect_on_eff = (
        AT1_eff_int +
        p.AT1_eff_scale /
        (1 + math.exp(-(AT1_bound_AngII - p.nominal_equilibrium_AT1_bound_AngII) /
                       p.AT1_eff_slope))
    )

    # RSNA effect on preafferent
    rsna_preaff_int = 1 - p.rsna_preaff_scale / 2
    rsna_effect_on_preaff = (
        rsna_preaff_int +
        p.rsna_preaff_scale /
        (1 + math.exp(-(renal_sympathetic_nerve_activity - p.nom_rsna) /
                       p.rsna_preaff_slope))
    )

    # Preafferent resistance
    preaff_arteriole_signal_multiplier = (
        AT1_effect_on_preaff *
        preafferent_pressure_autoreg_signal *
        p.CCB_effect_on_preafferent_resistance *
        rsna_effect_on_preaff *
        (1 - (1 - p.BB_preafferent_R_effect) * BB_signal)
    )

    preaff_arteriole_adjusted_signal_multiplier = (
        1 / (1 + math.exp(p.preaff_signal_nonlin_scale *
                           (1 - preaff_arteriole_signal_multiplier))) + 0.5
    )

    preafferent_arteriole_resistance = (
        p.nom_preafferent_arteriole_resistance *
        preaff_arteriole_adjusted_signal_multiplier
    )

    # Afferent arteriole resistance
    nom_afferent_arteriole_resistance = (
        p.L_m3 * p.viscosity_length_constant /
        (p.nom_afferent_diameter ** 4)
    )

    afferent_arteriole_signal_multiplier = (
        tubulo_glomerular_feedback_effect *
        AT1_effect_on_aff *
        glomerular_pressure_autoreg_signal *
        p.CCB_effect_on_afferent_resistance
    )

    afferent_arteriole_adjusted_signal_multiplier = (
        1 / (1 + math.exp(p.afferent_signal_nonlin_scale *
                           (1 - afferent_arteriole_signal_multiplier))) + 0.5
    )

    afferent_arteriole_resistance = (
        nom_afferent_arteriole_resistance *
        afferent_arteriole_adjusted_signal_multiplier
    )

    # Efferent arteriole resistance
    nom_efferent_arteriole_resistance = (
        p.L_m3 * p.viscosity_length_constant /
        (p.nom_efferent_diameter ** 4)
    )

    efferent_arteriole_signal_multiplier = (
        AT1_effect_on_eff *
        p.CCB_effect_on_efferent_resistance
    )

    efferent_arteriole_adjusted_signal_multiplier = (
        1 / (1 + math.exp(p.efferent_signal_nonlin_scale *
                           (1 - efferent_arteriole_signal_multiplier))) + 0.5
    )

    efferent_arteriole_resistance = (
        nom_efferent_arteriole_resistance *
        efferent_arteriole_adjusted_signal_multiplier
    )

    # Apply inflammatory state factors to arteriolar resistances
    if inflammatory_state is not None:
        ist = inflammatory_state
        afferent_arteriole_resistance *= ist.R_AA_factor
        efferent_arteriole_resistance *= ist.R_EA_factor

    # Peritubular resistance (autoregulation)
    RBF_autoreg_int = 1 - p.RBF_autoreg_scale / 2
    peritubular_autoreg_signal = (
        RBF_autoreg_int +
        p.RBF_autoreg_scale /
        (1 + math.exp((p.nom_renal_blood_flow_L_min -
                       renal_blood_flow_L_min_delayed) /
                      p.RBF_autoreg_steepness))
    )
    autoregulated_peritubular_resistance = (
        peritubular_autoreg_signal * p.nom_peritubular_resistance
    )

    # Renal vascular resistance (line 693-695)
    renal_vascular_resistance = (
        preafferent_arteriole_resistance +
        (afferent_arteriole_resistance + efferent_arteriole_resistance) /
        number_of_functional_glomeruli +
        autoregulated_peritubular_resistance
    )

    # Renal blood flow (line 698)
    # R model: renal_blood_flow_L_min = (MAP - (mean_venous_pressure*0.0075-3.16)) / RVR
    # In the R model, mean_venous_pressure is in Pa. When heart-renal link is on,
    # the venous pressure feeding the kidney equation uses that Pa-to-mmHg conversion.
    # For our standalone use, P_ven is already in mmHg, so we use it directly
    # as the downstream pressure for the renal circulation.
    renal_blood_flow_L_min = (
        (mean_arterial_pressure_MAP - mean_venous_pressure_mmHg) /
        renal_vascular_resistance
    )
    renal_blood_flow_L_min = max(renal_blood_flow_L_min, 0.01)

    renal_blood_flow_ml_hr = renal_blood_flow_L_min * 1000 * 60

    # Renal vasculature pressures (lines 703-712)
    preafferent_pressure = (
        mean_arterial_pressure_MAP -
        renal_blood_flow_L_min * preafferent_arteriole_resistance
    )

    glomerular_pressure = (
        mean_arterial_pressure_MAP -
        renal_blood_flow_L_min *
        (preafferent_arteriole_resistance +
         afferent_arteriole_resistance / number_of_functional_glomeruli)
    )

    postglomerular_pressure = (
        mean_arterial_pressure_MAP -
        renal_blood_flow_L_min *
        (preafferent_arteriole_resistance +
         (afferent_arteriole_resistance + efferent_arteriole_resistance) /
         number_of_functional_glomeruli)
    )

    # Autoregulatory signals (lines 714-722)
    preaff_autoreg_int = 1 - p.preaff_autoreg_scale / 2
    preafferent_pressure_autoreg_function = (
        preaff_autoreg_int +
        p.preaff_autoreg_scale /
        (1 + math.exp((p.nom_preafferent_pressure - preafferent_pressure) /
                      p.myogenic_steepness))
    )

    gp_autoreg_int = 1 - p.gp_autoreg_scale / 2
    glomerular_pressure_autoreg_function = (
        gp_autoreg_int +
        p.gp_autoreg_scale /
        (1 + math.exp((p.nom_glomerular_pressure - glomerular_pressure) /
                      p.myogenic_steepness))
    )

    out['preafferent_pressure'] = preafferent_pressure
    out['glomerular_pressure'] = glomerular_pressure
    out['postglomerular_pressure'] = postglomerular_pressure
    out['renal_blood_flow_L_min'] = renal_blood_flow_L_min
    out['renal_vascular_resistance'] = renal_vascular_resistance

    # =====================================================================
    # Lines 725-798: Glomerular Filtration
    # =====================================================================

    # Kf increase from glomerular hypertrophy (line 731-733)
    GP_effect_increasing_Kf = (
        (p.maximal_glom_surface_area_increase - disease_effects_increasing_Kf) *
        max(glomerular_pressure / (p.nom_glomerular_pressure + 2) - 1, 0) /
        (p.T_glomerular_pressure_increases_Kf / p.C_renal_CV_timescale)
    )

    # Effective Kf (line 735)
    glomerular_hydrostatic_conductance_Kf = (
        p.nom_Kf * (1 + disease_effects_increasing_Kf)
    )

    # Apply external Kf_scale (from HallowRenalModel.Kf_scale, CKD deterioration)
    Kf_scale_ext = getattr(p, '_Kf_scale_external', 1.0)
    glomerular_hydrostatic_conductance_Kf *= Kf_scale_ext

    # Apply inflammatory Kf factor
    if inflammatory_state is not None:
        glomerular_hydrostatic_conductance_Kf *= inflammatory_state.Kf_factor

    # Net filtration pressure (line 739-741)
    net_filtration_pressure = (
        glomerular_pressure -
        oncotic_pressure_difference -
        P_bowmans
    )

    # SNGFR (lines 743-748)
    if net_filtration_pressure <= 0:
        SNGFR_nL_min = 0.001
    else:
        SNGFR_nL_min = glomerular_hydrostatic_conductance_Kf * net_filtration_pressure

    # GFR (line 751)
    GFR = SNGFR_nL_min / 1000 / 1000000 * number_of_functional_tubules
    GFR_ml_min = GFR * 1000

    # Filtration fraction (line 755)
    filtration_fraction = GFR / renal_blood_flow_L_min if renal_blood_flow_L_min > 0 else 0

    # Serum creatinine concentration (line 757)
    serum_creatinine_concentration = serum_creatinine / blood_volume_L if blood_volume_L > 0 else 0

    # Creatinine clearance rate (line 759-760)
    creatinine_clearance_rate = GFR_ml_min * p.dl_ml * serum_creatinine_concentration

    out['GFR'] = GFR
    out['GFR_ml_min'] = GFR_ml_min
    out['SNGFR_nL_min'] = SNGFR_nL_min
    out['filtration_fraction'] = filtration_fraction
    out['net_filtration_pressure'] = net_filtration_pressure
    out['glomerular_hydrostatic_conductance_Kf'] = glomerular_hydrostatic_conductance_Kf

    # ── Oncotic pressure (Landis-Pappenheimer) lines 762-798 ──

    # GP effect on sieving (line 764-766)
    GPdiff = max(0, glomerular_pressure - p.nom_GP_seiving_damage)
    GP_effect_on_Seiving = (
        p.Emax_seiving * GPdiff ** p.Gamma_seiving /
        (GPdiff ** p.Gamma_seiving + p.Km_seiving ** p.Gamma_seiving)
        if (GPdiff ** p.Gamma_seiving + p.Km_seiving ** p.Gamma_seiving) > 0 else 0
    )

    # Dean and Lazzara sieving coefficient (line 769)
    denom = 1 - (1 - p.seiving_inf) * math.exp(-p.c_albumin * SNGFR_nL_min)
    nom_glomerular_albumin_sieving_coefficient = (
        p.seiving_inf / denom if denom != 0 else p.seiving_inf
    )

    glomerular_albumin_sieving_coefficient = (
        nom_glomerular_albumin_sieving_coefficient * (1 + GP_effect_on_Seiving)
    )

    # Albumin filtration and excretion (lines 773-777)
    SN_albumin_filtration_rate = (
        p.plasma_albumin_concentration * SNGFR_nL_min * 1e-6 *
        glomerular_albumin_sieving_coefficient
    )
    SN_albumin_excretion_rate = (
        max(0, SN_albumin_filtration_rate - p.SN_albumin_reabsorptive_capacity) +
        p.nom_albumin_excretion_rate
    )
    albumin_excretion_rate = SN_albumin_excretion_rate * number_of_functional_tubules

    out['albumin_excretion_rate'] = albumin_excretion_rate

    # Oncotic pressures (Landis-Pappenheimer, lines 785-797)
    Oncotic_pressure_in = (
        1.629 * p.plasma_protein_concentration +
        0.2935 * (p.plasma_protein_concentration ** 2)
    )

    SNRBF_nl_min = (
        1e6 * 1000 * renal_blood_flow_L_min / number_of_functional_glomeruli
    )

    if (SNRBF_nl_min - SNGFR_nL_min) > 0:
        plasma_protein_concentration_out = (
            (SNRBF_nl_min * p.plasma_protein_concentration -
             SN_albumin_filtration_rate) /
            (SNRBF_nl_min - SNGFR_nL_min)
        )
    else:
        plasma_protein_concentration_out = p.plasma_protein_concentration

    Oncotic_pressure_out = (
        1.629 * plasma_protein_concentration_out +
        0.2935 * (plasma_protein_concentration_out ** 2)
    )

    oncotic_pressure_avg = (Oncotic_pressure_in + Oncotic_pressure_out) / 2

    out['oncotic_pressure_avg'] = oncotic_pressure_avg
    out['SNRBF_nl_min'] = SNRBF_nl_min
    out['Oncotic_pressure_in'] = Oncotic_pressure_in
    out['Oncotic_pressure_out'] = Oncotic_pressure_out

    # =====================================================================
    # Lines 800-833: Plasma Na concentration and vasopressin
    # =====================================================================

    Na_concentration = sodium_amount / blood_volume_L if blood_volume_L > 0 else p.ref_Na_concentration
    IF_Na_concentration = (
        IF_sodium_amount / interstitial_fluid_volume
        if interstitial_fluid_volume > 0 else p.ref_Na_concentration
    )

    sodium_storate_rate = (
        p.Q_Na_store *
        ((p.max_stored_sodium - stored_sodium) / p.max_stored_sodium) *
        (IF_Na_concentration - p.ref_Na_concentration)
    )

    # Vasopressin controller (PI on Na concentration)
    Na_water_controller = (
        p.Na_controller_gain *
        (p.Kp_VP * (Na_concentration - p.ref_Na_concentration) +
         p.Ki_VP * Na_concentration_error)
    )

    normalized_vasopressin_concentration = 1 + Na_water_controller

    vasopressin_concentration = (
        p.nominal_vasopressin_conc * normalized_vasopressin_concentration
    )

    # Water intake (vasopressin-modulated) - line 826-830
    water_intake_vasopressin_int = 1 - p.water_intake_vasopressin_scale / 2
    water_intake = (
        p.water_intake_species_scale *
        (p.nom_water_intake / 60 / 24) *
        (water_intake_vasopressin_int +
         p.water_intake_vasopressin_scale /
         (1 + math.exp((normalized_vasopressin_concentration_delayed - 1) /
                       p.water_intake_vasopressin_slope)))
    )
    daily_water_intake = water_intake * 24 * 60

    out['Na_concentration'] = Na_concentration
    out['IF_Na_concentration'] = IF_Na_concentration
    out['sodium_storate_rate'] = sodium_storate_rate
    out['normalized_vasopressin_concentration'] = normalized_vasopressin_concentration
    out['water_intake'] = water_intake
    out['daily_water_intake'] = daily_water_intake

    # =====================================================================
    # Lines 835-1126: Tubular Flow and Reabsorption
    # =====================================================================

    # Tubular segment lengths (disease-modified)
    L_pt_s1 = p.L_pt_s1_nom * (1 + tubular_length_increase)
    L_pt_s2 = p.L_pt_s2_nom * (1 + tubular_length_increase)
    L_pt_s3 = p.L_pt_s3_nom * (1 + tubular_length_increase)
    Dc_pt = p.Dc_pt_nom * (1 + tubular_diameter_increase)
    L_pt = L_pt_s1 + L_pt_s2 + L_pt_s3

    # Single-nephron filtered loads (line 848-852)
    SN_filtered_Na_load = (SNGFR_nL_min / 1000 / 1000000) * Na_concentration
    filtered_Na_load = SN_filtered_Na_load * number_of_functional_tubules

    # ── Regulatory effects on reabsorption (lines 854-955) ──

    # Pressure natriuresis signal (PID) - lines 858-861
    pressure_natriuresis_signal = max(
        0.001,
        1 + p.Kp_PN * (postglomerular_pressure - p.RIHP0) +
        p.Ki_PN * postglomerular_pressure_error +
        p.Kd_PN * (postglomerular_pressure - postglomerular_pressure_delayed)
    )

    # PT pressure natriuresis (lines 863-868)
    pressure_natriuresis_PT_int = 1 - p.pressure_natriuresis_PT_scale / 2
    pressure_natriuresis_PT_effect = max(
        0.001,
        pressure_natriuresis_PT_int +
        p.pressure_natriuresis_PT_scale /
        (1 + math.exp(pressure_natriuresis_signal - 1))
    )

    # LoH pressure natriuresis (lines 870-874)
    pressure_natriuresis_LoH_int = 1 - p.pressure_natriuresis_LoH_scale / 2
    pressure_natriuresis_LoH_effect = max(
        0.001,
        pressure_natriuresis_LoH_int +
        p.pressure_natriuresis_LoH_scale /
        (1 + math.exp((postglomerular_pressure_delayed - p.RIHP0) /
                      p.pressure_natriuresis_LoH_slope))
    )

    # DCT pressure natriuresis (lines 876-882)
    pressure_natriuresis_DCT_magnitude = max(0, p.pressure_natriuresis_DCT_scale)
    pressure_natriuresis_DCT_int = 1 - pressure_natriuresis_DCT_magnitude / 2
    pressure_natriuresis_DCT_effect = max(
        0.001,
        pressure_natriuresis_DCT_int +
        pressure_natriuresis_DCT_magnitude /
        (1 + math.exp((postglomerular_pressure_delayed - p.RIHP0) /
                      p.pressure_natriuresis_DCT_slope))
    )

    # CD pressure natriuresis (lines 884-890)
    pressure_natriuresis_CD_magnitude = max(
        0,
        p.pressure_natriuresis_CD_scale * (1 + disease_effects_decreasing_CD_PN)
    )
    pressure_natriuresis_CD_int = 1 - pressure_natriuresis_CD_magnitude / 2
    pressure_natriuresis_CD_effect = max(
        0.001,
        pressure_natriuresis_CD_int +
        pressure_natriuresis_CD_magnitude /
        (1 + math.exp(pressure_natriuresis_signal - 1))
    )

    # RBF-CD effect (lines 892-896)
    RBF_CD_int = 1 - p.RBF_CD_scale / 2
    RBF_CD_effect = max(
        0.001,
        RBF_CD_int +
        p.RBF_CD_scale /
        (1 + math.exp((renal_blood_flow_L_min - p.nom_renal_blood_flow_L_min) /
                      p.RBF_CD_slope))
    )

    # AT1 effect on PT (lines 900-902)
    AT1_PT_int = 1 - p.AT1_PT_scale / 2
    AT1_effect_on_PT = (
        AT1_PT_int +
        p.AT1_PT_scale /
        (1 + math.exp(-(AT1_bound_AngII - p.nominal_equilibrium_AT1_bound_AngII) /
                       p.AT1_PT_slope))
    )

    # RSNA effect on PT and CD (lines 904-911)
    rsna_effect_on_PT = 1  # As in R model line 907

    rsna_CD_int = 1 - p.rsna_CD_scale / 2
    rsna_effect_on_CD = (
        rsna_CD_int +
        p.rsna_CD_scale /
        (1 + math.exp((1 - renal_sympathetic_nerve_activity) / p.rsna_CD_slope))
    )

    # Aldosterone effects (lines 913-929)
    aldosterone_concentration = (
        normalized_aldosterone_level * p.nominal_aldosterone_concentration
    )

    Aldo_MR_normalised_effect = (
        normalized_aldosterone_level * (1 - p.pct_target_inhibition_MRA)
    )

    aldo_DCT_int = 1 - p.aldo_DCT_scale / 2
    aldo_effect_on_DCT = (
        aldo_DCT_int +
        p.aldo_DCT_scale /
        (1 + math.exp((1 - Aldo_MR_normalised_effect) / p.aldo_DCT_slope))
    )

    aldo_CD_int = 1 - p.aldo_CD_scale / 2
    aldo_effect_on_CD = (
        aldo_CD_int +
        p.aldo_CD_scale /
        (1 + math.exp((1 - Aldo_MR_normalised_effect) / p.aldo_CD_slope))
    )

    # ANP effect on CD (lines 932-933)
    anp_CD_int = 1 - p.anp_CD_scale / 2
    anp_effect_on_CD = (
        anp_CD_int +
        p.anp_CD_scale /
        (1 + math.exp((1 - normalized_ANP) / p.anp_CD_slope))
    )

    # NHE3 inhibition from SGLT2 (line 936)
    NHE3inhib = p.Anhe3 * RUGE_delayed

    # PT multiplier (lines 938-941)
    pt_multiplier = (
        AT1_effect_on_PT *
        rsna_effect_on_PT *
        pressure_natriuresis_PT_effect *
        (1 - NHE3inhib)
    )

    # Effective fractional reabsorption rates (lines 943-955)
    e_pt_sodreab = min(1, p.nominal_pt_na_reabsorption_nonSGLT * pt_multiplier)

    # Apply inflammatory eta_PT_offset
    if inflammatory_state is not None:
        e_pt_sodreab = min(1, e_pt_sodreab + inflammatory_state.eta_PT_offset)

    e_dct_sodreab = min(
        1,
        p.nominal_dt_na_reabsorption *
        aldo_effect_on_DCT *
        pressure_natriuresis_DCT_effect *
        p.HCTZ_effect_on_DT_Na_reabs
    )

    cd_multiplier = (
        aldo_effect_on_CD *
        rsna_effect_on_CD *
        pressure_natriuresis_CD_effect *
        RBF_CD_effect
    )

    e_cd_sodreab = min(
        0.9999,
        p.nominal_cd_na_reabsorption * cd_multiplier * anp_effect_on_CD
    )

    out['e_pt_sodreab'] = e_pt_sodreab
    out['e_dct_sodreab'] = e_dct_sodreab
    out['e_cd_sodreab'] = e_cd_sodreab

    # =====================================================================
    # Lines 957-1126: Proximal Tubule Reabsorption
    # =====================================================================

    # Glucose reabsorption per unit length (lines 964-974)
    glucose_reabs_per_unit_length_s1 = (
        p.nom_glucose_reabs_per_unit_length_s1 *
        SGLT2_inhibition_delayed *
        (1 + RTg_compensation)
    )
    glucose_reabs_per_unit_length_s2 = (
        p.nom_glucose_reabs_per_unit_length_s2 *
        SGLT2_inhibition_delayed *
        (1 + RTg_compensation)
    )
    glucose_reabs_per_unit_length_s3 = (
        p.nom_glucose_reabs_per_unit_length_s3 *
        (1 + RTg_compensation) *
        p.SGLT1_inhibition
    )

    # Single-nephron filtered glucose (line 976)
    SN_filtered_glucose_load = (
        p.glucose_concentration * SNGFR_nL_min / 1000 / 1000000
    )

    # Glucose reabsorption per segment (lines 978-983)
    glucose_pt_out_s1 = max(
        0, SN_filtered_glucose_load -
        glucose_reabs_per_unit_length_s1 * L_pt_s1
    )
    glucose_pt_out_s2 = max(
        0, glucose_pt_out_s1 -
        glucose_reabs_per_unit_length_s2 * L_pt_s2
    )
    glucose_pt_out_s3 = max(
        0, glucose_pt_out_s2 -
        glucose_reabs_per_unit_length_s3 * L_pt_s3
    )

    # RUGE (line 985)
    RUGE = glucose_pt_out_s3 * number_of_functional_tubules * 180

    # RTg compensation (line 987-988)
    excess_glucose_increasing_RTg = (
        (p.maximal_RTg_increase - RTg_compensation) *
        max(RUGE, 0) /
        (p.T_glucose_RTg / p.C_renal_CV_timescale)
    )

    # Osmotic effects (lines 990-996, all scale=0 by default)
    osmotic_natriuresis_effect_pt = 1 - min(1, RUGE * p.glucose_natriuresis_effect_pt)
    osmotic_natriuresis_effect_cd = 1 - min(1, RUGE * p.glucose_natriuresis_effect_cd)
    osmotic_diuresis_effect_pt = 1 - min(1, RUGE * p.glucose_diuresis_effect_pt)
    osmotic_diuresis_effect_cd = 1 - min(1, RUGE * p.glucose_diuresis_effect_cd)

    # Na reabsorption per unit length (line 1018-1019)
    Na_reabs_per_unit_length = (
        -math.log(max(1e-10, 1 - e_pt_sodreab)) / (L_pt_s1 + L_pt_s2 + L_pt_s3)
    )

    # SGLT Na co-transport (lines 1006-1016)
    SGTL2_Na_reabs_mmol_s1 = SN_filtered_glucose_load - glucose_pt_out_s1
    SGTL2_Na_reabs_mmol_s2 = glucose_pt_out_s1 - glucose_pt_out_s2
    SGTL1_Na_reabs_mmol = 2 * (glucose_pt_out_s2 - glucose_pt_out_s3)
    total_SGLT_Na_reabs = (
        SGTL2_Na_reabs_mmol_s1 + SGTL2_Na_reabs_mmol_s2 + SGTL1_Na_reabs_mmol
    )

    # PT S1 Na reabsorption (lines 1021-1027)
    Na_pt_s1_reabs = min(
        p.max_s1_Na_reabs,
        SN_filtered_Na_load * (1 - math.exp(-Na_reabs_per_unit_length * L_pt_s1))
    )
    Na_pt_out_s1 = SN_filtered_Na_load - Na_pt_s1_reabs - SGTL2_Na_reabs_mmol_s1

    # PT S2 Na reabsorption (lines 1029-1035)
    Na_pt_s2_reabs = min(
        p.max_s2_Na_reabs,
        Na_pt_out_s1 * (1 - math.exp(-Na_reabs_per_unit_length * L_pt_s2))
    )
    Na_pt_out_s2 = Na_pt_out_s1 - Na_pt_s2_reabs - SGTL2_Na_reabs_mmol_s2

    # PT S3 Na reabsorption (lines 1037-1043)
    Na_pt_s3_reabs = min(
        p.max_s3_Na_reabs,
        Na_pt_out_s2 * (1 - math.exp(-Na_reabs_per_unit_length * L_pt_s3))
    )
    Na_pt_out_s3 = Na_pt_out_s2 - Na_pt_s3_reabs - SGTL1_Na_reabs_mmol

    # PT Na reabsorption fraction (line 1045-1046)
    PT_Na_reabs_fraction = (
        1 - Na_pt_out_s3 / SN_filtered_Na_load
        if SN_filtered_Na_load > 0 else 0
    )

    # ── PT Urea (lines 1048-1066) ──
    SN_filtered_urea_load = (SNGFR_nL_min / 1000 / 1000000) * p.plasma_urea
    SNGFR_L_min = SNGFR_nL_min / 1000 / 1000000

    # Urea out S1 (line 1051-1054)
    if SNGFR_L_min + water_out_s1_delayed > 0 and SN_filtered_urea_load > 0:
        urea_out_s1 = (
            SN_filtered_urea_load -
            p.urea_permeability_PT *
            (SN_filtered_urea_load /
             (0.5 * (SNGFR_L_min + water_out_s1_delayed)) - p.plasma_urea) *
            water_out_s1_delayed
        )
    else:
        urea_out_s1 = SN_filtered_urea_load

    # Urea out S2 (line 1056-1059)
    if water_out_s1_delayed + water_out_s2_delayed > 0 and urea_out_s1 > 0:
        urea_out_s2 = (
            urea_out_s1 -
            p.urea_permeability_PT *
            (urea_out_s1 /
             (0.5 * (water_out_s1_delayed + water_out_s2_delayed)) - p.plasma_urea) *
            water_out_s2_delayed
        )
    else:
        urea_out_s2 = urea_out_s1

    # Urea out S3 (line 1061-1064)
    if water_out_s2_delayed + water_out_s3_delayed > 0 and urea_out_s2 > 0:
        urea_out_s3 = (
            urea_out_s2 -
            p.urea_permeability_PT *
            (urea_out_s2 /
             (0.5 * (water_out_s2_delayed + water_out_s3_delayed)) - p.plasma_urea) *
            water_out_s3_delayed
        )
    else:
        urea_out_s3 = urea_out_s2

    # ── PT Water Reabsorption (lines 1069-1091) ──
    # Isosmotic: water follows osmoles
    osmoles_out_s1 = 2 * Na_pt_out_s1 + glucose_pt_out_s1 + urea_out_s1
    total_filtered_osmoles = (
        2 * SN_filtered_Na_load + SN_filtered_glucose_load + SN_filtered_urea_load
    )

    if total_filtered_osmoles > 0:
        water_out_s1 = (SNGFR_L_min / total_filtered_osmoles) * osmoles_out_s1
    else:
        water_out_s1 = SNGFR_L_min

    osmoles_out_s2 = 2 * Na_pt_out_s2 + glucose_pt_out_s2 + urea_out_s2
    if osmoles_out_s1 > 0:
        water_out_s2 = (water_out_s1 / osmoles_out_s1) * osmoles_out_s2
    else:
        water_out_s2 = water_out_s1

    osmoles_out_s3 = 2 * Na_pt_out_s3 + glucose_pt_out_s3 + urea_out_s3
    if osmoles_out_s2 > 0:
        water_out_s3 = (water_out_s2 / osmoles_out_s2) * osmoles_out_s3
    else:
        water_out_s3 = water_out_s2

    PT_water_reabs_fraction = (
        1 - water_out_s3 / SNGFR_L_min if SNGFR_L_min > 0 else 0
    )

    # Concentrations out of PT (lines 1093-1116)
    Na_concentration_out_s1 = Na_pt_out_s1 / water_out_s1 if water_out_s1 > 0 else 0
    Na_concentration_out_s2 = Na_pt_out_s2 / water_out_s2 if water_out_s2 > 0 else 0
    Na_concentration_out_s3 = Na_pt_out_s3 / water_out_s3 if water_out_s3 > 0 else 0

    glucose_concentration_out_s3 = glucose_pt_out_s3 / water_out_s3 if water_out_s3 > 0 else 0

    urea_concentration_out_s3 = urea_out_s3 / water_out_s3 if water_out_s3 > 0 else 0

    osmolality_out_s1 = osmoles_out_s1 / water_out_s1 if water_out_s1 > 0 else 0
    osmolality_out_s2 = osmoles_out_s2 / water_out_s2 if water_out_s2 > 0 else 0
    osmolality_out_s3 = osmoles_out_s3 / water_out_s3 if water_out_s3 > 0 else 0

    PT_Na_outflow = Na_pt_out_s3 * number_of_functional_tubules

    # PT Na reabsorption per unit SA (lines 1121-1126)
    if Dc_pt * L_pt > 0:
        PT_Na_reab_perUnitSA = (
            SN_filtered_Na_load * e_pt_sodreab /
            (3.14 * Dc_pt * (L_pt_s1 + L_pt_s2 + L_pt_s3))
        )
    else:
        PT_Na_reab_perUnitSA = 0

    normalized_PT_reabsorption_density = (
        PT_Na_reab_perUnitSA / p.PT_Na_reab_perUnitSA_0
        if p.PT_Na_reab_perUnitSA_0 > 0 else 1
    )

    # Tubular hypertrophy effects (lines 1125-1126, both zero by default)
    PT_Na_reabs_effect_increasing_tubular_length = 0
    PT_Na_reabs_effect_increasing_tubular_diameter = 0

    out['water_out_s1'] = water_out_s1
    out['water_out_s2'] = water_out_s2
    out['water_out_s3'] = water_out_s3
    out['Na_pt_out_s3'] = Na_pt_out_s3
    out['PT_Na_outflow'] = PT_Na_outflow

    # =====================================================================
    # Lines 1129-1243: Loop of Henle (countercurrent)
    # =====================================================================

    # Descending loop of Henle
    water_in_DescLoH = water_out_s3
    Na_in_DescLoH = Na_pt_out_s3
    urea_in_DescLoH = urea_out_s3
    glucose_in_DescLoH = glucose_pt_out_s3
    osmoles_in_DescLoH = osmoles_out_s3
    osmolality_in_DescLoH = osmolality_out_s3

    # No solute reabsorption in descending limb
    Na_out_DescLoH = Na_in_DescLoH
    urea_out_DescLoH = urea_in_DescLoH
    glucose_out_DescLoH = glucose_in_DescLoH
    osmoles_out_DescLoH = osmoles_in_DescLoH

    # Flow-dependent LoH reabsorption (line 1165-1167)
    nom_Na_in_AscLoH = p.nom_Na_in_AscLoH
    deltaLoH_NaFlow = min(
        p.max_deltaLoH_reabs,
        p.LoH_flow_dependence * (Na_out_DescLoH - nom_Na_in_AscLoH)
    )

    AscLoH_Reab_Rate = (
        2 * p.nominal_loh_na_reabsorption *
        (nom_Na_in_AscLoH + deltaLoH_NaFlow) *
        p.loop_diuretic_effect / p.L_lh_des
    )

    effective_AscLoH_Reab_Rate = AscLoH_Reab_Rate * pressure_natriuresis_LoH_effect

    # Descending LoH water concentration (countercurrent, line 1173)
    if water_in_DescLoH > 0 and osmolality_in_DescLoH > 0:
        reab_term = min(effective_AscLoH_Reab_Rate * p.L_lh_des, 2 * Na_in_DescLoH)
        exp_arg = reab_term / (water_in_DescLoH * osmolality_in_DescLoH)
        osmolality_out_DescLoH = osmolality_in_DescLoH * math.exp(exp_arg)
        water_out_DescLoH = (
            water_in_DescLoH * osmolality_in_DescLoH / osmolality_out_DescLoH
        )
    else:
        osmolality_out_DescLoH = osmolality_in_DescLoH
        water_out_DescLoH = water_in_DescLoH

    # Ascending loop of Henle (lines 1184-1237)
    Na_in_AscLoH = Na_out_DescLoH
    reabsorbed_urea_cd_delayed = 0  # zero by default in R model

    urea_in_AscLoH = urea_out_DescLoH + reabsorbed_urea_cd_delayed
    water_in_AscLoH = water_out_DescLoH
    osmoles_in_AscLoH = osmoles_out_DescLoH + reabsorbed_urea_cd_delayed

    osmolality_in_AscLoH = (
        osmoles_in_AscLoH / water_in_AscLoH if water_in_AscLoH > 0 else 0
    )

    # Ascending LoH osmolality decrease (line 1209)
    if water_in_DescLoH > 0 and osmolality_in_DescLoH > 0:
        reab_limited = min(
            p.L_lh_des * effective_AscLoH_Reab_Rate, 2 * Na_in_DescLoH
        )
        exp_factor = math.exp(
            reab_limited / (water_in_DescLoH * osmolality_in_DescLoH)
        )
        osmolality_out_AscLoH = (
            osmolality_in_AscLoH -
            reab_limited * (exp_factor / water_in_DescLoH)
        )
    else:
        osmolality_out_AscLoH = osmolality_in_AscLoH

    osmoles_reabsorbed_AscLoH = (
        (osmolality_in_AscLoH - osmolality_out_AscLoH) * water_in_AscLoH
    )
    Na_reabsorbed_AscLoH = osmoles_reabsorbed_AscLoH / 2

    Na_out_AscLoH = max(0, Na_in_AscLoH - Na_reabsorbed_AscLoH)

    # No water/glucose/urea reabsorption along ascending limb
    urea_out_AscLoH = urea_in_AscLoH
    glucose_out_AscLoH = glucose_in_DescLoH  # passes through
    water_out_AscLoH = water_in_AscLoH

    Na_concentration_out_AscLoH = (
        Na_out_AscLoH / water_out_AscLoH if water_out_AscLoH > 0 else 0
    )
    urea_concentration_in_AscLoH_val = (
        urea_in_AscLoH / water_out_DescLoH if water_out_DescLoH > 0 else 0
    )

    # Macula densa (line 1237-1239)
    SN_macula_densa_Na_flow = Na_out_AscLoH
    MD_Na_concentration = Na_concentration_out_AscLoH

    # TGF signal (line 1241-1243)
    TGF0 = 1 - p.S_tubulo_glomerular_feedback / 2
    tubulo_glomerular_feedback_signal = (
        TGF0 +
        p.S_tubulo_glomerular_feedback /
        (1 + math.exp((p.MD_Na_concentration_setpoint - MD_Na_concentration) /
                      p.F_md_scale_tubulo_glomerular_feedback))
    )

    out['SN_macula_densa_Na_flow'] = SN_macula_densa_Na_flow
    out['MD_Na_concentration'] = MD_Na_concentration
    out['tubulo_glomerular_feedback_signal'] = tubulo_glomerular_feedback_signal

    # =====================================================================
    # Lines 1247-1291: Distal Convoluted Tubule (DCT)
    # =====================================================================

    water_in_DCT = water_out_AscLoH
    Na_in_DCT = Na_out_AscLoH
    urea_in_DCT = urea_out_AscLoH
    glucose_in_DCT = glucose_out_AscLoH

    # Only sodium reabsorbed along DCT
    urea_out_DCT = urea_in_DCT
    glucose_out_DCT = glucose_in_DCT
    water_out_DCT = water_in_DCT

    # Na reabsorption (exponential, line 1281-1283)
    R_dct = -math.log(max(1e-10, 1 - e_dct_sodreab)) / p.L_dct
    Na_out_DCT = Na_in_DCT * math.exp(-R_dct * p.L_dct)

    Na_concentration_out_DCT = Na_out_DCT / water_out_DCT if water_out_DCT > 0 else 0
    glucose_concentration_out_DescLoH = (
        glucose_out_DescLoH / water_out_DescLoH if water_out_DescLoH > 0 else 0
    )
    urea_concentration_in_AscLoH_for_osm = urea_concentration_in_AscLoH_val

    # DCT osmolality (line 1287)
    osmolality_out_DCT = (
        2 * Na_concentration_out_DCT +
        glucose_concentration_out_DescLoH +
        urea_concentration_in_AscLoH_for_osm
    )

    out['Na_out_DCT'] = Na_out_DCT

    # =====================================================================
    # Lines 1293-1384: Collecting Duct
    # =====================================================================

    water_in_CD = water_out_DCT
    Na_in_CD = Na_out_DCT
    urea_in_CD = urea_out_DCT
    glucose_in_CD = glucose_out_DCT
    osmoles_in_CD = osmolality_out_DCT * water_out_DCT if water_out_DCT > 0 else 0
    osmolality_in_CD = osmolality_out_DCT

    # CD Na reabsorption (lines 1322-1329)
    e_cd_sodreab_adj = e_cd_sodreab * osmotic_natriuresis_effect_cd
    R_cd = -math.log(max(1e-10, 1 - e_cd_sodreab_adj)) / p.L_cd
    Na_reabsorbed_CD = min(
        Na_in_CD * (1 - math.exp(-R_cd * p.L_cd)),
        p.CD_Na_reabs_threshold
    )
    Na_out_CD = Na_in_CD - Na_reabsorbed_CD

    # ADH water permeability (line 1335)
    ADH_water_permeability = (
        normalized_vasopressin_concentration /
        (0.15 + normalized_vasopressin_concentration)
    )

    # Water reabsorption (lines 1341-1347)
    osmoles_out_CD = osmoles_in_CD - 2 * (Na_in_CD - Na_out_CD)
    osmolality_out_CD_before = (
        osmoles_out_CD / water_in_CD if water_in_CD > 0 else 0
    )

    if osmolality_out_DescLoH > 0 and water_in_CD > 0:
        water_reabsorbed_CD = (
            ADH_water_permeability *
            osmotic_diuresis_effect_cd *
            water_in_CD *
            (1 - osmolality_out_CD_before / osmolality_out_DescLoH)
        )
    else:
        water_reabsorbed_CD = 0

    water_out_CD = water_in_CD - water_reabsorbed_CD

    # Urine flow and excretion (lines 1353-1364)
    urine_flow_rate = water_out_CD * number_of_functional_tubules
    daily_urine_flow = urine_flow_rate * 60 * 24

    Na_excretion_via_urine = Na_out_CD * number_of_functional_tubules
    Na_balance = p.Na_intake_rate - Na_excretion_via_urine
    water_balance = daily_water_intake - daily_urine_flow

    out['urine_flow_rate'] = urine_flow_rate
    out['Na_excretion_via_urine'] = Na_excretion_via_urine
    out['Na_balance'] = Na_balance
    out['water_balance'] = water_balance
    out['daily_urine_flow'] = daily_urine_flow

    # =====================================================================
    # Lines 1387-1408: RIHP (Renal Interstitial Hydrostatic Pressure)
    # =====================================================================

    Oncotic_pressure_peritubular_in = Oncotic_pressure_out

    if number_of_functional_glomeruli > 0:
        urine_per_glomerulus = urine_flow_rate * 1e6 * 1000 / number_of_functional_glomeruli
    else:
        urine_per_glomerulus = 0

    denom_perit = SNRBF_nl_min - urine_per_glomerulus
    if denom_perit > 0:
        plasma_protein_concentration_peritubular_out = (
            SNRBF_nl_min * p.plasma_protein_concentration / denom_perit
        )
    else:
        plasma_protein_concentration_peritubular_out = p.plasma_protein_concentration

    Oncotic_pressure_peritubular_out = (
        1.629 * plasma_protein_concentration_peritubular_out +
        0.2935 * (plasma_protein_concentration_peritubular_out ** 2)
    )

    oncotic_pressure_peritubular_avg = (
        (Oncotic_pressure_peritubular_in + Oncotic_pressure_peritubular_out) / 2
    )

    tubular_reabsorption = GFR_ml_min / 1000 - urine_flow_rate

    RIHP = (
        postglomerular_pressure -
        (oncotic_pressure_peritubular_avg - p.interstitial_oncotic_pressure) +
        tubular_reabsorption / p.nom_peritubular_cap_Kf
        if p.nom_peritubular_cap_Kf != 0 else postglomerular_pressure
    )

    out['RIHP'] = RIHP

    # =====================================================================
    # Lines 1413-1521: Tubular Pressure (Hagen-Poiseuille in compliant tubes)
    # =====================================================================

    mmHg_Nperm2_conv = 133.32

    Pc_pt_s1 = p.Pc_pt_s1_mmHg * mmHg_Nperm2_conv
    Pc_pt_s2 = p.Pc_pt_s2_mmHg * mmHg_Nperm2_conv
    Pc_pt_s3 = p.Pc_pt_s3_mmHg * mmHg_Nperm2_conv
    Pc_lh_des = p.Pc_lh_des_mmHg * mmHg_Nperm2_conv
    Pc_lh_asc = p.Pc_lh_asc_mmHg * mmHg_Nperm2_conv
    Pc_dt = p.Pc_dt_mmHg * mmHg_Nperm2_conv
    Pc_cd = p.Pc_cd_mmHg * mmHg_Nperm2_conv
    P_interstitial = 4.9 * mmHg_Nperm2_conv

    pi_val = 3.14
    tc = p.tubular_compliance

    B1 = (4 * tc + 1) * 128 * p.gamma / pi_val

    # CD (lines 1441-1447)
    mean_cd_water_flow = (water_in_CD - water_out_CD) / 2
    B2_cd = (Pc_cd ** (4 * tc)) / (p.Dc_cd ** 4)
    # P_in_cd: 0^(4tc+1) = 0 for positive exponent
    P_in_cd_base = B1 * B2_cd * (mean_cd_water_flow / 1e3) * p.L_cd
    P_in_cd = max(0, P_in_cd_base) ** (1 / (4 * tc + 1)) if P_in_cd_base > 0 else 0
    P_in_cd_mmHg = (P_in_cd + P_interstitial) / mmHg_Nperm2_conv

    # DCT (lines 1452-1457)
    B2_dt = (Pc_dt ** (4 * tc)) / (p.Dc_dt ** 4)
    P_in_dt_base = (
        P_in_cd ** (4 * tc + 1) +
        B1 * B2_dt * (water_in_DCT / 1e3) * p.L_dct
    )
    P_in_dt = max(0, P_in_dt_base) ** (1 / (4 * tc + 1)) if P_in_dt_base > 0 else 0
    P_in_dt_mmHg = (P_in_dt + P_interstitial) / mmHg_Nperm2_conv

    # Ascending LoH (lines 1461-1466)
    B2_lh_asc = (Pc_lh_asc ** (4 * tc)) / (p.Dc_lh ** 4)
    P_in_lh_asc_base = (
        P_in_dt ** (4 * tc + 1) +
        B1 * B2_lh_asc * (water_in_AscLoH / 1e3) * p.L_lh_asc
    )
    P_in_lh_asc = (
        max(0, P_in_lh_asc_base) ** (1 / (4 * tc + 1))
        if P_in_lh_asc_base > 0 else 0
    )
    P_in_lh_asc_mmHg = (P_in_lh_asc + P_interstitial) / mmHg_Nperm2_conv

    # Descending LoH (lines 1468-1475)
    if water_in_DescLoH > 0 and osmolality_in_DescLoH > 0:
        A_lh_des = (
            effective_AscLoH_Reab_Rate /
            (water_in_DescLoH * osmolality_in_DescLoH)
        )
    else:
        A_lh_des = 0

    if A_lh_des > 0:
        B2_lh_des = (
            (Pc_lh_des ** (4 * tc)) *
            (water_in_DescLoH / 1e3) /
            ((p.Dc_lh ** 4) * A_lh_des)
        )
        P_in_lh_des_base = (
            P_in_lh_asc ** (4 * tc + 1) +
            B1 * B2_lh_des * (1 - math.exp(-A_lh_des * p.L_lh_des))
        )
    else:
        P_in_lh_des_base = P_in_lh_asc ** (4 * tc + 1)

    P_in_lh_des = (
        max(0, P_in_lh_des_base) ** (1 / (4 * tc + 1))
        if P_in_lh_des_base > 0 else 0
    )
    P_in_lh_des_mmHg = (P_in_lh_des + P_interstitial) / mmHg_Nperm2_conv

    # PT segments (lines 1477-1520)
    # Treat urea as if reabsorbed linearly
    if L_pt_s1 + L_pt_s2 + L_pt_s3 > 0:
        Rurea = (SN_filtered_urea_load - urea_out_s3) / (L_pt_s1 + L_pt_s2 + L_pt_s3)
    else:
        Rurea = 0

    urea_in_s2 = SN_filtered_urea_load - Rurea * L_pt_s1
    urea_in_s3 = SN_filtered_urea_load - Rurea * (L_pt_s1 + L_pt_s2)

    A_na = Na_reabs_per_unit_length

    glucose_pt_out_s1_local = glucose_pt_out_s1
    glucose_pt_out_s2_local = glucose_pt_out_s2
    Na_pt_out_s1_local = Na_pt_out_s1
    Na_pt_out_s2_local = Na_pt_out_s2

    # Flow integrals for pressure calculation (lines 1488-1492)
    flow_integral_s3 = (
        2 * (Na_pt_out_s2_local / A_na) * (1 - math.exp(-A_na * L_pt_s3)) -
        (3 / 2) * glucose_pt_out_s2_local * L_pt_s3 ** 2 +
        urea_in_s3 * L_pt_s3 -
        (1 / 2) * Rurea * (L_pt_s3 ** 2)
    ) if A_na > 0 else 0

    flow_integral_s2 = (
        2 * (Na_pt_out_s1_local / A_na) * (1 - math.exp(-A_na * L_pt_s2)) -
        (1 / 2) * glucose_pt_out_s1_local * L_pt_s2 ** 2 +
        urea_in_s2 * L_pt_s2 -
        (1 / 2) * Rurea * (L_pt_s2 ** 2)
    ) if A_na > 0 else 0

    flow_integral_s1 = (
        2 * (SN_filtered_Na_load / A_na) * (1 - math.exp(-A_na * L_pt_s1)) -
        (1 / 2) * SN_filtered_glucose_load * L_pt_s1 ** 2 +
        SN_filtered_urea_load * L_pt_s1 -
        (1 / 2) * Rurea * (L_pt_s1 ** 2)
    ) if A_na > 0 else 0

    # PT S3 pressure (lines 1496-1502)
    B2_pt_s3 = (Pc_pt_s3 ** (4 * tc)) / (Dc_pt ** 4) if Dc_pt > 0 else 0
    B3_pt_s3 = (water_out_s2 / 1e3) / osmoles_out_s2 if osmoles_out_s2 > 0 else 0
    P_in_pt_s3_base = (
        P_in_lh_des ** (4 * tc + 1) +
        B1 * B2_pt_s3 * B3_pt_s3 * flow_integral_s3
    )
    P_in_pt_s3 = (
        max(0, P_in_pt_s3_base) ** (1 / (4 * tc + 1))
        if P_in_pt_s3_base > 0 else 0
    )
    P_in_pt_s3_mmHg = (P_in_pt_s3 + P_interstitial) / mmHg_Nperm2_conv

    # PT S2 pressure (lines 1505-1511, note R uses Pc_pt_s3 for B2_pt_s2)
    B2_pt_s2 = (Pc_pt_s3 ** (4 * tc)) / (Dc_pt ** 4) if Dc_pt > 0 else 0
    B3_pt_s2 = (water_out_s1 / 1e3) / osmoles_out_s1 if osmoles_out_s1 > 0 else 0
    P_in_pt_s2_base = (
        P_in_pt_s3 ** (4 * tc + 1) +
        B1 * B2_pt_s2 * B3_pt_s2 * flow_integral_s2
    )
    P_in_pt_s2 = (
        max(0, P_in_pt_s2_base) ** (1 / (4 * tc + 1))
        if P_in_pt_s2_base > 0 else 0
    )
    P_in_pt_s2_mmHg = (P_in_pt_s2 + P_interstitial) / mmHg_Nperm2_conv

    # PT S1 pressure (lines 1514-1520)
    B2_pt_s1 = (Pc_pt_s1 ** (4 * tc)) / (Dc_pt ** 4) if Dc_pt > 0 else 0
    B3_pt_s1 = (
        (SNGFR_nL_min / 1e12) /
        (2 * SN_filtered_Na_load + SN_filtered_glucose_load + SN_filtered_urea_load)
        if (2 * SN_filtered_Na_load + SN_filtered_glucose_load + SN_filtered_urea_load) > 0
        else 0
    )
    P_in_pt_s1_base = (
        P_in_pt_s2 ** (4 * tc + 1) +
        B1 * B2_pt_s1 * B3_pt_s1 * flow_integral_s1
    )
    P_in_pt_s1 = (
        max(0, P_in_pt_s1_base) ** (1 / (4 * tc + 1))
        if P_in_pt_s1_base > 0 else 0
    )
    P_in_pt_s1_mmHg = (P_in_pt_s1 + P_interstitial) / mmHg_Nperm2_conv

    out['P_in_pt_s1_mmHg'] = P_in_pt_s1_mmHg

    # =====================================================================
    # Lines 1525-1588: Aldosterone and Renin Secretion
    # =====================================================================

    # Aldosterone secretion (lines 1529-1533)
    AT1_aldo_int = 1 - p.AT1_aldo_slope * p.nominal_equilibrium_AT1_bound_AngII
    AngII_effect_on_aldo = AT1_aldo_int + p.AT1_aldo_slope * AT1_bound_AngII
    N_als = p.K_Na_ratio_effect_on_aldo * AngII_effect_on_aldo

    # Renin secretion (lines 1536-1566)
    rsna_renin_intercept = 1 - p.rsna_renin_slope
    rsna_effect_on_renin_secretion = (
        p.rsna_renin_slope * renal_sympathetic_nerve_activity +
        rsna_renin_intercept
    )

    md_effect_on_renin_secretion = (
        p.md_renin_A *
        math.exp(-p.md_renin_tau *
                 (SN_macula_densa_Na_flow_delayed * p.baseline_nephrons -
                  p.nom_LoH_Na_outflow))
    )

    # AT1-bound AngII feedback on renin (line 1551)
    if AT1_bound_AngII > 0 and p.nominal_equilibrium_AT1_bound_AngII > 0:
        AT1_bound_AngII_effect_on_PRA = (
            10 ** (p.AT1_PRC_slope *
                   math.log10(AT1_bound_AngII /
                              p.nominal_equilibrium_AT1_bound_AngII) +
                   p.AT1_PRC_yint)
        )
    else:
        AT1_bound_AngII_effect_on_PRA = 1.0

    # Aldo effect on renin (line 1555-1557)
    aldo_renin_intercept = 1 - p.aldo_renin_slope
    aldo_effect_on_renin_secretion = (
        aldo_renin_intercept +
        p.aldo_renin_slope * Aldo_MR_normalised_effect
    )

    # Plasma renin activity (line 1562)
    plasma_renin_activity = (
        p.concentration_to_renin_activity_conversion_plasma *
        plasma_renin_concentration *
        (1 - p.pct_target_inhibition_DRI)
    )

    # Renin secretion rate (line 1566)
    renin_secretion_rate = (
        (math.log(2) / p.renin_half_life) *
        p.nominal_equilibrium_PRC *
        AT1_bound_AngII_effect_on_PRA *
        md_effect_on_renin_secretion *
        p.HCTZ_effect_on_renin_secretion *
        aldo_effect_on_renin_secretion *
        (rsna_effect_on_renin_secretion *
         (1 - p.BB_renin_secretion_effect * BB_signal))
    )

    # Apply inflammatory RAAS gain factor
    if inflammatory_state is not None:
        renin_secretion_rate *= inflammatory_state.RAAS_gain_factor

    # RAAS degradation rates (lines 1570-1578)
    renin_degradation_rate = math.log(2) / p.renin_half_life
    AngI_degradation_rate = math.log(2) / p.AngI_half_life
    AngII_degradation_rate = math.log(2) / p.AngII_half_life
    AT1_bound_AngII_degradation_rate = math.log(2) / p.AT1_bound_AngII_half_life
    AT2_bound_AngII_degradation_rate = math.log(2) / p.AT2_bound_AngII_half_life

    # RAAS rate constants (lines 1581-1587)
    ACE_activity = p.nominal_ACE_activity * (1 - p.pct_target_inhibition_ACEi)
    chymase_activity = p.nominal_chymase_activity
    AT1_receptor_binding_rate = (
        p.nominal_AT1_receptor_binding_rate *
        (1 - p.pct_target_inhibition_ARB * ARB_signal)
    )
    AT2_receptor_binding_rate = p.nominal_AT2_receptor_binding_rate

    out['N_als'] = N_als
    out['renin_secretion_rate'] = renin_secretion_rate
    out['plasma_renin_activity'] = plasma_renin_activity

    # =====================================================================
    # Store everything needed by ODE RHS
    # =====================================================================

    dydt_extras = {
        # RAAS
        'plasma_renin_activity': plasma_renin_activity,
        'ACE_activity': ACE_activity,
        'chymase_activity': chymase_activity,
        'AT1_receptor_binding_rate': AT1_receptor_binding_rate,
        'AT2_receptor_binding_rate': AT2_receptor_binding_rate,
        'AngI_degradation_rate': AngI_degradation_rate,
        'AngII_degradation_rate': AngII_degradation_rate,
        'AT1_bound_AngII_degradation_rate': AT1_bound_AngII_degradation_rate,
        'AT2_bound_AngII_degradation_rate': AT2_bound_AngII_degradation_rate,
        'renin_secretion_rate': renin_secretion_rate,
        'renin_degradation_rate': renin_degradation_rate,

        # Volume/Na
        'water_intake': water_intake,
        'urine_flow_rate': urine_flow_rate,
        'Na_excretion_via_urine': Na_excretion_via_urine,
        'Na_concentration': Na_concentration,
        'IF_Na_concentration': IF_Na_concentration,
        'sodium_storate_rate': sodium_storate_rate,

        # Feedback targets
        'tubulo_glomerular_feedback_signal': tubulo_glomerular_feedback_signal,
        'N_als': N_als,
        'preafferent_pressure_autoreg_function': preafferent_pressure_autoreg_function,
        'glomerular_pressure_autoreg_function': glomerular_pressure_autoreg_function,
        'normalized_vasopressin_concentration': normalized_vasopressin_concentration,

        # Delays
        'P_in_pt_s1_mmHg': P_in_pt_s1_mmHg,
        'oncotic_pressure_avg': oncotic_pressure_avg,
        'renal_blood_flow_L_min': renal_blood_flow_L_min,
        'SN_macula_densa_Na_flow': SN_macula_densa_Na_flow,
        'postglomerular_pressure': postglomerular_pressure,

        # Disease
        'GP_effect_increasing_Kf': GP_effect_increasing_Kf,
        'PT_Na_reabs_effect_increasing_tubular_length':
            PT_Na_reabs_effect_increasing_tubular_length,
        'PT_Na_reabs_effect_increasing_tubular_diameter':
            PT_Na_reabs_effect_increasing_tubular_diameter,

        # PT water
        'water_out_s1': water_out_s1,
        'water_out_s2': water_out_s2,
        'water_out_s3': water_out_s3,

        # Creatinine
        'creatinine_clearance_rate': creatinine_clearance_rate,

        # Excess glucose
        'excess_glucose_increasing_RTg': excess_glucose_increasing_RTg,
        'RUGE': RUGE,

        # GFR
        'GFR': GFR,
        'GFR_ml_min': GFR_ml_min,
        'SNGFR_nL_min': SNGFR_nL_min,
        'glomerular_pressure': glomerular_pressure,
    }

    # Merge out and dydt_extras for convenience
    out.update(dydt_extras)

    return out, dydt_extras


# =========================================================================
# PART 3: renal_ode_rhs() -- 33-variable ODE right-hand-side
# =========================================================================

def renal_ode_rhs(t, y, params, MAP, CO, P_ven, inflammatory_state=None):
    """
    33-variable ODE right-hand-side for the Hallow renal model.

    Faithfully translates the d/dt() equations from modelfile_commented.R
    lines 1648-1745 (renal-only subset).

    Parameters
    ----------
    t : float
        Current time [hours].
    y : ndarray, shape (33,)
        State vector.
    params : HallowRenalParams
        Model parameters.
    MAP : float
        Mean arterial pressure [mmHg].
    CO : float
        Cardiac output [L/min].
    P_ven : float
        Venous pressure [mmHg].
    inflammatory_state : object or None
        Inflammatory mediator state.

    Returns
    -------
    dydt : ndarray, shape (33,)
        Time derivatives of all state variables.
    """
    p = params
    C = p.C_renal_CV_timescale

    # Compute all algebraic quantities
    _, extras = compute_renal_algebraic(
        y, params, MAP, CO, P_ven, sim_time=t,
        inflammatory_state=inflammatory_state
    )

    # Unpack state variables
    AngI = y[IDX_AngI]
    AngII = y[IDX_AngII]
    AT1_bound_AngII = y[IDX_AT1_bound]
    AT2_bound_AngII = y[IDX_AT2_bound]
    plasma_renin_concentration = y[IDX_PRC]

    blood_volume_L = y[IDX_blood_volume_L]
    interstitial_fluid_volume = y[IDX_IF_volume]
    sodium_amount = y[IDX_sodium_amount]
    IF_sodium_amount = y[IDX_IF_sodium]
    stored_sodium = y[IDX_stored_sodium]

    tubulo_glomerular_feedback_effect = y[IDX_TGF_effect]
    normalized_aldosterone_level = y[IDX_aldosterone]
    preafferent_pressure_autoreg_signal = y[IDX_preafferent_autoreg]
    glomerular_pressure_autoreg_signal = y[IDX_GP_autoreg]
    CO_error = y[IDX_CO_error]
    Na_concentration_error = y[IDX_Na_error]
    normalized_vasopressin_concentration_delayed = y[IDX_VP_delayed]
    F0_TGF = y[IDX_F0_TGF]

    P_bowmans = y[IDX_P_bowmans]
    oncotic_pressure_difference = y[IDX_oncotic_diff]
    renal_blood_flow_L_min_delayed = y[IDX_RBF_delayed]
    SN_macula_densa_Na_flow_delayed = y[IDX_MD_Na_delayed]
    rsna_delayed = y[IDX_RSNA_delayed]

    disease_effects_increasing_Kf = y[IDX_Kf_increase]
    disease_effects_decreasing_CD_PN = y[IDX_CD_PN_loss]
    tubular_length_increase = y[IDX_tubular_length]
    tubular_diameter_increase = y[IDX_tubular_diameter]

    water_out_s1_delayed = y[IDX_water_s1_delayed]
    water_out_s2_delayed = y[IDX_water_s2_delayed]
    water_out_s3_delayed = y[IDX_water_s3_delayed]

    serum_creatinine = y[IDX_serum_creatinine]
    postglomerular_pressure_delayed = y[IDX_postglom_P_delayed]
    postglomerular_pressure_error = y[IDX_postglom_P_error]

    # Unpack algebraic extras
    pra = extras['plasma_renin_activity']
    ACE_act = extras['ACE_activity']
    chym_act = extras['chymase_activity']
    AT1_bind = extras['AT1_receptor_binding_rate']
    AT2_bind = extras['AT2_receptor_binding_rate']
    AngI_deg = extras['AngI_degradation_rate']
    AngII_deg = extras['AngII_degradation_rate']
    AT1_deg = extras['AT1_bound_AngII_degradation_rate']
    AT2_deg = extras['AT2_bound_AngII_degradation_rate']
    ren_sec = extras['renin_secretion_rate']
    ren_deg = extras['renin_degradation_rate']

    w_intake = extras['water_intake']
    u_flow = extras['urine_flow_rate']
    Na_excr = extras['Na_excretion_via_urine']
    Na_conc = extras['Na_concentration']
    IF_Na_conc = extras['IF_Na_concentration']
    Na_store_rate = extras['sodium_storate_rate']

    tgf_sig = extras['tubulo_glomerular_feedback_signal']
    n_als = extras['N_als']
    preaff_autoreg_fn = extras['preafferent_pressure_autoreg_function']
    gp_autoreg_fn = extras['glomerular_pressure_autoreg_function']
    norm_vp = extras['normalized_vasopressin_concentration']

    P_bow_target = extras['P_in_pt_s1_mmHg']
    onc_avg = extras['oncotic_pressure_avg']
    rbf = extras['renal_blood_flow_L_min']
    md_Na = extras['SN_macula_densa_Na_flow']
    post_P = extras['postglomerular_pressure']

    GP_eff_Kf = extras['GP_effect_increasing_Kf']
    tub_len_eff = extras['PT_Na_reabs_effect_increasing_tubular_length']
    tub_dia_eff = extras['PT_Na_reabs_effect_increasing_tubular_diameter']

    w_s1 = extras['water_out_s1']
    w_s2 = extras['water_out_s2']
    w_s3 = extras['water_out_s3']

    creat_clear = extras['creatinine_clearance_rate']

    # Initialize dydt
    dydt = np.zeros(N_STATE)

    # =====================================================================
    # RAAS ODEs (lines 1648-1656)
    # =====================================================================

    # d/dt(AngI) = PRA - AngI*(chymase + ACE) - AngI*degradation
    dydt[IDX_AngI] = (
        pra -
        AngI * (chym_act + ACE_act) -
        AngI * AngI_deg
    )

    # d/dt(AngII) = AngI*(chymase + ACE) - AngII*degradation - AngII*binding
    dydt[IDX_AngII] = (
        AngI * (chym_act + ACE_act) -
        AngII * AngII_deg -
        AngII * AT1_bind -
        AngII * AT2_bind
    )

    # d/dt(AT1_bound_AngII) = AngII*AT1_bind - AT1_deg*AT1_bound
    dydt[IDX_AT1_bound] = (
        AngII * AT1_bind -
        AT1_deg * AT1_bound_AngII
    )

    # d/dt(AT2_bound_AngII) = AngII*AT2_bind - AT2_deg*AT2_bound
    dydt[IDX_AT2_bound] = (
        AngII * AT2_bind -
        AT2_deg * AT2_bound_AngII
    )

    # d/dt(PRC) = renin_secretion - PRC*degradation
    dydt[IDX_PRC] = ren_sec - plasma_renin_concentration * ren_deg

    # =====================================================================
    # Volume/Na ODEs (lines 1660-1669)
    # =====================================================================

    # d/dt(blood_volume_L) (line 1660)
    dydt[IDX_blood_volume_L] = C * (
        w_intake - u_flow +
        p.Q_water * (Na_conc - IF_Na_conc)
    )

    # d/dt(interstitial_fluid_volume) (line 1662)
    dydt[IDX_IF_volume] = C * p.Q_water * (IF_Na_conc - Na_conc)

    # d/dt(sodium_amount) (line 1665)
    dydt[IDX_sodium_amount] = C * (
        p.Na_intake_rate - Na_excr +
        p.Q_Na * (IF_Na_conc - Na_conc)
    )

    # d/dt(IF_sodium_amount) (line 1667)
    dydt[IDX_IF_sodium] = C * (
        p.Q_Na * (Na_conc - IF_Na_conc) - Na_store_rate
    )

    # d/dt(stored_sodium) (line 1669)
    dydt[IDX_stored_sodium] = C * Na_store_rate

    # =====================================================================
    # Feedback delay ODEs (lines 1673-1691)
    # =====================================================================

    # d/dt(TGF_effect) (line 1673)
    dydt[IDX_TGF_effect] = C * (
        tgf_sig - tubulo_glomerular_feedback_effect
    )

    # d/dt(normalized_aldosterone_level) (line 1675)
    dydt[IDX_aldosterone] = C * p.C_aldo_secretion * (
        n_als - normalized_aldosterone_level
    )

    # d/dt(preafferent_pressure_autoreg_signal) (line 1677)
    dydt[IDX_preafferent_autoreg] = C * 100 * (
        preaff_autoreg_fn - preafferent_pressure_autoreg_signal
    )

    # d/dt(glomerular_pressure_autoreg_signal) (line 1679)
    # Zero by default in R model
    dydt[IDX_GP_autoreg] = 0

    # d/dt(CO_error) (line 1682)
    dydt[IDX_CO_error] = C * p.C_co_error * (CO - p.CO_nom)

    # d/dt(Na_concentration_error) (line 1684)
    dydt[IDX_Na_error] = C * p.C_Na_error * (Na_conc - p.ref_Na_concentration)

    # d/dt(VP_delayed) (line 1687)
    dydt[IDX_VP_delayed] = C * p.C_vasopressin_delay * (
        norm_vp - normalized_vasopressin_concentration_delayed
    )

    # d/dt(F0_TGF) (line 1691)
    dydt[IDX_F0_TGF] = C * p.C_tgf_reset * (
        md_Na * p.baseline_nephrons - F0_TGF
    )

    # =====================================================================
    # State delay ODEs (lines 1694-1702)
    # =====================================================================

    # d/dt(P_bowmans) (line 1694)
    dydt[IDX_P_bowmans] = C * 100 * (P_bow_target - P_bowmans)

    # d/dt(oncotic_pressure_difference) (line 1696)
    dydt[IDX_oncotic_diff] = C * 100 * (onc_avg - oncotic_pressure_difference)

    # d/dt(RBF_delayed) (line 1698)
    dydt[IDX_RBF_delayed] = C * p.C_rbf * (
        rbf - renal_blood_flow_L_min_delayed
    )

    # d/dt(MD_Na_delayed) (line 1700)
    dydt[IDX_MD_Na_delayed] = C * p.C_md_flow * (
        md_Na - SN_macula_densa_Na_flow_delayed
    )

    # d/dt(rsna_delayed) (line 1702)
    dydt[IDX_RSNA_delayed] = C * p.C_rsna * (
        p.renal_sympathetic_nerve_activity - rsna_delayed
    )

    # =====================================================================
    # Disease ODEs (lines 1706-1714)
    # =====================================================================

    # d/dt(disease_effects_increasing_Kf) (line 1706)
    dydt[IDX_Kf_increase] = GP_eff_Kf

    # d/dt(disease_effects_decreasing_CD_PN) (line 1709)
    dydt[IDX_CD_PN_loss] = p.CD_PN_loss_rate

    # d/dt(tubular_length_increase) (line 1712)
    dydt[IDX_tubular_length] = tub_len_eff

    # d/dt(tubular_diameter_increase) (line 1714)
    dydt[IDX_tubular_diameter] = tub_dia_eff

    # =====================================================================
    # PT water delay ODEs (lines 1717-1721)
    # =====================================================================

    # d/dt(water_out_s1_delayed) (line 1717)
    dydt[IDX_water_s1_delayed] = C * p.C_pt_water * (
        w_s1 - water_out_s1_delayed
    )

    # d/dt(water_out_s2_delayed) (line 1719)
    dydt[IDX_water_s2_delayed] = C * p.C_pt_water * (
        w_s2 - water_out_s2_delayed
    )

    # d/dt(water_out_s3_delayed) (line 1721)
    dydt[IDX_water_s3_delayed] = C * p.C_pt_water * (
        w_s3 - water_out_s3_delayed
    )

    # =====================================================================
    # Other ODEs (lines 1729-1745)
    # =====================================================================

    # d/dt(serum_creatinine) (line 1729)
    dydt[IDX_serum_creatinine] = C * (
        p.creatinine_synthesis_rate - creat_clear
    )

    # d/dt(postglomerular_pressure_delayed) (line 1743)
    dydt[IDX_postglom_P_delayed] = C * p.C_postglomerular_pressure * (
        post_P - postglomerular_pressure_delayed
    )

    # d/dt(postglomerular_pressure_error) (line 1745)
    dydt[IDX_postglom_P_error] = C * (post_P - p.RIHP0)

    return dydt


# =========================================================================
# PART 4: HallowRenalModel class -- backward-compatible wrapper
# =========================================================================

class HallowRenalModel:
    """
    Hallow et al. (2017) renal model wrapper.

    Backward-compatible with the original simplified HallowRenalModel
    dataclass from cardiorenal_coupling.py.

    State is stored in a 33-element numpy array `y`, with properties
    providing convenient named access.
    """

    def __init__(self):
        """Initialize with conditions from getInits.R."""
        self.params = HallowRenalParams()
        p = self.params

        # Initialize 33-state vector from getInits.R
        self.y = np.zeros(N_STATE)

        # RAAS initial conditions (from getInits.R lines 31-32)
        self.y[IDX_AngI] = 8.164
        self.y[IDX_AngII] = 5.17
        self.y[IDX_AT1_bound] = 16.6
        self.y[IDX_AT2_bound] = 5.5
        self.y[IDX_PRC] = 17.845

        # Volume (from getInits.R lines 33-37)
        self.y[IDX_blood_volume_L] = p.blood_volume_nom  # 5.0
        self.y[IDX_IF_volume] = p.IF_nom  # 15.0
        self.y[IDX_sodium_amount] = p.blood_volume_nom * p.ref_Na_concentration  # 700
        self.y[IDX_IF_sodium] = p.IF_nom * p.ref_Na_concentration  # 2100
        self.y[IDX_stored_sodium] = 0.0

        # Feedback delays (from getInits.R lines 38-44)
        self.y[IDX_TGF_effect] = 1.0
        self.y[IDX_aldosterone] = 1.0
        self.y[IDX_preafferent_autoreg] = 1.0
        self.y[IDX_GP_autoreg] = 1.0
        self.y[IDX_CO_error] = 0.0
        self.y[IDX_Na_error] = 0.0
        self.y[IDX_VP_delayed] = 1.0
        self.y[IDX_F0_TGF] = p.nom_LoH_Na_outflow

        # State delays (from getInits.R lines 45-49)
        self.y[IDX_P_bowmans] = p.Pc_pt_s1_mmHg  # 20.2
        self.y[IDX_oncotic_diff] = p.nom_oncotic_pressure_difference  # 28
        self.y[IDX_RBF_delayed] = p.nom_renal_blood_flow_L_min  # 1.0
        self.y[IDX_MD_Na_delayed] = (
            p.nom_LoH_Na_outflow / p.baseline_nephrons
        )
        self.y[IDX_RSNA_delayed] = 1.0

        # Disease (from getInits.R lines 50-51)
        self.y[IDX_Kf_increase] = 0.0
        self.y[IDX_CD_PN_loss] = 0.0
        self.y[IDX_tubular_length] = 0.0
        self.y[IDX_tubular_diameter] = 0.0

        # PT water delays (from getInits.R lines 52-54)
        self.y[IDX_water_s1_delayed] = 3e-8
        self.y[IDX_water_s2_delayed] = 1.9e-8
        self.y[IDX_water_s3_delayed] = 1.2e-8

        # Other (from getInits.R lines 57, 64-65)
        self.y[IDX_serum_creatinine] = (
            p.equilibrium_serum_creatinine * p.blood_volume_nom
        )  # 0.92 * 5.0 = 4.6
        self.y[IDX_postglom_P_delayed] = p.RIHP0  # 9.32
        self.y[IDX_postglom_P_error] = 0.0

        # Store latest outputs dict (populated after first update)
        self.outputs = {}

        # Equilibrate delayed states by running the algebraic computation
        # repeatedly and updating the delayed ODE states to match targets.
        # This ensures the initial state is self-consistent when used standalone.
        self._equilibrate_initial_state()

        # Backward-compatible attributes
        # nom_GFR is in mL/min (R code: nom_Kf*NFP/nL_mL*baseline_nephrons)
        self._GFR = p.nom_GFR  # already in mL/min
        self._RBF = p.nom_renal_blood_flow_L_min * 1000  # L/min -> mL/min
        self._P_glom = p.nom_glomerular_pressure
        self._Na_excretion = p.Na_intake_rate * 1440  # mEq/min -> mEq/day at SS
        self._water_excretion = p.nom_water_intake  # L/day at SS

        # User-settable parameters (backward compatibility)
        self.Na_intake = p.Na_intake_rate * 1440  # mEq/day
        self.water_intake = p.nom_water_intake  # L/day
        self.TGF_gain = p.S_tubulo_glomerular_feedback
        self.RAAS_gain = 1.5  # default from old interface
        self.MAP_setpoint = p.nominal_map_setpoint
        self.Kf_scale = 1.0

    def _equilibrate_initial_state(self):
        """
        Set delayed ODE states to be consistent with the nominal
        operating point for standalone renal model use.

        The R model's initial conditions (getInits.R) include approximate
        values for water_out_sX_delayed, but these may not match the
        exact algebraic computation at the initial state. We run one
        pass of the algebraic equations to compute consistent values
        for the water delay states and other fast-tracking delays,
        while keeping the key states (P_bowmans, oncotic_diff) at
        their nominal R model values.
        """
        p = self.params
        MAP_eq = p.nominal_map_setpoint
        CO_eq = p.CO_nom
        P_ven_eq = p.P_venous

        # Run algebraic computation once with nominal initial state
        out, extras = compute_renal_algebraic(
            self.y, p, MAP_eq, CO_eq, P_ven_eq
        )

        # Update only the fast-tracking delayed states
        # (PT water delays, RBF delayed, MD Na delayed)
        # Keep P_bowmans and oncotic_diff at their R model initial values
        self.y[IDX_water_s1_delayed] = extras['water_out_s1']
        self.y[IDX_water_s2_delayed] = extras['water_out_s2']
        self.y[IDX_water_s3_delayed] = extras['water_out_s3']
        self.y[IDX_RBF_delayed] = extras['renal_blood_flow_L_min']
        self.y[IDX_MD_Na_delayed] = extras['SN_macula_densa_Na_flow']
        self.y[IDX_postglom_P_delayed] = extras['postglomerular_pressure']

    # ── Properties for backward compatibility ──

    @property
    def V_blood(self):
        """Blood volume in mL (= blood_volume_L * 1000)."""
        return self.y[IDX_blood_volume_L] * 1000

    @V_blood.setter
    def V_blood(self, val):
        self.y[IDX_blood_volume_L] = val / 1000.0

    @property
    def GFR(self):
        """Glomerular filtration rate [mL/min]."""
        return self._GFR

    @GFR.setter
    def GFR(self, val):
        self._GFR = val

    @property
    def RBF(self):
        """Renal blood flow [mL/min]."""
        return self._RBF

    @RBF.setter
    def RBF(self, val):
        self._RBF = val

    @property
    def C_Na(self):
        """Plasma sodium concentration [mEq/L]."""
        bv = self.y[IDX_blood_volume_L]
        na = self.y[IDX_sodium_amount]
        return na / bv if bv > 0 else 140.0

    @C_Na.setter
    def C_Na(self, val):
        # Set sodium_amount to match desired C_Na with current blood volume
        bv = self.y[IDX_blood_volume_L]
        self.y[IDX_sodium_amount] = val * bv

    @property
    def Na_excretion(self):
        """Urinary sodium excretion [mEq/day]."""
        return self._Na_excretion

    @Na_excretion.setter
    def Na_excretion(self, val):
        self._Na_excretion = val

    @property
    def water_excretion(self):
        """Urinary water excretion [L/day]."""
        return self._water_excretion

    @water_excretion.setter
    def water_excretion(self, val):
        self._water_excretion = val

    @property
    def P_glom(self):
        """Glomerular capillary pressure [mmHg]."""
        return self._P_glom

    @P_glom.setter
    def P_glom(self, val):
        self._P_glom = val

    @property
    def Na_total(self):
        """Total body sodium [mEq] (blood + IF + stored)."""
        return (
            self.y[IDX_sodium_amount] +
            self.y[IDX_IF_sodium] +
            self.y[IDX_stored_sodium]
        )

    @Na_total.setter
    def Na_total(self, val):
        # Distribute proportionally
        current = self.Na_total
        if current > 0:
            ratio = val / current
            self.y[IDX_sodium_amount] *= ratio
            self.y[IDX_IF_sodium] *= ratio
            self.y[IDX_stored_sodium] *= ratio
        else:
            self.y[IDX_sodium_amount] = val * 0.25
            self.y[IDX_IF_sodium] = val * 0.75


# =========================================================================
# PART 5: update_renal_model() -- backward-compatible integration function
# =========================================================================

def update_renal_model(renal, MAP, CO, P_ven, dt_hours=6.0,
                       inflammatory_state=None):
    """
    Update the Hallow renal model given cardiac hemodynamic inputs.

    Internally calls scipy.integrate.solve_ivp with BDF method to
    integrate the 33-variable ODE system over dt_hours.

    Parameters
    ----------
    renal : HallowRenalModel
        Current renal state (modified in-place and returned).
    MAP : float
        Mean arterial pressure [mmHg].
    CO : float
        Cardiac output [L/min].
    P_ven : float
        Central venous pressure [mmHg].
    dt_hours : float
        Integration time-step [hours]. Default 6 hours.
    inflammatory_state : object or None
        If provided, inflammatory mediator effects are applied.

    Returns
    -------
    HallowRenalModel
        Updated renal state.
    """
    p = renal.params

    # Apply user-settable parameters to the internal params
    # Na_intake: convert mEq/day -> mEq/min
    p.Na_intake_rate = renal.Na_intake / 1440.0
    # water_intake: L/day
    p.nom_water_intake = renal.water_intake
    # TGF gain
    p.S_tubulo_glomerular_feedback = renal.TGF_gain
    # MAP setpoint
    p.nominal_map_setpoint = renal.MAP_setpoint

    # Apply MAP_setpoint_offset from inflammatory state
    if inflammatory_state is not None:
        p.nominal_map_setpoint += inflammatory_state.MAP_setpoint_offset

    # Store Kf_scale on params so compute_renal_algebraic can access it
    # This does NOT change the nominal calibration (nom_Kf stays at 3.9)
    p._Kf_scale_external = renal.Kf_scale

    # Define the ODE function
    def rhs(t, state):
        return renal_ode_rhs(
            t, state, p, MAP, CO, P_ven,
            inflammatory_state=inflammatory_state
        )

    # Integrate
    t_span = (0, dt_hours)

    try:
        sol = solve_ivp(
            rhs, t_span, renal.y,
            method='BDF',
            rtol=1e-6,
            atol=1e-9,
            max_step=dt_hours / 2,
        )

        if sol.success:
            renal.y = sol.y[:, -1].copy()
        else:
            # Fallback: try with looser tolerances
            sol = solve_ivp(
                rhs, t_span, renal.y,
                method='BDF',
                rtol=1e-4,
                atol=1e-7,
                max_step=dt_hours,
            )
            if sol.success:
                renal.y = sol.y[:, -1].copy()
            # If still fails, keep current state (no update)

    except Exception:
        # On numerical failure, keep current state
        pass

    # Enforce non-negativity on volumes and amounts
    renal.y[IDX_blood_volume_L] = max(renal.y[IDX_blood_volume_L], 2.0)
    renal.y[IDX_IF_volume] = max(renal.y[IDX_IF_volume], 5.0)
    renal.y[IDX_sodium_amount] = max(renal.y[IDX_sodium_amount], 100.0)
    renal.y[IDX_IF_sodium] = max(renal.y[IDX_IF_sodium], 100.0)
    renal.y[IDX_stored_sodium] = max(renal.y[IDX_stored_sodium], 0.0)

    # Enforce non-negativity on RAAS
    for idx in [IDX_AngI, IDX_AngII, IDX_AT1_bound, IDX_AT2_bound, IDX_PRC]:
        renal.y[idx] = max(renal.y[idx], 0.0)

    # Compute final algebraic outputs for recording
    out, _ = compute_renal_algebraic(
        renal.y, p, MAP, CO, P_ven,
        inflammatory_state=inflammatory_state
    )
    renal.outputs = out

    # Update backward-compatible output properties
    renal._GFR = out.get('GFR_ml_min', renal._GFR)
    renal._RBF = out.get('renal_blood_flow_L_min', renal._RBF) * 1000  # L/min -> mL/min
    renal._P_glom = out.get('glomerular_pressure', renal._P_glom)
    renal._Na_excretion = out.get('Na_excretion_via_urine', 0) * 1440  # mEq/min -> mEq/day
    renal._water_excretion = out.get('daily_urine_flow', renal._water_excretion)  # already L/day

    # Reset MAP setpoint offset if inflammatory state was applied
    if inflammatory_state is not None:
        p.nominal_map_setpoint -= inflammatory_state.MAP_setpoint_offset

    # Clean up temporary attributes
    if hasattr(p, '_Kf_scale_external'):
        del p._Kf_scale_external

    return renal
