# Hallow et al. 2017 Parameter Accounting

## Reference-to-Python Translation Guide for Cardiorenal Coupling

This document provides a complete accounting of every parameter defined in `calcNomParams_timescale.R` (the Hallow et al. 2017 renal model reference implementation), with classification for how each parameter maps to the Python cardiorenal coupling framework that uses **CircAdapt VanOsta2024** for the cardiac model instead of Hallow's built-in cardiac model.

### Classification Key

| Category | Meaning |
|---|---|
| `DIRECT` | Used as-is in the Python renal model (faithful translation from R) |
| `ADAPTED` | May need adjustment because the Python system uses CircAdapt for cardiac output instead of Hallow's cardiac sub-model |
| `CARDIAC_ONLY` | Not used in Python -- CircAdapt handles this entirely (cardiac mechanics, vascular loop, hypertrophy) |

---

## 1. Unit Conversions & Constants (Lines 6-25)

These are pure mathematical/physical constants with no model-specific assumptions.

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nL_mL` | `1e+06` | nL/mL | `DIRECT` | Unit conversion constant |
| `dl_ml` | `0.01` | dL/mL | `DIRECT` | Unit conversion constant |
| `L_dL` | `10` | L/dL | `DIRECT` | Unit conversion constant |
| `L_mL` | `1000` | L/mL | `DIRECT` | Unit conversion constant |
| `L_m3` | `0.001` | L/m^3 | `DIRECT` | Unit conversion constant |
| `m3_mL` | `1000000` | m^3/mL | `DIRECT` | Unit conversion constant |
| `m_mm` | `1000` | m/mm | `DIRECT` | Unit conversion constant |
| `g_mg` | `0.001` | g/mg | `DIRECT` | Unit conversion constant |
| `ng_mg` | `1e-06` | ng/mg | `DIRECT` | Unit conversion constant |
| `secs_mins` | `60` | s/min | `DIRECT` | Unit conversion constant |
| `min_hr` | `60` | min/hr | `DIRECT` | Unit conversion constant |
| `min_sec` | `60` | min/sec | `DIRECT` | Unit conversion constant |
| `hr_day` | `24` | hr/day | `DIRECT` | Unit conversion constant |
| `min_day` | `1440` | min/day | `DIRECT` | Unit conversion constant |
| `Pa_mmHg` | `0.0075` | mmHg/Pa | `DIRECT` | Unit conversion constant |
| `MW_creatinine` | `113.12` | g/mol | `DIRECT` | Molecular weight of creatinine |
| `Pi` | `3.1416` | dimensionless | `DIRECT` | Mathematical constant |
| `viscosity_length_constant` | `1.5e-09` | Pa*s*m^4/(m) | `DIRECT` | Used in Poiseuille resistance calculation for renal arterioles: R = viscosity_length_constant / d^4. This is 128*mu*L/(pi) pre-combined for renal vessel geometry |
| `gamma` | `1.16667e-5` | Pa*s | `DIRECT` | Viscosity of tubular fluid; used in tubular pressure calculations |
| `water_intake_species_scale` | `1` | dimensionless | `DIRECT` | Species scaling factor (=1 for human) |

---

## 2. Cardiovascular Parameters (Lines 30-170)

These parameters define Hallow's built-in lumped-parameter cardiac model. In the Python framework, **CircAdapt VanOsta2024 replaces this entire sub-model**, so nearly all parameters here are `CARDIAC_ONLY`. A few serve as interface setpoints for the renal model and are classified `ADAPTED`.

### 2.1 Hemodynamic Setpoints

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `CO_nom` | `5` | L/min | `ADAPTED` | **Nominal cardiac output setpoint.** The renal model uses this as the reference CO for tissue autoregulation feedback (line 53 of modelfile: `tissue_autoreg_scale*(Kp_CO*(CO_delayed - CO_nom*CO_species_scale)...)`). CircAdapt's baseline CO is ~5.2 L/min. **Recommendation:** Set to match CircAdapt's actual baseline CO at initialization, or accept the ~4% mismatch as within physiological tolerance. The renal model's tissue autoregulation will treat the CircAdapt baseline as slightly above setpoint. |
| `BV` | `0.005` | m^3 (= 5 L) | `ADAPTED` | **Total blood volume.** In Hallow's model this initializes the cardiac compartments. In the Python framework, CircAdapt has its own blood volume initialization (~4.7-5.0 L). The renal model's `blood_volume_L` state variable must be initialized to match CircAdapt's actual total volume. **Recommendation:** Query CircAdapt's initial total volume and use that to initialize the renal `blood_volume_L` state. |
| `HR_heart_rate` | `70` | bpm | `CARDIAC_ONLY` | CircAdapt sets its own heart rate. Not used in renal sub-model directly. |

### 2.2 Vascular Resistances (Cardiac Loop)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `R_per0` | `1.27e+08` | Pa*s/m^3 | `CARDIAC_ONLY` | Peripheral resistance -- CircAdapt uses ArtVen elements |
| `R_ven0` | `5e+06` | Pa*s/m^3 | `CARDIAC_ONLY` | Venous resistance |
| `R_art0` | `5e+06` | Pa*s/m^3 | `CARDIAC_ONLY` | Arterial resistance |
| `R_art_pulm` | `3e+06` | Pa*s/m^3 | `CARDIAC_ONLY` | Pulmonary arterial resistance |
| `R_ven_pulm` | `6.44e+06` | Pa*s/m^3 | `CARDIAC_ONLY` | Pulmonary venous resistance |
| `R_r_atrium` | `1e+06` | Pa*s/m^3 | `CARDIAC_ONLY` | Right atrial valve resistance |
| `R_left_atrium` | `1e+06` | Pa*s/m^3 | `CARDIAC_ONLY` | Left atrial (mitral) valve resistance |
| `min_flux` | `5e-07` | m^3/s | `CARDIAC_ONLY` | Minimum valve flow rate |
| `P_ven0` | `0` | Pa | `CARDIAC_ONLY` | Venous reference pressure |
| `P_art0` | `0` | Pa | `CARDIAC_ONLY` | Arterial reference pressure |

### 2.3 Compartment Volumes (Cardiac Loop)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `V_art0` | `0.00045` | m^3 (450 mL) | `CARDIAC_ONLY` | Arterial unstressed volume |
| `V_per0` | `0.00042` | m^3 (420 mL) | `CARDIAC_ONLY` | Peripheral unstressed volume |
| `V_ven0` | `0.0030` | m^3 (3000 mL) | `CARDIAC_ONLY` | Venous unstressed volume |
| `V_pulm0` | `0.00015` | m^3 (150 mL) | `CARDIAC_ONLY` | Pulmonary unstressed volume |
| `V_pulm_art0` | `0.00004` | m^3 (40 mL) | `CARDIAC_ONLY` | Pulmonary arterial unstressed volume |
| `V_pulm_ven0` | `0.00025` | m^3 (250 mL) | `CARDIAC_ONLY` | Pulmonary venous unstressed volume |

### 2.4 Compliances (Cardiac Loop)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `C_art_initial` | `1.10e-08` | m^3/Pa | `CARDIAC_ONLY` | Arterial compliance |
| `C_per` | `1.0e-08` | m^3/Pa | `CARDIAC_ONLY` | Peripheral compliance |
| `C_ven0` | `1.8e-07` | m^3/Pa | `CARDIAC_ONLY` | Venous compliance |
| `C_pulm_ven` | `1.65e-07` | m^3/Pa | `CARDIAC_ONLY` | Pulmonary venous compliance |
| `C_pulm_art` | `2e-8` | m^3/Pa | `CARDIAC_ONLY` | Pulmonary arterial compliance |
| `C_pulm` | `1.65e-7` | m^3/Pa | `CARDIAC_ONLY` | Total pulmonary compliance |
| `C_lv` | `4e-08` | m^3/Pa | `CARDIAC_ONLY` | LV compliance |
| `L_pulm` | `60000` | Pa*s^2/m^3 | `CARDIAC_ONLY` | Pulmonary inertance |
| `L_art` | `60000` | Pa*s^2/m^3 | `CARDIAC_ONLY` | Arterial inertance |

### 2.5 Left Ventricle Mechanics

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `LV_V0_baseline` | `0.000052` | m^3 (52 mL) | `CARDIAC_ONLY` | LV reference volume |
| `LV_V0_min` | `0.00001` | m^3 (10 mL) | `CARDIAC_ONLY` | LV minimum volume |
| `V_w_0` | `0.00012` | m^3 (120 mL) | `CARDIAC_ONLY` | LV wall volume |
| `c_r_LV` | `9` | dimensionless | `CARDIAC_ONLY` | LV radial stiffness exponent |
| `s_r0` | `200` | Pa | `CARDIAC_ONLY` | LV radial stress reference |
| `cf` | `11` | dimensionless | `CARDIAC_ONLY` | LV cardiac stiffness parameter |
| `s_f0` | `900` | Pa | `CARDIAC_ONLY` | LV fiber passive stress reference |
| `sigma_ar` | `55000` | Pa | `CARDIAC_ONLY` | LV active stress reference |
| `contractility` | `1` | dimensionless | `CARDIAC_ONLY` | LV contractility scale |
| `tau_r_LV_twitch_shape` | `0.2` | dimensionless | `CARDIAC_ONLY` | LV twitch rise time fraction |
| `tau_d_LV_twitch_shape` | `0.2` | dimensionless | `CARDIAC_ONLY` | LV twitch decay time fraction |
| `n_r_LV_twitch_shape` | `2` | dimensionless | `CARDIAC_ONLY` | LV twitch rise exponent |
| `n_d_LV_twitch_shape` | `4` | dimensionless | `CARDIAC_ONLY` | LV twitch decay exponent |
| `ls_a0` | `0.0000015` | m | `CARDIAC_ONLY` | Sarcomere activation length |
| `ls_ar_sarcomere_length_effect_in_LV` | `0.000002` | m | `CARDIAC_ONLY` | Sarcomere length effect reference |
| `ls_0_passive_LV_sarcomere_length` | `0.0000019` | m | `CARDIAC_ONLY` | Passive sarcomere length |
| `v0_LV_contraction_velocity_effect_in_LV` | `0.000050` | m/s | `CARDIAC_ONLY` | LV contraction velocity reference |
| `Cv_contraction_velocity_effect_in_LV` | `0` | dimensionless | `CARDIAC_ONLY` | Contraction velocity asymmetry factor |

### 2.6 Right Ventricle Mechanics

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `RV_systolic_time_fraction` | `0.3` | dimensionless | `CARDIAC_ONLY` | RV systolic fraction |
| `RV_V0_min` | `0.00001` | m^3 | `CARDIAC_ONLY` | RV minimum volume |
| `RV_V0` | `0.000075` | m^3 | `CARDIAC_ONLY` | RV reference volume |
| `contractility_RV` | `1` | dimensionless | `CARDIAC_ONLY` | RV contractility |
| `cf_RV` | `8` | dimensionless | `CARDIAC_ONLY` | RV stiffness parameter |
| `s_f0_RV` | `900` | Pa | `CARDIAC_ONLY` | RV fiber passive stress |
| `V_w_0_RV` | `0.0001` | m^3 | `CARDIAC_ONLY` | RV wall volume |
| `c_r_RV` | `9` | dimensionless | `CARDIAC_ONLY` | RV radial stiffness exponent |
| `s_r0_RV` | `200` | Pa | `CARDIAC_ONLY` | RV radial stress reference |
| `sigma_ar_RV` | `55000` | Pa | `CARDIAC_ONLY` | RV active stress reference |
| `ls_a0_RV` | `0.0000015` | m | `CARDIAC_ONLY` | RV sarcomere activation length |
| `v0_RV_contraction_velocity_effect_in_RV` | `0.00005` | m/s | `CARDIAC_ONLY` | RV contraction velocity reference |

### 2.7 Stroke Volume Initial Conditions

| Parameter | R Value / Expression | Unit | Category | Rationale |
|---|---|---|---|---|
| `SVnom` | `CO_nom*1000/HR_heart_rate` = 71.43 | mL | `CARDIAC_ONLY` | Nominal stroke volume |
| `LV_EDV_nom` | `110/1e6` = 1.1e-4 | m^3 | `CARDIAC_ONLY` | Nominal LVEDV |
| `LV_ESV_nom` | `(110 - SVnom)/1e6` ~3.86e-5 | m^3 | `CARDIAC_ONLY` | Nominal LVESV |

### 2.8 Hypertrophy Parameters

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `Baseline_Myocyte_Number` | `3.3e+9` | count | `CARDIAC_ONLY` | CircAdapt does not model myocyte-level remodeling |
| `Baseline_Myocyte_Length` | `0.000115` | m | `CARDIAC_ONLY` | |
| `Baseline_Myocyte_Diameter` | `0.000023313467` | m | `CARDIAC_ONLY` | |
| `Baseline_Myocyte_Volume` | `4.909090909091e-14` | m^3 | `CARDIAC_ONLY` | |
| `max_myocyte_diameter_increase` | `1.5*Baseline_Myocyte_Diameter` | m | `CARDIAC_ONLY` | |
| `max_myocyte_length_increase` | `Baseline_Myocyte_Length` | m | `CARDIAC_ONLY` | |
| `kD_HYPERTROPHY` | `(2*3e-8)/60` = 1e-9 | m/min | `CARDIAC_ONLY` | |
| `kL_HYPERTROPHY` | `(8*2e-9/60)` = 2.67e-10 | m/min | `CARDIAC_ONLY` | |
| `hypertrophy_Cf_slope` | `0.39` | dimensionless | `CARDIAC_ONLY` | |
| `hypertrophy_contractility_slope` | `0` | dimensionless | `CARDIAC_ONLY` | |
| `myo_L_scale` | `1` | dimensionless | `CARDIAC_ONLY` | |
| `myo_D_scale` | `0` | dimensionless | `CARDIAC_ONLY` | |
| `LV_active_stress_threshhold` | `49550` | Pa | `CARDIAC_ONLY` | |
| `LV_passive_stress_along_fiber_threshhold` | `4500` | Pa | `CARDIAC_ONLY` | |

### 2.9 Fibrosis & Tissue Composition

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `Baseline_Interstitial_Fibrosis` | `V_w_0*0.02` = 2.4e-6 | m^3 | `CARDIAC_ONLY` | |
| `Baseline_Replacement_Fibrosis` | `V_w_0*0.02` = 2.4e-6 | m^3 | `CARDIAC_ONLY` | |
| `Baseline_Interstitial_Tissue` | `V_w_0*0.22` = 2.64e-5 | m^3 | `CARDIAC_ONLY` | |

### 2.10 BNP & NTproBNP

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `BNP_factor` | `0.0008` | dimensionless | `CARDIAC_ONLY` | Slope of ln(BNP) vs LVEDP; CircAdapt does not compute BNP directly |

### 2.11 Vascular Responsiveness & Autoregulation (Systemic)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `vascular_responsiveness_scale` | `1` | dimensionless | `CARDIAC_ONLY` | Systemic vascular responsiveness -- CircAdapt handles SVR via ArtVen |
| `TPR_scale_peripheral_resistance` | `1` | dimensionless | `CARDIAC_ONLY` | TPR scaling |
| `compliance_scale_arterial_compliance` | `1` | dimensionless | `CARDIAC_ONLY` | Arterial compliance scaling |
| `disease_effect_on_TPR_peripheral_resistance` | `1` | dimensionless | `CARDIAC_ONLY` | Disease effect on TPR |
| `contractility_scale_LV_active_stress` | `1` | dimensionless | `CARDIAC_ONLY` | Contractility scaling |
| `c_contr_loss` | `1` | dimensionless | `CARDIAC_ONLY` | |
| `Kp_CO` | `0.1` | dimensionless | `CARDIAC_ONLY` | CO proportional gain for tissue autoregulation |
| `Ki_CO` | `0.001` | dimensionless | `CARDIAC_ONLY` | CO integral gain |
| `tissue_autoreg_scale` | `0.1` | dimensionless | `CARDIAC_ONLY` | Tissue autoregulation scale |
| `venous_autoregulation_signal_Km` | `4` | dimensionless | `CARDIAC_ONLY` | |
| `venous_autoregulation_signal_scale` | `0.5` | dimensionless | `CARDIAC_ONLY` | |
| `venous_autoregulation_signal_slope` | `0.75` | dimensionless | `CARDIAC_ONLY` | |
| `min_tissue_autoreg` | `0.4` | dimensionless | `CARDIAC_ONLY` | |
| `Vmax_tissue_autoreg` | `2` | dimensionless | `CARDIAC_ONLY` | |
| `Km_tissue_autoreg` | `1.47` | dimensionless | `CARDIAC_ONLY` | |
| `hill_tissue_autoreg` | `2` | dimensionless | `CARDIAC_ONLY` | |
| `Stiffness_BP_slope` | `0.01` | 1/mmHg | `CARDIAC_ONLY` | BP effect on arterial stiffness |
| `C_art_scale` | `1` | dimensionless | `CARDIAC_ONLY` | |
| `stretch_min_LV_passive_stress_along_fiber` | `1` | dimensionless | `CARDIAC_ONLY` | |
| `stretch_scale_LV_passive_stress_along_fiber` | `0` | dimensionless | `CARDIAC_ONLY` | |

### 2.12 Initial Volume Distribution (Cardiac Loop)

These are computed initial conditions distributing `BV` across Hallow's cardiac compartments.

| Parameter | Expression | Unit | Category | Rationale |
|---|---|---|---|---|
| `venous_volume_0` | complex expression (line 164) | m^3 | `CARDIAC_ONLY` | Initial venous volume |
| `LV_volume_0` | `LV_V0_baseline` | m^3 | `CARDIAC_ONLY` | Initial LV volume |
| `arterial_volume_0` | complex expression (line 166) | m^3 | `CARDIAC_ONLY` | Initial arterial volume |
| `peripheral_circulation_volume_0` | complex expression (line 167) | m^3 | `CARDIAC_ONLY` | Initial peripheral volume |
| `RV_volume_0` | complex expression (line 168) | m^3 | `CARDIAC_ONLY` | Initial RV volume |
| `pulmonary_arterial_volume_0` | complex expression (line 169) | m^3 | `CARDIAC_ONLY` | Initial pulmonary arterial volume |
| `pulmonary_venous_volume_0` | complex expression (line 170) | m^3 | `CARDIAC_ONLY` | Initial pulmonary venous volume |

---

## 3. Systemic Renal Parameters (Lines 176-200)

These are the core systemic parameters that set the operating point of the renal model.

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nominal_map_setpoint` | `85` | mmHg | `ADAPTED` | **MAP setpoint for renal autoregulation.** This value is used as the reference for (1) preafferent myogenic autoregulation, (2) arterial stiffness calculation, and (3) nominal TPR calculation. CircAdapt's resting MAP is ~86 mmHg. **Recommendation:** Keep at 85 mmHg. The 1 mmHg difference is within physiological beat-to-beat variability and will not meaningfully affect renal autoregulation. The renal myogenic response is designed to buffer small MAP deviations. |
| `IF_nom` | `15` | L | `DIRECT` | Nominal interstitial fluid volume |
| `blood_volume_nom` | `5` | L | `ADAPTED` | **Nominal blood volume.** Must match CircAdapt's actual blood volume at initialization. CircAdapt typically initializes with ~5.0 L. **Recommendation:** Query CircAdapt initial volume; if it differs, adjust `blood_volume_L` initial condition accordingly. |
| `Na_intake_rate` | `100/24/60` = 0.0694 | mEq/min | `DIRECT` | Sodium intake rate (100 mEq/day = 2300 mg/day) |
| `nom_water_intake` | `2.1` | L/day | `DIRECT` | Nominal water intake |
| `ref_Na_concentration` | `140` | mEq/L | `DIRECT` | Reference plasma sodium concentration |
| `glucose_concentration` | `5.5` | mmol/L | `DIRECT` | Plasma glucose (normal fasting) |
| `plasma_albumin_concentration` | `35` | mg/mL | `DIRECT` | Plasma albumin |
| `plasma_protein_concentration` | `7` | g/dL | `DIRECT` | Total plasma protein |
| `plasma_urea` | `0` | mmol/L | `DIRECT` | Plasma urea (set to 0 in reference; urea module optional) |
| `nom_serum_uric_acid_concentration` | `5.5` | mg/dL | `DIRECT` | Serum uric acid |
| `equilibrium_serum_creatinine` | `0.92` | mg/dL | `DIRECT` | Equilibrium serum creatinine |
| `P_venous` | `4` | mmHg | `ADAPTED` | **Central venous pressure reference for renal blood flow.** RBF = (MAP - P_venous) / RVR. In the Python framework, CircAdapt provides CVP directly. **Recommendation:** Use CircAdapt's CVP output each coupling step rather than this fixed value. At baseline CircAdapt CVP is ~3-5 mmHg, close to this value. |

---

## 4. Renal Vascular Parameters (Lines 192-204)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nom_renal_blood_flow_L_min` | `1` | L/min | `DIRECT` | Nominal renal blood flow (~20% of CO) |
| `baseline_nephrons` | `2e6` | count | `DIRECT` | Total number of nephrons (both kidneys) |
| `nom_Kf` | `3.9` | nL/min/mmHg | `DIRECT` | **Single-nephron glomerular ultrafiltration coefficient.** Critical parameter for GFR calculation: SNGFR = Kf * (P_glom - oncotic - P_bowmans). Value of 3.9 is from Hallow 2017. |
| `nom_oncotic_pressure_difference` | `28` | mmHg | `DIRECT` | Average glomerular oncotic pressure difference (initial guess; updated dynamically) |
| `P_renal_vein` | `4` | mmHg | `DIRECT` | Renal venous pressure |
| `nom_oncotic_pressure_peritubular` | `28.05` | mmHg | `DIRECT` | Peritubular oncotic pressure |
| `interstitial_oncotic_pressure` | `5` | mmHg | `DIRECT` | Interstitial oncotic pressure |
| `nom_preafferent_arteriole_resistance` | `14` | mmHg/(L/min) | `DIRECT` | Preafferent arteriolar resistance |
| `nom_afferent_diameter` | `1.65e-5` | m (16.5 um) | `DIRECT` | Afferent arteriole diameter |
| `nom_efferent_diameter` | `1.1e-5` | m (11 um) | `DIRECT` | Efferent arteriole diameter |

---

## 5. Tubular Geometry (Lines 207-228)

All tubular geometry parameters are purely renal and transfer directly.

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `Dc_pt_nom` | `27e-6` | m (27 um) | `DIRECT` | Proximal tubule nominal diameter |
| `Dc_lh` | `17e-6` | m (17 um) | `DIRECT` | Loop of Henle diameter |
| `Dc_dt` | `17e-6` | m (17 um) | `DIRECT` | Distal tubule diameter |
| `Dc_cd` | `22e-6` | m (22 um) | `DIRECT` | Collecting duct diameter |
| `L_pt_s1_nom` | `0.005` | m (5 mm) | `DIRECT` | PT segment 1 length |
| `L_pt_s2_nom` | `0.005` | m (5 mm) | `DIRECT` | PT segment 2 length |
| `L_pt_s3_nom` | `0.004` | m (4 mm) | `DIRECT` | PT segment 3 length |
| `L_lh_des` | `0.01` | m (10 mm) | `DIRECT` | Descending loop of Henle length |
| `L_lh_asc` | `0.01` | m (10 mm) | `DIRECT` | Ascending loop of Henle length |
| `L_dct` | `0.005` | m (5 mm) | `DIRECT` | Distal convoluted tubule length |
| `L_cd` | `L_lh_des` = 0.01 | m (10 mm) | `DIRECT` | Collecting duct length |
| `tubular_compliance` | `0.2` | dimensionless | `DIRECT` | Tubular wall compliance exponent |
| `Pc_pt_s1_mmHg` | `20.2` | mmHg | `DIRECT` | PT S1 compliance pressure |
| `Pc_pt_s2_mmHg` | `15` | mmHg | `DIRECT` | PT S2 compliance pressure |
| `Pc_pt_s3_mmHg` | `11` | mmHg | `DIRECT` | PT S3 compliance pressure |
| `Pc_lh_des_mmHg` | `8` | mmHg | `DIRECT` | Descending LoH compliance pressure |
| `Pc_lh_asc_mmHg` | `7` | mmHg | `DIRECT` | Ascending LoH compliance pressure |
| `Pc_dt_mmHg` | `6` | mmHg | `DIRECT` | DT compliance pressure |
| `Pc_cd_mmHg` | `5` | mmHg | `DIRECT` | CD compliance pressure |
| `P_interstitial_mmHg` | `5` | mmHg | `DIRECT` | Renal interstitial pressure |

---

## 6. Reabsorption Fractions (Lines 229-232)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nominal_pt_na_reabsorption` | `0.7` | fraction | `DIRECT` | **PT fractional Na reabsorption** (70% of filtered Na reabsorbed in PT) |
| `nominal_loh_na_reabsorption` | `0.8` | fraction | `DIRECT` | **LoH fractional Na reabsorption** (80% of PT outflow reabsorbed in ascending LoH) |
| `nominal_dt_na_reabsorption` | `0.5` | fraction | `DIRECT` | **DT fractional Na reabsorption** (50% of LoH outflow reabsorbed in DCT) |
| `LoH_flow_dependence` | `0.75` | dimensionless | `DIRECT` | Flow dependence of LoH reabsorption rate |

**Note:** The CD reabsorption fraction (`nominal_cd_na_reabsorption`) is not a free parameter -- it is **derived** at initialization (see Section 21) to ensure Na balance: Na excretion = Na intake at steady state. It evaluates to approximately 0.93.

---

## 7. Glucose Parameters (Lines 235-246)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nom_glucose_reabs_per_unit_length_s1` | `5.4e-5` | mmol/min/m | `DIRECT` | SGLT2 reabsorption rate in S1 |
| `nom_glucose_reabs_per_unit_length_s2` | `0` | mmol/min/m | `DIRECT` | SGLT2 reabsorption rate in S2 (set to 0) |
| `nom_glucose_reabs_per_unit_length_s3` | `2.8e-5` | mmol/min/m | `DIRECT` | SGLT1 reabsorption rate in S3 |
| `diabetic_adaptation` | `1` | dimensionless | `DIRECT` | Diabetic adaptation factor |
| `maximal_RTg_increase` | `0.3` | fraction | `DIRECT` | Max increase in glucose transport capacity |
| `T_glucose_RTg` | `6000000` | min | `DIRECT` | Time constant for glucose transport adaptation |
| `glucose_natriuresis_effect_pt` | `0` | dimensionless | `DIRECT` | Osmotic natriuresis in PT (off by default) |
| `glucose_natriuresis_effect_cd` | `0` | dimensionless | `DIRECT` | Osmotic natriuresis in CD (off by default) |
| `glucose_diuresis_effect_pt` | `0` | dimensionless | `DIRECT` | Osmotic diuresis in PT (off by default) |
| `glucose_diuresis_effect_cd` | `0` | dimensionless | `DIRECT` | Osmotic diuresis in CD (off by default) |

---

## 8. Urea Parameters (Line 249)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `urea_permeability_PT` | `0.5` | dimensionless | `DIRECT` | PT urea permeability coefficient |

---

## 9. Albumin Sieving (Lines 253-262)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nom_glomerular_albumin_sieving_coefficient` | `0.00062` | dimensionless | `DIRECT` | Glomerular albumin sieving coefficient |
| `SN_albumin_reabsorptive_capacity` | `1.4e-6` | mg/min | `DIRECT` | Single-nephron albumin reabsorption capacity |
| `Emax_seiving` | `4` | dimensionless | `DIRECT` | Max GP effect on sieving |
| `Gamma_seiving` | `3` | dimensionless | `DIRECT` | Hill coefficient for GP-sieving relationship |
| `Km_seiving` | `25` | mmHg | `DIRECT` | Half-max GP for sieving effect |
| `max_PT_albumin_reabsorption_rate` | `0.1` | mg/min | `DIRECT` | Max PT albumin reabsorption rate |
| `nom_albumin_excretion_rate` | `3.5e-9` | mg/min | `DIRECT` | Baseline albumin excretion rate per nephron |
| `nom_GP_seiving_damage` | `65` | mmHg | `DIRECT` | GP threshold for sieving damage |
| `c_albumin` | `0.0231` | min/nL | `DIRECT` | Dean & Lazzara constant for GFR-dependent sieving |
| `seiving_inf` | `4.25e-4` | dimensionless | `DIRECT` | Asymptotic sieving coefficient (from Dean & Lazzara) |

---

## 10. RAAS Kinetics (Lines 265-506)

### 10.1 Equilibrium Values and Half-Lives

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `concentration_to_renin_activity_conversion_plasma` | `61` | (fmol/mL/hr)/(fmol/mL) | `DIRECT` | PRA = PRC * conversion factor |
| `nominal_equilibrium_PRA` | `1000` | fmol/mL/hr | `DIRECT` | Equilibrium plasma renin activity |
| `nominal_equilibrium_AngI` | `7.5` | fmol/mL | `DIRECT` | Equilibrium Ang I concentration |
| `nominal_equilibrium_AngII` | `4.75` | fmol/mL | `DIRECT` | Equilibrium Ang II concentration |
| `nominal_renin_half_life` | `0.1733` | hr | `DIRECT` | Renin half-life (~10.4 min) |
| `nominal_AngI_half_life` | `0.5/60` = 0.00833 | hr | `DIRECT` | Ang I half-life (0.5 min) |
| `nominal_AngII_half_life` | `0.66/60` = 0.011 | hr | `DIRECT` | Ang II half-life (0.66 min) |
| `nominal_AT1_bound_AngII_half_life` | `12/60` = 0.2 | hr | `DIRECT` | AT1-bound Ang II half-life (12 min) |
| `nominal_AT2_bound_AngII_half_life` | `12/60` = 0.2 | hr | `DIRECT` | AT2-bound Ang II half-life (12 min) |
| `ACE_chymase_fraction` | `0.95` | fraction | `DIRECT` | Fraction of Ang I converted by ACE (vs chymase) |
| `fraction_AT1_bound_AngII` | `0.75` | fraction | `DIRECT` | Fraction of bound Ang II on AT1 (vs AT2) receptors |

### 10.2 RAAS Operational Parameters (Lines 491-506)

These are the actual runtime RAAS constants (may differ from the nominal values above which are used only for equilibrium calculation).

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `AngI_half_life` | `0.008333` | hr | `DIRECT` | Runtime Ang I half-life |
| `AngII_half_life` | `0.011` | hr | `DIRECT` | Runtime Ang II half-life |
| `AT1_bound_AngII_half_life` | `0.2` | hr | `DIRECT` | Runtime AT1-bound half-life |
| `AT1_PRC_slope` | `-0.9` | dimensionless | `DIRECT` | AT1 feedback slope on PRC (negative = negative feedback) |
| `AT1_PRC_yint` | `0` | dimensionless | `DIRECT` | AT1 feedback y-intercept on PRC |
| `AT2_bound_AngII_half_life` | `0.2` | hr | `DIRECT` | Runtime AT2-bound half-life |
| `concentration_to_renin_activity_conversion_plasma` | `61` | (fmol/mL/hr)/(fmol/mL) | `DIRECT` | (repeated from above, same value) |
| `fraction_AT1_bound_AngII` | `0.75` | fraction | `DIRECT` | (repeated from above, same value) |
| `nominal_ACE_activity` | `48.9` | 1/hr | `DIRECT` | ACE activity (overrides calculated value) |
| `nominal_AT1_receptor_binding_rate` | `12.1` | 1/hr | `DIRECT` | AT1 binding rate (overrides calculated value) |
| `nominal_AT2_receptor_binding_rate` | `4.0` | 1/hr | `DIRECT` | AT2 binding rate |
| `nominal_chymase_activity` | `1.25` | 1/hr | `DIRECT` | Chymase activity |
| `nominal_equilibrium_AT1_bound_AngII` | `16.63` | fmol/mL | `DIRECT` | Equilibrium AT1-bound Ang II (overrides calculated value) |
| `nominal_equilibrium_PRC` | `16.4` | fmol/mL | `DIRECT` | Equilibrium PRC (overrides calculated value) |
| `renin_half_life` | `0.1733` | hr | `DIRECT` | Renin half-life |

---

## 11. AT1 Effect Parameters (Lines 395-411)

These parameters control the effect of AT1-bound Ang II on renal vascular resistance and aldosterone secretion.

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `AT1_svr_slope` | `0` | dimensionless | `DIRECT` | AT1 effect on SVR (off by default) |
| `AT1_preaff_scale` | `0.8` | dimensionless | `DIRECT` | AT1 effect magnitude on preafferent resistance |
| `AT1_preaff_slope` | `16` | fmol/mL | `DIRECT` | AT1 effect slope on preafferent resistance |
| `AT1_aff_scale` | `0.8` | dimensionless | `DIRECT` | AT1 effect magnitude on afferent resistance |
| `AT1_aff_slope` | `16` | fmol/mL | `DIRECT` | AT1 effect slope on afferent resistance |
| `AT1_eff_scale` | `0.8` | dimensionless | `DIRECT` | AT1 effect magnitude on efferent resistance |
| `AT1_eff_slope` | `16` | fmol/mL | `DIRECT` | AT1 effect slope on efferent resistance |
| `AT1_PT_scale` | `0` | dimensionless | `DIRECT` | AT1 effect on PT reabsorption (off by default) |
| `AT1_PT_slope` | `16` | fmol/mL | `DIRECT` | AT1 effect slope on PT reabsorption |
| `AT1_aldo_slope` | `0.02` | 1/(fmol/mL) | `DIRECT` | AT1 effect slope on aldosterone secretion |
| `AT1_aff_EC50` | `1e-9` | mol/L | `DIRECT` | AT1 EC50 for afferent effect |
| `AT1_eff_EC50` | `nominal_equilibrium_AT1_bound_AngII*1e-12` | mol/L | `DIRECT` | AT1 EC50 for efferent effect |
| `Emax_AT1_eff` | `0` | dimensionless | `DIRECT` | Max AT1 efferent effect (off) |
| `Emax_AT1_aff` | `0` | dimensionless | `DIRECT` | Max AT1 afferent effect (off) |
| `AT1_hill` | `15` | dimensionless | `DIRECT` | Hill coefficient for AT1 dose-response |
| `AngII_effect_on_venous_compliance` | `1` | dimensionless | `CARDIAC_ONLY` | AngII effect on venous compliance (cardiac loop parameter) |

---

## 12. Aldosterone Parameters (Lines 416-421)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `nominal_aldosterone_concentration` | `85` | pg/mL | `DIRECT` | Baseline aldosterone |
| `aldo_DCT_scale` | `0` | dimensionless | `DIRECT` | Aldosterone effect on DCT (off by default) |
| `aldo_DCT_slope` | `0.5` | dimensionless | `DIRECT` | Aldosterone DCT effect slope |
| `aldo_CD_scale` | `0.2` | dimensionless | `DIRECT` | Aldosterone effect magnitude on CD |
| `aldo_CD_slope` | `0.5` | dimensionless | `DIRECT` | Aldosterone CD effect slope |
| `aldo_renin_slope` | `-0.4` | dimensionless | `DIRECT` | Aldosterone effect on renin secretion (negative feedback) |

---

## 13. ANP Parameters (Lines 424-437)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `normalized_atrial_NP_concentration` | `1` | dimensionless | `DIRECT` | Baseline normalized ANP (=1 by definition) |
| `nom_ANP` | `50` | pg/mL | `DIRECT` | Nominal ANP concentration |
| `ANP_aff_scale` | `0.2` | dimensionless | `DIRECT` | ANP effect on afferent resistance (commented out in modelfile, but parameter exists) |
| `ANP_aff_slope` | `1` | dimensionless | `DIRECT` | ANP afferent slope |
| `ANP_preaff_scale` | `0` | dimensionless | `DIRECT` | ANP effect on preafferent (off) |
| `ANP_preaff_slope` | `1` | dimensionless | `DIRECT` | ANP preafferent slope |
| `ANP_eff_scale` | `0` | dimensionless | `DIRECT` | ANP effect on efferent (off) |
| `ANP_eff_slope` | `1` | dimensionless | `DIRECT` | ANP efferent slope |
| `anp_CD_scale` | `-0.1` | dimensionless | `DIRECT` | ANP effect on CD Na reabsorption (negative = inhibits) |
| `anp_CD_slope` | `2` | dimensionless | `DIRECT` | ANP CD effect slope |
| `ANP_effect_on_venous_compliance` | `1` | dimensionless | `CARDIAC_ONLY` | Venous compliance effect (cardiac loop) |
| `LVEDP_ANP_slope` | `20` | mmHg | `ADAPTED` | **Slope of ANP vs LVEDP relationship (Maeda 1998).** The Hallow model computes ANP = nom_ANP * exp(max(0, LVEDP*0.0075 - 10) / LVEDP_ANP_slope). This requires LVEDP. **CircAdapt does not provide LVEDP directly.** However, the Python coupling can approximate LVEDP from CVP or from CircAdapt's LV end-diastolic pressure (available as `model['Cavity']['p']` for the LV cavity at end-diastole). **Recommendation:** Extract LVEDP from CircAdapt's LV cavity pressure waveform at end-diastole, or use CVP as an approximation (LVEDP ~ CVP + 2-5 mmHg in the absence of mitral valve disease). |
| `ANP_infused_concentration` | `0` | pg/mL | `DIRECT` | Exogenous ANP infusion (off) |
| `ANP_infusion` | `0` | dimensionless | `DIRECT` | ANP infusion flag (off) |

---

## 14. RSNA Parameters (Lines 442-457)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `renal_sympathetic_nerve_activity` | `1` | dimensionless | `ADAPTED` | **RSNA baseline value.** Both models use RSNA=1 as the nominal value. CircAdapt does not generate RSNA dynamically. **Recommendation:** Keep constant at 1.0 unless a separate sympathetic nervous system model is added. The renal model's RSNA effects (preafferent vasoconstriction, renin secretion) are designed to be neutral at RSNA=1. |
| `nom_rsna` | `1` | dimensionless | `DIRECT` | Nominal RSNA reference value |
| `rsna_preaff_scale` | `0.2` | dimensionless | `DIRECT` | RSNA effect on preafferent resistance |
| `rsna_preaff_slope` | `0.25` | dimensionless | `DIRECT` | RSNA preafferent slope |
| `rsna_PT_scale` | `0` | dimensionless | `DIRECT` | RSNA effect on PT reabsorption (off) |
| `rsna_PT_slope` | `1` | dimensionless | `DIRECT` | RSNA PT slope |
| `rsna_CD_scale` | `0` | dimensionless | `DIRECT` | RSNA effect on CD reabsorption (off) |
| `rsna_CD_slope` | `1` | dimensionless | `DIRECT` | RSNA CD slope |
| `rsna_renin_slope` | `1` | dimensionless | `DIRECT` | RSNA effect on renin secretion |
| `rsna_svr_slope` | `0` | dimensionless | `CARDIAC_ONLY` | RSNA effect on SVR (cardiac loop) |
| `rsna_HR_slope` | `0` | dimensionless | `CARDIAC_ONLY` | RSNA effect on heart rate (cardiac loop) |
| `sna_effect_on_contractility` | `1` | dimensionless | `CARDIAC_ONLY` | SNA effect on contractility |
| `SNA_effect_on_venous_compliance` | `1` | dimensionless | `CARDIAC_ONLY` | SNA effect on venous compliance |
| `B2sna_effect_on_TPR` | `1` | dimensionless | `CARDIAC_ONLY` | Beta-2 SNA effect on TPR |
| `A1sna_effect_on_TPR` | `1` | dimensionless | `CARDIAC_ONLY` | Alpha-1 SNA effect on TPR |

---

## 15. Vasopressin / ADH (Lines 461-471)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `Na_controller_gain` | `0.05` | dimensionless | `DIRECT` | Sodium-VP controller gain |
| `Kp_VP` | `2` | dimensionless | `DIRECT` | Proportional gain for VP PI controller |
| `Ki_VP` | `0.005` | dimensionless | `DIRECT` | Integral gain for VP PI controller |
| `nom_ADH_urea_permeability` | `0.98` | fraction | `DIRECT` | ADH-regulated urea permeability |
| `nom_ADH_water_permeability` | `0.98` | fraction | `DIRECT` | ADH-regulated water permeability |
| `nominal_vasopressin_conc` | `4` | pg/mL | `DIRECT` | Nominal vasopressin concentration |
| `water_intake_vasopressin_scale` | `0.25` | dimensionless | `DIRECT` | VP effect on water intake magnitude |
| `water_intake_vasopressin_slope` | `-0.5` | dimensionless | `DIRECT` | VP effect on water intake slope |

---

## 16. TGF Parameters (Lines 475-490)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `S_tubulo_glomerular_feedback` | `0.7` | dimensionless | `DIRECT` | TGF sensitivity (Eq. 9 in paper) |
| `F_md_scale_tubulo_glomerular_feedback` | `6` | mEq/L | `DIRECT` | TGF macula densa concentration scale |
| `MD_Na_concentration_setpoint` | `63.29` | mEq/L | `DIRECT` | Macula densa Na concentration setpoint for TGF |
| `md_renin_A` | `1` | dimensionless | `DIRECT` | MD effect on renin secretion amplitude |
| `md_renin_tau` | `1` | dimensionless | `DIRECT` | MD effect on renin secretion time constant |
| `preaff_diameter_range` | `0.25` | dimensionless | `DIRECT` | Max preafferent diameter modulation range |
| `afferent_diameter_range` | `1.2e-05` | m | `DIRECT` | Max afferent diameter modulation range |
| `efferent_diameter_range` | `3e-06` | m | `DIRECT` | Max efferent diameter modulation range |
| `preaff_signal_nonlin_scale` | `4` | dimensionless | `DIRECT` | Preafferent signal saturation steepness |
| `afferent_signal_nonlin_scale` | `4` | dimensionless | `DIRECT` | Afferent signal saturation steepness |
| `efferent_signal_nonlin_scale` | `4` | dimensionless | `DIRECT` | Efferent signal saturation steepness |

---

## 17. Timescale Constants (Lines 508-543)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `C_renal_CV_timescale` | `60` | dimensionless | `ADAPTED` | **Timescale coupling factor.** In Hallow's original model, this scales ODE rates to accelerate the renal model relative to the cardiac model (since the cardiac model runs on a beat-to-beat timescale but renal processes evolve over minutes-hours). In the Python framework, the coupling is done via message passing at each coupling step. **Recommendation:** Set to 1.0 if the Python renal model's time step is already in minutes. Or retain the value if the Python ODE integration uses the same time units as R. This needs careful calibration during integration testing. |
| `C_cycle` | `50` | 1/min | `CARDIAC_ONLY` | Cardiac cycle tracking |
| `C_cycle2` | `100` | 1/min | `CARDIAC_ONLY` | Cardiac delay filter |
| `C_cycle3` | `100` | 1/min | `CARDIAC_ONLY` | Peak stress filter |
| `C_co` | `0.1` | 1/min | `CARDIAC_ONLY` | CO smoothing filter |
| `C_co_delay` | `0.25` | 1/min | `CARDIAC_ONLY` | CO delay filter |
| `C_map` | `0.25` | 1/min | `CARDIAC_ONLY` | MAP smoothing filter |
| `time_step` | `1/C_cycle` = 0.02 | min | `CARDIAC_ONLY` | Cardiac integration time step |
| `C_co_error` | `1` | dimensionless | `CARDIAC_ONLY` | CO error integrator gain |
| `C_vasopressin_delay` | `1` | 1/min | `DIRECT` | Vasopressin effect delay |
| `Q_water` | `1` | L/min | `DIRECT` | Blood-to-IF water transfer rate |
| `Q_Na` | `1` | mEq/min | `DIRECT` | Blood-to-IF Na transfer rate |
| `Q_Na_store` | `0` | mEq/min | `DIRECT` | Na storage rate (off) |
| `max_stored_sodium` | `500` | mEq | `DIRECT` | Max Na storage capacity |
| `C_Na_error` | `1` | dimensionless | `DIRECT` | Na concentration error integrator |
| `C_aldo_secretion` | `100` | dimensionless | `DIRECT` | Aldosterone secretion rate |
| `C_tgf_reset` | `0` | dimensionless | `DIRECT` | TGF resetting rate (off by default) |
| `C_md_flow` | `0.06` | dimensionless | `DIRECT` | MD flow delay constant |
| `C_tgf` | `1` | dimensionless | `DIRECT` | TGF effect time constant |
| `C_rbf` | `100` | dimensionless | `DIRECT` | RBF delay filter constant |
| `C_serum_creatinine` | `1` | dimensionless | `DIRECT` | Creatinine kinetics rate |
| `C_pt_water` | `1` | dimensionless | `DIRECT` | PT water delay filter |
| `C_rsna` | `100` | dimensionless | `DIRECT` | RSNA delay filter |
| `C_postglomerular_pressure` | `1` | dimensionless | `DIRECT` | Postglomerular pressure delay filter |

---

## 18. Drug Effects (Lines 545-645)

### 18.1 Diuretics

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `HCTZ_effect_on_DT_Na_reabs` | `1` | dimensionless | `DIRECT` | Hydrochlorothiazide effect on DT (1 = no drug) |
| `HCTZ_effect_on_renin_secretion` | `1` | dimensionless | `DIRECT` | HCTZ effect on renin |
| `loop_diuretic_effect` | `1` | dimensionless | `DIRECT` | Loop diuretic effect (1 = no drug) |

### 18.2 Calcium Channel Blockers

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `CCB_effect_on_preafferent_resistance` | `1` | dimensionless | `DIRECT` | CCB on preafferent (1 = no drug) |
| `CCB_effect_on_afferent_resistance` | `1` | dimensionless | `DIRECT` | CCB on afferent |
| `CCB_effect_on_efferent_resistance` | `1` | dimensionless | `DIRECT` | CCB on efferent |

### 18.3 RAAS Inhibitors

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `pct_target_inhibition_MRA` | `0` | fraction | `DIRECT` | MRA inhibition (0 = no drug) |
| `pct_target_inhibition_ARB` | `0` | fraction | `DIRECT` | ARB inhibition |
| `pct_target_inhibition_ACEi` | `0` | fraction | `DIRECT` | ACEi inhibition |
| `pct_target_inhibition_DRI` | `0` | fraction | `DIRECT` | DRI inhibition |
| `ARB_is_on` | `0` | flag | `DIRECT` | ARB active flag |

### 18.4 Beta Blockers

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `BB_TPR_effect` | `1` | dimensionless | `CARDIAC_ONLY` | BB effect on TPR |
| `BB_cardiac_relax_effect` | `0` | dimensionless | `CARDIAC_ONLY` | BB effect on relaxation |
| `BB_venous_compliance_effect` | `0` | dimensionless | `CARDIAC_ONLY` | BB effect on venous compliance |
| `BB_preafferent_R_effect` | `1` | dimensionless | `DIRECT` | BB effect on preafferent resistance |
| `BB_renin_secretion_effect` | `1` | dimensionless | `DIRECT` | BB effect on renin secretion |
| `BB_HR_effect` | `1` | dimensionless | `CARDIAC_ONLY` | BB effect on heart rate |
| `BB_contractility_effect` | `1` | dimensionless | `CARDIAC_ONLY` | BB effect on contractility |
| `BB_is_on` | `0` | flag | `DIRECT` | BB active flag |
| `k_PD` | `0.001` | 1/min | `DIRECT` | PK/PD drug onset rate |

### 18.5 SGLT2 Inhibitors

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `SGLT2_inhibition` | `1` | dimensionless | `DIRECT` | SGLT2 inhibition (1 = no drug, <1 = inhibited) |
| `SGLT1_inhibition` | `1` | dimensionless | `DIRECT` | SGLT1 inhibition |
| `C_sglt2_delay` | `0.1*60` = 6 | min | `DIRECT` | SGLT2 onset delay |
| `C_ruge` | `0.0001*60` = 0.006 | min | `DIRECT` | Urinary glucose excretion delay |
| `Anhe3` | `0` | dimensionless | `DIRECT` | NHE3 inhibition coupling (off) |
| `deltaCanp` | `0` | dimensionless | `DIRECT` | Endogenous ANP modification (off) |
| `ANP_effect_on_Arterial_Resistance` | `0` | dimensionless | `DIRECT` | ANP effect on arterial resistance (off) |

### 18.6 Miscellaneous

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `K_Na_ratio_effect_on_aldo` | `1` | dimensionless | `DIRECT` | K/Na ratio effect on aldosterone |

---

## 19. Pressure Natriuresis (Lines 584-607)

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `Kp_PN` | `1` | dimensionless | `DIRECT` | Pressure natriuresis proportional gain |
| `Kd_PN` | `0` | dimensionless | `DIRECT` | Pressure natriuresis derivative gain (off) |
| `Ki_PN` | `0` | dimensionless | `DIRECT` | Pressure natriuresis integral gain (off) |
| `max_pt_reabs_rate` | `0.995` | fraction | `DIRECT` | Max PT reabsorption rate |
| `pressure_natriuresis_PT_scale` | `0.5` | dimensionless | `DIRECT` | PT pressure natriuresis magnitude |
| `pressure_natriuresis_PT_slope` | `1` | dimensionless | `DIRECT` | PT pressure natriuresis slope |
| `pressure_natriuresis_LoH_scale` | `0` | dimensionless | `DIRECT` | LoH pressure natriuresis (off) |
| `pressure_natriuresis_LoH_slope` | `1` | dimensionless | `DIRECT` | LoH pressure natriuresis slope |
| `pressure_natriuresis_DCT_scale` | `0` | dimensionless | `DIRECT` | DCT pressure natriuresis (off) |
| `pressure_natriuresis_DCT_slope` | `1` | dimensionless | `DIRECT` | DCT pressure natriuresis slope |
| `max_cd_reabs_rate` | `0.995` | fraction | `DIRECT` | Max CD reabsorption rate |
| `pressure_natriuresis_CD_scale` | `0.5` | dimensionless | `DIRECT` | CD pressure natriuresis magnitude |
| `pressure_natriuresis_CD_slope` | `1` | dimensionless | `DIRECT` | CD pressure natriuresis slope |
| `RBF_CD_scale` | `1` | dimensionless | `DIRECT` | RBF effect on CD reabsorption magnitude |
| `RBF_CD_slope` | `0.3` | L/min | `DIRECT` | RBF effect on CD reabsorption slope |
| `CD_PN_loss_rate` | `0` | 1/min | `DIRECT` | Rate of PN mechanism loss in diabetes (off) |
| `RIHP0` | `9.32` | mmHg | `DIRECT` | Nominal renal interstitial hydrostatic pressure setpoint |

### 19.1 Renal Autoregulation

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `gp_autoreg_scale` | `0` | dimensionless | `DIRECT` | Glomerular pressure autoregulation (off) |
| `preaff_autoreg_scale` | `0` | dimensionless | `DIRECT` | Preafferent autoregulation (off) |
| `myogenic_steepness` | `2` | mmHg | `DIRECT` | Myogenic response steepness |
| `RBF_autoreg_scale` | `0` | dimensionless | `DIRECT` | RBF autoregulation (off) |
| `RBF_autoreg_steepness` | `1` | L/min | `DIRECT` | RBF autoregulation steepness |

---

## 20. Disease Effects (Lines 613-644)

### 20.1 Glomerular Hypertrophy

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `maximal_glom_surface_area_increase` | `0.5` | fraction | `DIRECT` | Max glomerular surface area increase |
| `T_glomerular_pressure_increases_Kf` | `120000` | min | `DIRECT` | Time constant for GP-induced Kf increase |

### 20.2 Tubular Hypertrophy

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `maximal_tubule_length_increase` | `0` | fraction | `DIRECT` | Max tubule length increase (off) |
| `maximal_tubule_diameter_increase` | `0` | fraction | `DIRECT` | Max tubule diameter increase (off) |
| `T_PT_Na_reabs_PT_length` | `1e10` | min | `DIRECT` | Time constant (effectively off) |
| `T_PT_Na_reabs_PT_diameter` | `1e10` | min | `DIRECT` | Time constant (effectively off) |

### 20.3 Disease Effects on Kf and Nephrons

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `disease_effects_decreasing_Kf` | `0` | dimensionless | `DIRECT` | Kf reduction from glomerulosclerosis (off) |
| `disease_effect_on_nephrons` | `0` | fraction | `DIRECT` | Nephron loss (off) |

### 20.4 Na Reabsorption Saturation

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `max_s1_Na_reabs` | `7.5e-6` | mEq/min | `DIRECT` | Max S1 Na reabsorption rate |
| `max_s2_Na_reabs` | `2e-6` | mEq/min | `DIRECT` | Max S2 Na reabsorption rate |
| `max_s3_Na_reabs` | `1` | mEq/min | `DIRECT` | Max S3 Na reabsorption rate (effectively unlimited) |
| `max_deltaLoH_reabs` | `0.75e-6` | mEq/min | `DIRECT` | Max LoH flow-dependent reabsorption increase |
| `CD_Na_reabs_threshold` | `7e-7` | mEq/min | `DIRECT` | CD Na reabsorption threshold |

### 20.5 Species/Scale Parameters

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `water_intake_species_scale` | `1` | dimensionless | `DIRECT` | Species water intake scale (human) |
| `CO_species_scale` | `1` | dimensionless | `DIRECT` | Species CO scale (human) |

### 20.6 Aortic Stenosis / Valvular Disease

| Parameter | R Value | Unit | Category | Rationale |
|---|---|---|---|---|
| `heart_renal_link` | `1` | flag | `ADAPTED` | **Heart-renal coupling flag.** In Hallow's model, this enables bidirectional blood volume coupling. In the Python framework, this should always be 1 (coupling is the whole point). **Recommendation:** Always set to 1. |
| `aortic_valve_stenosis` | `0` | flag | `CARDIAC_ONLY` | Aortic stenosis flag |
| `R_art_stenosis_factor` | `0` | dimensionless | `CARDIAC_ONLY` | Stenosis severity |
| `stenosis_rate` | `0.005` | 1/min | `CARDIAC_ONLY` | Stenosis progression rate |
| `mitral_regurgitation` | `0` | flag | `CARDIAC_ONLY` | Mitral regurgitation flag |
| `mitral_regurgitation_pressure_diff` | `1e10` | Pa | `CARDIAC_ONLY` | MR pressure threshold |
| `max_mitral_diff` | `19000` | Pa | `CARDIAC_ONLY` | Max MR pressure diff |
| `min_mitral_diff` | `16500` | Pa | `CARDIAC_ONLY` | Min MR pressure diff |
| `k_mitral_diff` | `0.05` | 1/min | `CARDIAC_ONLY` | MR progression rate |
| `aortic_regurgitation` | `0` | flag | `CARDIAC_ONLY` | Aortic regurgitation flag |
| `aortic_regurgitation_pressure_diff` | `1e10` | Pa | `CARDIAC_ONLY` | AR pressure threshold |
| `max_aortic_diff` | `7000` | Pa | `CARDIAC_ONLY` | Max AR pressure diff |
| `min_aortic_diff` | `5500` | Pa | `CARDIAC_ONLY` | Min AR pressure diff |
| `k_aortic_diff` | `0.05` | 1/min | `CARDIAC_ONLY` | AR progression rate |

---

## 21. Derived Parameters (Lines 280-360)

These parameters are computed at initialization from the parameters defined above. They establish the steady-state operating point of the renal model.

| Parameter | Expression | Approx Value | Unit | Category | Rationale |
|---|---|---|---|---|---|
| `nom_preafferent_pressure` | `nominal_map_setpoint - nom_renal_blood_flow_L_min * nom_preafferent_arteriole_resistance` | 85 - 1*14 = 71 | mmHg | `ADAPTED` | **Preafferent pressure setpoint.** Depends on MAP setpoint. If CircAdapt MAP differs from 85, this shifts proportionally. **Recommendation:** Recompute dynamically or accept that a 1 mmHg MAP difference yields ~1 mmHg shift here. |
| `nom_glomerular_pressure` | `nom_preafferent_pressure - nom_renal_blood_flow_L_min * (L_m3 * viscosity_length_constant / (nom_afferent_diameter^4) / baseline_nephrons)` | ~50-55 | mmHg | `DIRECT` | Nominal glomerular pressure (computed from renal parameters only) |
| `nom_postglomerular_pressure` | `nom_preafferent_pressure - nom_renal_blood_flow_L_min * (L_m3 * viscosity_length_constant * (1/nom_afferent_diameter^4 + 1/nom_efferent_diameter^4) / baseline_nephrons)` | ~9-12 | mmHg | `DIRECT` | Nominal postglomerular pressure |
| `nom_GFR` | `nom_Kf * (nom_glomerular_pressure - nom_oncotic_pressure_difference - Pc_pt_s1_mmHg) / nL_mL * baseline_nephrons` | ~100-120 | mL/min | `DIRECT` | Nominal GFR |
| `nom_filtered_sodium_load` | `nom_GFR / L_mL * ref_Na_concentration` | ~14-17 | mEq/min | `DIRECT` | Nominal filtered Na load |
| `nom_filtered_glucose_load` | `glucose_concentration * nom_GFR / 1000` | ~0.55-0.66 | mmol/min | `DIRECT` | Nominal filtered glucose load |
| `nom_glucose_pt_out_s1` | `max(0, nom_filtered_glucose_load - nom_glucose_reabs_per_unit_length_s1 * L_pt_s1_nom * baseline_nephrons)` | computed | mmol/min | `DIRECT` | Glucose outflow after S1 |
| `nom_glucose_pt_out_s2` | `max(0, nom_glucose_pt_out_s1 - nom_glucose_reabs_per_unit_length_s2 * L_pt_s2_nom * baseline_nephrons)` | computed | mmol/min | `DIRECT` | Glucose outflow after S2 |
| `nom_glucose_pt_out_s3` | `max(0, nom_glucose_pt_out_s2 - nom_glucose_reabs_per_unit_length_s3 * L_pt_s3_nom * baseline_nephrons)` | computed | mmol/min | `DIRECT` | Glucose outflow after S3 |
| `nom_SGTL2_Na_reabs_mmol_s1` | `nom_filtered_glucose_load - nom_glucose_pt_out_s1` | computed | mmol/min | `DIRECT` | SGLT2-coupled Na reabsorption in S1 |
| `nom_SGTL2_Na_reabs_mmol_s2` | `nom_glucose_pt_out_s1 - nom_glucose_pt_out_s2` | computed | mmol/min | `DIRECT` | SGLT2-coupled Na reabsorption in S2 |
| `nom_SGTL1_Na_reabs_mmol` | `2 * (nom_glucose_pt_out_s2 - nom_glucose_pt_out_s3)` | computed | mmol/min | `DIRECT` | SGLT1-coupled Na reabsorption |
| `nom_total_SGLT_Na_reabs` | sum of above | computed | mmol/min | `DIRECT` | Total SGLT-coupled Na reabsorption |
| `nom_SGLT_fractional_na_reabs` | `nom_total_SGLT_Na_reabs / nom_filtered_sodium_load` | computed | fraction | `DIRECT` | Fractional SGLT Na reabsorption |
| `nominal_pt_na_reabsorption_nonSGLT` | `nominal_pt_na_reabsorption - nom_SGLT_fractional_na_reabs` | computed | fraction | `DIRECT` | Non-SGLT PT Na reabsorption fraction |
| `nom_Na_reabs_per_unit_length` | `-log(1 - nominal_pt_na_reabsorption_nonSGLT) / (L_pt_s1_nom + L_pt_s2_nom + L_pt_s3_nom)` | computed | 1/m | `DIRECT` | Na reabsorption rate per unit length |
| `nom_Na_pt_s1_reabs` | `nom_filtered_sodium_load * (1 - exp(-nom_Na_reabs_per_unit_length * L_pt_s1_nom))` | computed | mEq/min | `DIRECT` | S1 Na reabsorption |
| `nom_Na_pt_out_s1` | `nom_filtered_sodium_load - nom_Na_pt_s1_reabs - nom_SGTL2_Na_reabs_mmol_s1` | computed | mEq/min | `DIRECT` | Na outflow from S1 |
| `nom_Na_pt_s2_reabs` | `nom_Na_pt_out_s1 * (1 - exp(-nom_Na_reabs_per_unit_length * L_pt_s2_nom))` | computed | mEq/min | `DIRECT` | S2 Na reabsorption |
| `nom_Na_pt_out_s2` | `nom_Na_pt_out_s1 - nom_Na_pt_s2_reabs - nom_SGTL2_Na_reabs_mmol_s2` | computed | mEq/min | `DIRECT` | Na outflow from S2 |
| `nom_Na_pt_s3_reabs` | `nom_Na_pt_out_s2 * (1 - exp(-nom_Na_reabs_per_unit_length * L_pt_s3_nom))` | computed | mEq/min | `DIRECT` | S3 Na reabsorption |
| `nom_Na_pt_out_s3` | `nom_Na_pt_out_s2 - nom_Na_pt_s3_reabs - nom_SGTL1_Na_reabs_mmol` | computed | mEq/min | `DIRECT` | Na outflow from S3 |
| `nom_PT_Na_outflow` | `nom_Na_pt_out_s3` | computed | mEq/min | `DIRECT` | Total PT Na outflow |
| `nom_Na_in_AscLoH` | `nom_PT_Na_outflow / baseline_nephrons` | computed | mEq/min/nephron | `DIRECT` | SN Na inflow to ascending LoH |
| `AscLoH_Reab_Rate` | `(2 * nominal_loh_na_reabsorption * nom_Na_in_AscLoH) / L_lh_des` | computed | mEq/min/m | `DIRECT` | LoH reabsorption rate per unit length |
| `nom_LoH_Na_outflow` | `nom_PT_Na_outflow * (1 - nominal_loh_na_reabsorption)` | computed | mEq/min | `DIRECT` | LoH Na outflow |
| `nom_DT_Na_outflow` | `nom_LoH_Na_outflow * (1 - nominal_dt_na_reabsorption)` | computed | mEq/min | `DIRECT` | DT Na outflow |
| `nominal_cd_na_reabsorption` | `1 - Na_intake_rate / nom_DT_Na_outflow` | **~0.93** | fraction | `DIRECT` | **CD Na reabsorption fraction -- derived to ensure Na balance.** This is the key calibration: at steady state, Na excretion = Na intake, so CD must reabsorb (1 - intake/delivery). |
| `nom_RVR` | `(nominal_map_setpoint - P_venous) / nom_renal_blood_flow_L_min` | 81 | mmHg/(L/min) | `DIRECT` | Nominal renal vascular resistance |
| `nom_peritubular_resistance` | `nom_RVR - (nom_preafferent_arteriole_resistance + ...)` | computed | mmHg/(L/min) | `DIRECT` | Nominal peritubular resistance |
| `PT_Na_reab_perUnitSA_0` | computed expression | computed | mEq/min/m^2 | `DIRECT` | PT Na reabsorption per unit surface area |
| `nom_TPR` | `nominal_map_setpoint / CO_nom` = 17 | mmHg/(L/min) | `ADAPTED` | **Nominal TPR.** Depends on MAP and CO setpoints. In Python, CircAdapt determines TPR via its ArtVen elements. **Recommendation:** This is used only in Hallow's cardiac sub-model and is not directly used in the renal equations. |
| `tubular_reabsorption` | `nom_GFR/1000 - nom_water_intake * water_intake_species_scale / 60 / 24` | computed | L/min | `DIRECT` | Net tubular fluid reabsorption at SS |
| `nom_peritubular_cap_Kf` | computed from Starling equation | computed | L/min/mmHg | `DIRECT` | Peritubular capillary Kf |
| `creatinine_synthesis_rate` | `equilibrium_serum_creatinine * dl_ml * nom_GFR` | computed | mg/min | `DIRECT` | Creatinine production rate |

### RAAS Derived Parameters

| Parameter | Expression | Approx Value | Unit | Category | Rationale |
|---|---|---|---|---|---|
| `nominal_equilibrium_PRC` | `nominal_equilibrium_PRA / concentration_to_renin_activity_conversion_plasma` | 16.4 | fmol/mL | `DIRECT` | Equilibrium PRC |
| `nominal_AngI_degradation_rate` | `log(2) / nominal_AngI_half_life` | 83.2 | 1/hr | `DIRECT` | Ang I degradation rate |
| `nominal_AngII_degradation_rate` | `log(2) / nominal_AngII_half_life` | 63.0 | 1/hr | `DIRECT` | Ang II degradation rate |
| `nominal_AT1_bound_AngII_degradation_rate` | `log(2) / nominal_AT1_bound_AngII_half_life` | 3.47 | 1/hr | `DIRECT` | AT1-bound Ang II degradation rate |
| `nominal_AT2_bound_AngII_degradation_rate` | `log(2) / nominal_AT2_bound_AngII_half_life` | 3.47 | 1/hr | `DIRECT` | AT2-bound Ang II degradation rate |
| `nominal_ACE_activity` | computed from equilibrium | ~48.9 | 1/hr | `DIRECT` | ACE activity |
| `nominal_chymase_activity` | computed from equilibrium | ~1.25 | 1/hr | `DIRECT` | Chymase activity |
| `nominal_AT1_receptor_binding_rate` | computed from equilibrium | ~12.1 | 1/hr | `DIRECT` | AT1 binding rate |
| `nominal_AT2_receptor_binding_rate` | computed from equilibrium | ~4.0 | 1/hr | `DIRECT` | AT2 binding rate |
| `nominal_equilibrium_AT1_bound_AngII` | computed from equilibrium | ~16.63 | fmol/mL | `DIRECT` | Equilibrium AT1-bound Ang II |
| `nominal_equilibrium_AT2_bound_AngII` | computed from equilibrium | computed | fmol/mL | `DIRECT` | Equilibrium AT2-bound Ang II |

---

## 22. Parameters That Were Wrong in the Previous Python Implementation

The following parameters were incorrectly set in an earlier version of the Python renal model and have been corrected to match the Hallow et al. 2017 reference values from `calcNomParams_timescale.R`.

| Parameter | Old (Wrong) Value | New (Correct) Value | Unit | Impact of Error |
|---|---|---|---|---|
| `Kf` (nom_Kf) | **8.0** | **3.9** | nL/min/mmHg | Old value ~2x too high. Produced unrealistically high GFR (~200+ mL/min). Correct value from Hallow 2017 gives GFR ~100-120 mL/min at nominal operating point. |
| `MAP_setpoint` (nominal_map_setpoint) | **93** | **85** | mmHg | Old value was 8 mmHg too high, shifting the entire renal autoregulation setpoint and pressure-natriuresis curve rightward. This masked hypertension effects and elevated baseline GFR/RBF. |
| `Na_intake` (Na_intake_rate) | **150** mEq/day | **100** mEq/day (= 100/24/60 mEq/min) | mEq/day | Old value represented a high-sodium Western diet. Hallow reference uses 100 mEq/day (2300 mg/day) which is the RDA. Higher intake forces the CD to work harder, shifts Na balance. |
| `R_preAA` (nom_preafferent_arteriole_resistance) | **12** | **14** | mmHg/(L/min) | Old value 14% too low. Reduces the pressure drop proximal to the glomerulus, elevating glomerular pressure and GFR. |
| `eta_PT` (nominal_pt_na_reabsorption) | **0.67** | **0.70** | fraction | Old value slightly too low. Increases Na delivery to downstream segments, requiring higher LoH/DT/CD reabsorption to maintain balance. |
| `eta_LoH` (nominal_loh_na_reabsorption) | **0.25** | **0.80** | fraction | **Major error.** Old value was ~3x too low. The LoH reabsorbs ~80% of delivered Na in reality (thick ascending limb NKCC2 cotransporter). With 25%, massive Na overload was delivered to the DCT and CD. This likely caused unrealistic compensatory CD reabsorption rates and unstable Na balance. |
| `eta_DT` (nominal_dt_na_reabsorption) | **0.05** | **0.50** | fraction | **Major error.** Old value was 10x too low. The DCT (NCC cotransporter) reabsorbs ~50% of delivered Na. With only 5%, almost all Na reached the CD, forcing it to bear the entire burden of Na balance. |
| `eta_CD` (nominal_cd_na_reabsorption) | **0.024** | **~0.93** (calibrated) | fraction | **Major error.** This is not a free parameter -- it is derived from the constraint that Na excretion = Na intake at steady state. The old value of 0.024 (2.4%) would excrete ~50x the intake rate, making steady state impossible. The correct value of ~0.93 means the CD reabsorbs ~93% of the small amount of Na that reaches it (after PT, LoH, and DT have already reabsorbed ~97% of the filtered load). |

### Summary of Cascading Effects

The LoH, DT, and CD errors were deeply interrelated. The tubular reabsorption cascade works as follows:

```
Filtered Na (100%)
  -> PT reabsorbs 70% -> 30% delivered to LoH
  -> LoH reabsorbs 80% of 30% = 24% -> 6% delivered to DT
  -> DT reabsorbs 50% of 6% = 3% -> 3% delivered to CD
  -> CD reabsorbs ~93% of 3% -> ~0.2% excreted = Na intake
```

With the old values:
```
Filtered Na (100%)
  -> PT reabsorbs 67% -> 33% delivered to LoH
  -> LoH reabsorbs 25% of 33% = 8.25% -> 24.75% delivered to DT
  -> DT reabsorbs 5% of 24.75% = 1.24% -> 23.5% delivered to CD
  -> CD reabsorbs 2.4% of 23.5% = 0.56% -> 22.9% excreted
```

The old implementation would have tried to excrete ~23% of the filtered Na load -- approximately 100x the correct rate -- guaranteeing Na depletion and hemodynamic instability.

---

## 23. CircAdapt Coupling Interface Concerns

### Summary of All ADAPTED Parameters

| # | Parameter | Concern | Recommendation |
|---|---|---|---|
| 1 | `CO_nom` = 5 L/min | CircAdapt baseline CO is ~5.2 L/min | Accept ~4% mismatch or set to CircAdapt's actual baseline. The tissue autoregulation signal treats the 0.2 L/min surplus as mild hyperemia. |
| 2 | `BV` = 5 L | CircAdapt may initialize at 4.7-5.0 L | Query CircAdapt initial volume. Initialize `blood_volume_L` to match. |
| 3 | `nominal_map_setpoint` = 85 mmHg | CircAdapt baseline MAP ~86 mmHg | Keep at 85. The 1 mmHg difference is within beat-to-beat variability. |
| 4 | `blood_volume_nom` = 5 L | Must match CircAdapt | Same as #2 |
| 5 | `P_venous` = 4 mmHg | CircAdapt provides CVP dynamically | Use CircAdapt CVP output at each coupling step rather than this constant. |
| 6 | `LVEDP_ANP_slope` = 20 mmHg | CircAdapt doesn't output LVEDP directly | Extract LV end-diastolic pressure from CircAdapt's cavity pressure waveform, or approximate from CVP. |
| 7 | `renal_sympathetic_nerve_activity` = 1 | CircAdapt does not generate RSNA | Keep at 1.0 (neutral). Both models treat RSNA=1 as the default. |
| 8 | `C_renal_CV_timescale` = 60 | Hallow uses this to accelerate renal ODEs relative to cardiac cycle | Set to 1.0 if Python renal time step is in minutes, or calibrate to ensure stable convergence. |
| 9 | `nom_preafferent_pressure` (derived) | Depends on MAP setpoint | Recompute at initialization using CircAdapt's actual baseline MAP if desired. |
| 10 | `nom_TPR` = 17 (derived) | Depends on MAP and CO setpoints | Not used in renal equations directly; only in cardiac sub-model. |
| 11 | `heart_renal_link` = 1 | Must be on for coupling | Always set to 1. |

### Blood Volume Initialization Protocol

The most critical coupling concern is blood volume consistency:

1. CircAdapt initializes with a specific total blood volume (sum across all cardiovascular compartments).
2. The renal model's `blood_volume_L` state variable tracks total intravascular volume.
3. At initialization, `blood_volume_L` must equal CircAdapt's total volume.
4. During simulation, renal model changes to `blood_volume_L` (from water intake/excretion) are fed back to CircAdapt as the new total volume.
5. CircAdapt redistributes this volume across its compartments according to its own compliance/pressure relationships.

### ANP/LVEDP Interface

The Hallow model computes ANP from LVEDP (line 437 of modelfile):
```
LVEDP_ANP_effect = exp(max(0, LV_EDP*0.0075 - 10) / LVEDP_ANP_slope)
ANP = nom_ANP * LVEDP_ANP_effect
```

Where `LV_EDP` is in Pa and `0.0075` converts to mmHg. This means ANP increases exponentially when LVEDP exceeds ~10 mmHg. CircAdapt can provide this via the LV cavity pressure at the end-diastolic time point (just before the QRS complex in each cardiac cycle). The Python wrapper should extract this from the CircAdapt pressure waveform.

---

## Appendix: Parameter Count Summary

| Category | Count |
|---|---|
| `DIRECT` (faithful renal translation) | ~180 |
| `ADAPTED` (needs CircAdapt interface attention) | ~11 |
| `CARDIAC_ONLY` (handled by CircAdapt) | ~85 |
| **Total unique parameters** | **~276** |

The vast majority of parameters (DIRECT) transfer faithfully from the R reference to the Python renal model because they govern intrarenal physiology (tubular reabsorption, RAAS kinetics, glomerular filtration) that is independent of the cardiac model choice. The ADAPTED parameters cluster around the heart-kidney interface (MAP setpoint, CO setpoint, blood volume, CVP, LVEDP for ANP, RSNA, timescale) and require careful initialization and message-passing protocol design.
