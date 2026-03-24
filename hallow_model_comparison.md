# Hallow Model: R Reference vs. Python Implementation — Discrepancy Report

**Date**: 2025-03-24
**Reference**: `hallow_model/modelfile_commented.R`, `hallow_model/calcNomParams_timescale.R`
**Implementation**: `cardiorenal_coupling.py` (`HallowRenalModel`, `update_renal_model()`)

---

## Summary

The Python implementation in `cardiorenal_coupling.py` is a **heavily simplified steady-state reduction** of the full Hallow et al. (2017) ODE-based renal model. Many subsystems present in the R reference are either absent or replaced with algebraic approximations. This document catalogs every discrepancy found, organized by severity.

---

## CRITICAL Issues (Incorrect Physiology / Wrong Numerical Results)

### C1. Tubular Reabsorption Fractions — Order-of-Magnitude Errors

| Segment | R Model | Python | Ratio |
|---------|---------|--------|-------|
| PT | 0.70 (total, incl. SGLT) | 0.67 | ~OK |
| **LoH** | **0.80** | **0.25** | **3.2x too low** |
| **DT** | **0.50** | **0.05** | **10x too low** |
| CD | Calibrated (~0.93 at default intake) | 0.024 | **~39x too low** |

**Impact**: At steady state (MAP=93, GFR=120, C_Na=140):

- **R model**: Excretes ~0.07 mEq/min → ~100 mEq/day (matches 100 mEq/day intake)
- **Python model**: Na_after_CD = 16.8 × 0.33 × 0.75 × 0.95 × 0.976 = **3.86 mEq/min → 5551 mEq/day** (37x intake!)

The Python model relies on pressure-natriuresis and TGF to bring excretion into line, but the base tubular math produces grossly non-physiological values. The model cannot reach sodium balance at the stated default parameters without extreme compensatory adjustments.

**Fix**: Adopt the R model's fractional reabsorption values:
```python
eta_PT:  float = 0.70   # was 0.67
eta_LoH: float = 0.80   # was 0.25  (CRITICAL)
eta_DT:  float = 0.50   # was 0.05  (CRITICAL)
# eta_CD should be calibrated: eta_CD = 1 - Na_intake_rate / Na_delivered_to_CD
```
For CD, compute at initialization:
```python
Na_to_CD = Na_filt * (1-0.70) * (1-0.80) * (1-0.50)
eta_CD0 = 1.0 - (Na_intake / 1440.0) / Na_to_CD  # Calibrate to balance intake
```

---

### C2. Kf (Ultrafiltration Coefficient) — 2x Too High

| Parameter | R Model | Python |
|-----------|---------|--------|
| Kf | **3.9** nL/min/mmHg | **8.0** nL/min/mmHg |

**Impact**: The doubled Kf inflates baseline SNGFR, interacting with the broken tubular fractions. The R model's Kf=3.9 is calibrated against its own resistance network and oncotic pressure model; the Python value of 8.0 was likely compensating for other simplifications but is not faithful to the reference.

**Fix**: Set `Kf = 3.9` and recalibrate the resistance network to produce GFR ~120 mL/min.

---

### C3. MAP Setpoint — 8 mmHg Higher Than Reference

| Parameter | R Model | Python |
|-----------|---------|--------|
| MAP setpoint | **85** mmHg | **93** mmHg |

**Impact**: The MAP setpoint determines the quiescence point for RAAS and pressure-natriuresis. An 8 mmHg offset shifts all pressure-volume curves. The Python model's setpoint of 93 corresponds to a normotensive MAP value, while the R model uses 85 mmHg as the **renal** autoregulatory setpoint. These serve different physiological roles — 93 is closer to whole-body MAP, while 85 is the renal perfusion setpoint.

**Fix**: Use `MAP_setpoint = 85` for the renal model's internal regulation, or explicitly document that 93 is an intentional modification for the coupled model.

---

### C4. Na Intake — 50% Higher Than Reference

| Parameter | R Model | Python |
|-----------|---------|--------|
| Na intake | **100** mEq/day | **150** mEq/day |

**Impact**: Combined with the CD reabsorption calibration issue (C1), this compounds the sodium balance error. The R model calibrates CD reabsorption to balance 100 mEq/day; the Python uses a fixed eta_CD0=0.024 with 150 mEq/day intake, leading to massive imbalance.

**Fix**: Either use 100 mEq/day (faithful to R model) or recalibrate eta_CD0 for 150 mEq/day intake.

---

### C5. Preafferent Resistance — Slightly Different

| Parameter | R Model | Python |
|-----------|---------|--------|
| R_preAA | **14** mmHg·min/L | **12** mmHg·min/L |

**Impact**: Shifts P_gc and RBF calculations. Together with C2 (Kf), C3 (MAP setpoint), and the missing per-nephron resistance division (see S1), the entire hemodynamic operating point differs.

**Fix**: Use `R_preAA = 14`.

---

## STRUCTURAL Issues (Missing Subsystems)

### S1. Renal Vascular Resistance — Wrong Structure

**R model** (Poiseuille per-nephron resistance):
```r
nom_afferent_resistance = L_m3 * viscosity_constant / (diameter^4)
RVR = R_preAA + (R_AA + R_EA) / n_functional_glomeruli + R_peritubular
RBF = (MAP - P_venous) / RVR
```
- Afferent and efferent resistances are per-nephron (divided by nephron count)
- Sigmoid saturation limits vasoreactivity: `1/(1+exp(4*(1-signal))) + 0.5`
- Includes peritubular resistance

**Python model** (lumped constants):
```python
R_total = R_preAA + R_AA + R_EA  # Simple sum, NOT per-nephron
RBF = (MAP - P_renal_vein) / R_total * 1000
```
- No per-nephron division
- No sigmoid saturation
- No peritubular resistance

**Fix**:
```python
R_total = R_preAA + (R_AA + R_EA) / (2 * N_nephrons) + R_peritubular
# Where R_AA and R_EA are computed from Poiseuille's law:
# R_AA = L_m3 * viscosity_constant / d_AA^4
# R_EA = L_m3 * viscosity_constant / d_EA^4
# Apply sigmoid saturation to limit vasoreactivity
```

Note: The Python model's lumped R_AA0=26 and R_EA0=43 may already incorporate the per-nephron division implicitly if they were tuned as total (two-kidney) resistances. This should be verified by checking whether `RBF = (93-4)/(12+26+43)*1000 ≈ 1099 mL/min` matches the expected ~1000 mL/min.

---

### S2. RAAS Pathway — Replaced with Linear Proportional Controller

**R model** (5 ODE state variables + enzymatic kinetics):
```r
d/dt(AngI) = PRA - AngI*(ACE + chymase) - AngI*degradation
d/dt(AngII) = AngI*(ACE + chymase) - AngII*degradation - AngII*AT1_binding - AngII*AT2_binding
d/dt(AT1_bound_AngII) = AngII*AT1_binding - AT1_degradation*AT1_bound_AngII
d/dt(AT2_bound_AngII) = AngII*AT2_binding - AT2_degradation*AT2_bound_AngII
d/dt(PRC) = renin_secretion - PRC*degradation

# Multi-factor renin secretion:
renin_rate = k * nom_PRC * AT1_effect * MD_effect * HCTZ_effect * aldo_effect * RSNA_effect * BB_effect

# AT1 effects on resistances (sigmoid):
AT1_effect = (1 - scale/2) + scale / (1 + exp(-(AT1_bound - nominal) / slope))
```

**Python model** (algebraic):
```python
RAAS_factor = clip(1 - RAAS_gain * 0.005 * dMAP, 0.5, 2.0)
R_EA = R_EA0_eff * RAAS_factor
eta_CD = eta_CD0 * RAAS_factor
```

**Impact**: The Python RAAS is a linear proportional controller keyed on MAP. The R model tracks actual peptide concentrations with pharmacologically relevant binding kinetics. The Python model cannot capture:
- Drug effects (ARB, ACEi, DRI, MRA) that target specific RAAS components
- Nonlinear saturation of receptor binding
- Time-delayed renin secretion responses
- Aldosterone's separate downstream effects

**Fix (full)**: Implement the 5-variable RAAS ODE system. **Fix (minimal)**: Add AT1-mediated sigmoid effects on AA, EA, and CD instead of the linear RAAS_factor.

---

### S3. TGF — Different Signal, Different Math

**R model** (concentration-based sigmoid):
```r
MD_Na_concentration = Na_out_AscLoH / water_out_AscLoH  # Concentration at macula densa
TGF_signal = TGF0 + S_TGF / (1 + exp((setpoint - MD_Na_concentration) / F_md_scale))
# S_TGF = 0.7, F_md_scale = 6, setpoint = 63.29 mEq/L
d/dt(TGF_effect) = C * (TGF_signal - TGF_effect)  # ODE delay
```

**Python model** (flow-based proportional):
```python
MD_Na = Na_filt * (1 - eta_PT) * (1 - eta_LoH)  # Flow, not concentration
TGF_err = (MD_Na - TGF_setpoint) / max(TGF_setpoint, 1e-6)
R_AA_new = R_AA0_eff * (1 + TGF_gain * TGF_err)  # Proportional
R_AA = 0.8 * R_AA + 0.2 * R_AA_new  # Relaxed iteration
```

**Differences**:
1. R model senses **concentration** at macula densa; Python senses **mass flow**
2. R model uses a **sigmoid** transfer function; Python uses **proportional**
3. R model has a **fixed setpoint** (63.29 mEq/L); Python initializes dynamically
4. R model includes an **ODE delay**; Python uses iterative convergence within a single timestep
5. R model's TGF operates on the **afferent arteriole signal multiplier** through the sigmoid saturation; Python directly modifies R_AA

**Fix**: Implement concentration-based TGF:
```python
# After computing LoH outflows:
MD_Na_conc = Na_after_LoH / water_after_LoH  # Need water tracking
TGF0 = 1 - S_TGF / 2  # S_TGF = 0.7
TGF_signal = TGF0 + S_TGF / (1 + np.exp((MD_setpoint - MD_Na_conc) / F_md_scale))
# MD_setpoint = 63.29, F_md_scale = 6
```

---

### S4. Oncotic Pressure — Different Equation

**R model** (Landis-Pappenheimer):
```r
Oncotic_pressure_in = 1.629 * C_protein + 0.2935 * C_protein^2  # g/dL
# C_protein_out calculated from mass balance after filtration
Oncotic_pressure_out = 1.629 * C_protein_out + 0.2935 * C_protein_out^2
oncotic_pressure_avg = (in + out) / 2
```
With albumin sieving affecting protein concentration at exit.

**Python model** (filtration fraction approximation):
```python
FF = GFR / RPF
pi_avg = pi_plasma * (1 + FF / (2*(1-FF)))
```

**Impact**: The Python approximation is a first-order linearization that becomes inaccurate at high filtration fractions. The R model's quadratic Landis-Pappenheimer equation is the standard physiological relationship and differs significantly at FF > 0.3.

**Fix**: Implement Landis-Pappenheimer:
```python
C_protein_in = 7.0  # g/dL (plasma protein)
pi_in = 1.629 * C_protein_in + 0.2935 * C_protein_in**2
C_protein_out = C_protein_in * RPF / (RPF - GFR)
pi_out = 1.629 * C_protein_out + 0.2935 * C_protein_out**2
pi_avg = (pi_in + pi_out) / 2
```

---

### S5. Bowman's Pressure — Static vs. Dynamic

**R model**: `P_bowmans` is a state variable computed from tubular pressure model (Poiseuille flow through compliant tubes). It rises with increased GFR (higher tubular flow → higher intratubular pressure).
```r
d/dt(P_bowmans) = C * 100 * (P_in_pt_s1_mmHg - P_bowmans)
```

**Python model**: `P_Bow = 18.0` mmHg — a constant.

**Impact**: Missing the negative feedback where high GFR raises intratubular pressure, which opposes further filtration. This feedback is physiologically important in hyperfiltration states.

**Fix**: Either implement the tubular pressure model or use a GFR-dependent approximation:
```python
P_Bow = 18.0 + 0.05 * (GFR - 120.0)  # Simple linear proxy
```

---

### S6. Vasopressin / ADH System — Missing

**R model**:
```r
# PI controller for Na concentration
Na_water_controller = Na_controller_gain * (Kp_VP*(Na_conc - ref) + Ki_VP*Na_error)
normalized_vasopressin = 1 + Na_water_controller

# Water intake modulated by vasopressin
water_intake = nom_intake * sigmoid(vasopressin)

# CD water permeability modulated by ADH
ADH_water_permeability = normalized_VP / (0.15 + normalized_VP)
water_reabsorbed_CD = ADH_permeability * water_in * (1 - osmolality_out/osmolality_reference)
```

**Python model**:
```python
water_excr_min = GFR * (1 - frac_water_reabs)  # Fixed 99% reabsorption
```

**Impact**: Water excretion is unregulated — no response to osmolality changes. The kidney cannot concentrate or dilute urine. Volume regulation depends entirely on the crude `frac_water_reabs = 0.99` constant.

**Fix**: Add vasopressin controller:
```python
Na_error_integral += (C_Na - 140.0) * dt
VP = 1.0 + 0.05 * (2.0 * (C_Na - 140.0) + 0.005 * Na_error_integral)
ADH_perm = VP / (0.15 + VP)
water_excr = GFR * (1 - frac_water_reabs_base) * (1 - ADH_perm * osmotic_factor)
```

---

### S7. Aldosterone — Missing

**R model**: Dynamic aldosterone with ODE:
```r
N_als = K_Na_ratio_effect * (AT1_aldo_int + AT1_aldo_slope * AT1_bound_AngII)
d/dt(normalized_aldo) = C * C_aldo_secretion * (N_als - normalized_aldo)
# Effects on DCT and CD reabsorption via sigmoid functions
```

**Python model**: No aldosterone. CD reabsorption modulated only by RAAS_factor (which is a MAP-based proxy).

**Fix**: At minimum, add aldosterone-like modulation of CD that responds to RAAS:
```python
aldo_effect_CD = 0.9 + 0.2 / (1 + np.exp((1 - RAAS_factor) / 0.5))
eta_CD *= aldo_effect_CD
```

---

### S8. Volume Balance — Single vs. Two-Compartment

**R model** (two-compartment):
```r
d/dt(blood_volume_L) = C * (water_intake - urine_flow + Q_water*(Na_conc - IF_Na_conc))
d/dt(IF_volume) = C * Q_water * (IF_Na_conc - Na_conc)
d/dt(sodium_amount) = C * (Na_intake - Na_excretion + Q_Na*(IF_Na - Na_conc))
d/dt(IF_sodium) = C * (Q_Na*(Na_conc - IF_Na) - storage_rate)
d/dt(stored_sodium) = C * storage_rate
```

**Python model** (single compartment):
```python
V_blood += (W_in - water_excr) * dt * 0.33
Na_total += (Na_in - Na_excr) * dt
```

**Impact**: The single-compartment model cannot capture interstitial edema formation, sodium-driven water redistribution, or the buffering role of the interstitial space. The 0.33 Starling factor is a crude approximation of what the two-compartment model computes dynamically.

**Fix**: Add interstitial fluid volume and sodium tracking:
```python
Q_water = 1.0  # L/min transfer coefficient
Q_Na = 1.0     # mEq/min transfer coefficient
dV_blood = dt * (water_intake - urine_flow + Q_water*(C_Na - IF_Na))
dV_IF = dt * Q_water * (IF_Na - C_Na)
dNa_blood = dt * (Na_intake - Na_excretion + Q_Na*(IF_Na - C_Na))
dNa_IF = dt * (Q_Na*(C_Na - IF_Na) - storage_rate)
```

---

### S9. Proximal Tubule — Missing Segmental Detail

**R model**: 3 PT segments (S1, S2, S3) with:
- Individual lengths (5, 5, 4 mm) and geometry
- SGLT2-mediated Na-glucose cotransport in S1/S2
- SGLT1-mediated Na-glucose cotransport in S3 (2:1 Na:glucose)
- Exponential Na reabsorption: `Na_reabs = load * (1 - exp(-rate * length))`
- Osmolality-driven water reabsorption (isosmotic)
- Urea handling with permeability-based reabsorption

**Python model**: Single fractional step: `Na_after_PT = Na_filt * (1 - eta_PT)`

**Fix (minimal)**: Keep the single-fraction approach but use the R model's total PT reabsorption (0.70) and note this is a simplification.

**Fix (full)**: Implement the 3-segment model with SGLT handling.

---

### S10. Loop of Henle — Missing Countercurrent Model

**R model**: Full countercurrent multiplication:
- Descending limb: water-permeable, Na-impermeable
- Ascending limb: Na-reabsorbing, water-impermeable
- Flow-dependent reabsorption: `deltaLoH = min(max, flow_dep * (Na_current - Na_nominal))`
- Exponential osmolality equations

**Python model**: Simple fraction: `Na_after_LoH = Na_after_PT * (1 - eta_LoH)`

**Impact**: The countercurrent mechanism is fundamental to urine concentration ability and the response to loop diuretics. Without it, the model cannot simulate furosemide effects or medullary washout.

---

### S11. Collecting Duct — Missing ADH-Regulated Water Reabsorption

**R model**:
```r
ADH_perm = VP / (0.15 + VP)
osmoles_out_CD = osmoles_in - 2*(Na_in - Na_out)
water_reabsorbed = ADH_perm * osmotic_diuresis_effect * water_in * (1 - osmolality/ref_osmolality)
water_out = water_in - water_reabsorbed
```

**Python model**: Water reabsorption is a global constant (99%), not segment-specific.

---

### S12. Pressure Natriuresis — Wrong Mechanism

**R model**: Based on **postglomerular pressure** with PID controller and segment-specific effects:
```r
PN_signal = max(0.001, 1 + Kp*(P_postglom - RIHP0) + Ki*error + Kd*derivative)
# Separate effects on PT, LoH, DCT, CD:
PN_CD_effect = PN_CD_int + PN_CD_magnitude / (1 + exp(signal - 1))
# Plus RBF effect on CD:
RBF_CD_effect = RBF_CD_int + RBF_CD_scale / (1 + exp((RBF - nom_RBF) / slope))
```

**Python model**: Based on **MAP** with piecewise linear function:
```python
if MAP > setpoint:
    pn = 1 + 0.03 * (MAP - setpoint)
else:
    pn = max(0.3, 1 + 0.015 * (MAP - setpoint))
Na_excr = Na_after_CD * pn
```

**Differences**:
1. R model uses postglomerular pressure; Python uses MAP
2. R model has PID control; Python has proportional only
3. R model applies effects per-segment; Python applies post-CD
4. R model includes RBF effect on CD; Python does not
5. R model uses sigmoid transfer; Python uses piecewise linear

---

### S13. Albumin/Proteinuria Model — Missing

**R model**: Complete albumin sieving with glomerular pressure-dependent sieving coefficient, filtration-reabsorption balance, and Hill function for pressure-induced damage.

**Python model**: None. Emission function uses empirical `UACR = 10 * (P_glom/60)^2 * (1/Kf_scale)^1.5`.

---

### S14. Creatinine Dynamics — Missing

**R model**: `d/dt(serum_creatinine) = C * (synthesis_rate - GFR * dl_ml * Scr/V_blood)`

**Python model**: None. Emission function uses `Scr = 72/GFR`.

---

### S15. Glucose/SGLT2 Handling — Missing

**R model**: Complete glucose handling (SGLT2 in S1/S2, SGLT1 in S3) with SGLT2 inhibitor drug effects, osmotic natriuresis/diuresis, and tubular glucose compensation.

**Python model**: None.

---

### S16. ANP (Atrial Natriuretic Peptide) — Missing

**R model**: LVEDP-driven ANP secretion with effects on CD Na reabsorption and vascular resistances.

**Python model**: None.

---

### S17. RSNA (Renal Sympathetic Nerve Activity) — Missing

**R model**: RSNA effects on preafferent resistance, PT reabsorption, CD reabsorption, renin secretion, and heart rate.

**Python model**: None.

---

### S18. Drug Effects — Missing

**R model**: Complete pharmacological modeling for 8 drug classes (ARB, ACEi, BB, CCB, thiazide, MRA, DRI, SGLT2i).

**Python model**: None. The inflammatory mediator layer provides scaling factors but no drug-specific mechanisms.

---

### S19. Cardiac Hypertrophy — Missing (Intentional)

**R model**: LV hypertrophy via myocyte length/diameter growth ODEs driven by wall stress.

**Python model**: Uses CircAdapt VanOsta2024 for cardiac mechanics (intentional architectural choice). However, there is no equivalent hypertrophy mechanism in the Python implementation.

---

### S20. Glomerular Hypertrophy / Adaptive Kf — Missing

**R model**:
```r
GP_effect_increasing_Kf = (max_increase - disease_Kf) * max(GP/(nom_GP+2) - 1, 0) / T_timescale
d/dt(disease_effects_increasing_Kf) = GP_effect_increasing_Kf
Kf_eff = nom_Kf * (1 + disease_effects_increasing_Kf)
```

**Python model**: `Kf_eff = Kf * Kf_scale * Kf_factor` — no adaptive Kf from glomerular hypertrophy.

---

## MINOR Issues (Parameter Differences That May Be Intentional)

### M1. Water Intake
R model: 2.1 L/day. Python: 2.0 L/day. Small difference, likely intentional rounding.

### M2. Renal Vein Pressure
Both use P_renal_vein = 4 mmHg. However, the R model uses `mean_venous_pressure*0.0075 - 3.16` for RBF calculation (dynamic), while Python uses the constant.

### M3. Hematocrit
Both use Hct = 0.45. Consistent.

### M4. Nephron Count
R model: `baseline_nephrons = 2e6` (total). Python: `N_nephrons = 1e6` per kidney, with `GFR = 2 * N * SNGFR`. Effectively equivalent.

### M5. Plasma Oncotic Pressure
R model: Computed from protein concentration (7 g/dL) → ~28 mmHg via Landis-Pappenheimer.
Python: `pi_plasma = 25.0` mmHg (constant). The R model's equation gives ~28 mmHg at entrance. The Python value is lower.

---

## Prioritized Implementation Roadmap

### Phase 1: Fix Critical Parameters (Minimum Viable Fidelity)
1. **C1**: Fix tubular reabsorption fractions (LoH: 0.80, DT: 0.50, calibrate CD)
2. **C2**: Set Kf = 3.9
3. **C3**: Set MAP_setpoint = 85 or document deviation
4. **C5**: Set R_preAA = 14
5. Verify steady-state sodium balance

### Phase 2: Add Missing Feedback Mechanisms
6. **S3**: Implement sigmoid TGF based on macula densa Na concentration
7. **S4**: Implement Landis-Pappenheimer oncotic pressure
8. **S12**: Fix pressure-natriuresis to use postglomerular pressure
9. **S5**: Add dynamic Bowman's pressure (at least GFR-dependent proxy)
10. **S6**: Add vasopressin-mediated water balance

### Phase 3: Add Missing Subsystems
11. **S2**: Implement RAAS peptide kinetics (5 ODEs)
12. **S7**: Add aldosterone dynamics
13. **S8**: Upgrade to two-compartment volume balance
14. **S1**: Fix vascular resistance structure (per-nephron, sigmoid saturation)

### Phase 4: Full Fidelity (Optional)
15. **S9-S11**: Implement segmented tubular model (PT S1/S2/S3, LoH countercurrent, CD water)
16. **S15**: Add glucose/SGLT2 handling
17. **S13-S14**: Add albumin sieving, creatinine dynamics
18. **S16-S17**: Add ANP, RSNA systems
19. **S18**: Add drug effect pathways
