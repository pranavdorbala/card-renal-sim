# Known Issues

Tracked bugs, design issues, and fixes in the cardiorenal simulator.

---

## Open Issues

### 1. Water Balance Has No Feedback — V_blood Drifts Upward Forever

**Status:** OPEN — fundamental model limitation

**What happens:** V_blood increases monotonically in every simulation, regardless of disease parameters. Non-RL path: 5000 → 5096 → 5190 → 5284 → 5378 over 4 steps. RL path: hits the 8000 mL ceiling clamp within the first monthly step.

**Root cause:** `frac_water_reabs = 0.99` is a fixed constant (`HallowRenalModel`, ~line 631). Water excretion = `GFR × (1 - 0.99)` = 0.6 mL/min = 0.86 L/day. But water intake = 2.0 L/day. Net gain = +1.14 L/day, always positive.

The sodium balance has feedback loops (TGF adjusts GFR, RAAS adjusts excretion). The water balance has none — `frac_water_reabs` never changes in response to volume status. In a real kidney, ADH (vasopressin) modulates water reabsorption: high volume → low ADH → dilute urine → more water excreted. This feedback loop is missing from the Hallow model.

**Impact:**
- Non-RL path: V_blood drifts ~94 mL per 6h step, but runs are short (4-8 steps) so the drift is small (~300-750 mL). Results are still usable.
- RL path: 4 substeps × 180h per monthly step → V_blood hits 8000 ceiling at step 1. Subsequent steps produce NaN (solver crash from extreme volume).
- Synthetic cohort: long trajectories (96 months) will always have V_blood pinned at 8000.

**Possible fixes:**
1. Add ADH feedback: `frac_water_reabs = f(V_blood)` — decrease reabsorption when volume is high
2. Set water intake = water excretion at baseline (force equilibrium): `water_intake = GFR × (1 - frac_water_reabs) × 1440 / 1000`
3. Make `frac_water_reabs` a function of plasma sodium concentration (osmolality-driven)

**Why the non-RL path appeared to work:** Short run duration masked the drift. The drift is ~94 mL/6h, so 4 steps = +375 mL — noticeable but not catastrophic. The RL path exposed it because it integrates for 720h (30 days) per coupling step.

---

### 2. RL Path V_blood = 8000 at Step 1

**Status:** OPEN — blocked by Issue #1

**What happens:** Every `run_coupled_simulation_rl` run shows `V_blood=8000` at step 1, even with a healthy patient (all params 1.0/0.0) and identity policy (alphas=1, residuals=0). Steps 2+ often produce NaN.

**Root cause:** This is a direct consequence of Issue #1. The RL path uses `dt_renal_hours=180.0` with `renal_substeps=4` (720h total per coupling step). The water balance accumulates +1.14 L/day × 30 days = +34 L of ECF, of which 1/3 enters V_blood = +11.3 L. Clamped at 8000.

**Pre-equilibration attempted:** Added 5 × 6h baseline renal updates before the main loop (`run_coupled_simulation_rl` ~line 2183, `rl_env.reset` ~line 133). This initializes TGF and sodium balance, but does NOT fix the water balance drift — V_blood still rises during pre-equilibration and continues rising in the main loop.

**Blocked by:** Fix requires solving Issue #1 (water balance feedback). Pre-equilibration alone is necessary (for TGF/sodium) but not sufficient (water still drifts).

---

## Resolved Issues

### 3. Silent RL Residual Clamping — RL Didn't Know What Value Was Applied

**Status:** FIXED

**What was wrong:** `apply_inflammatory_residuals()` silently clamped corrected factor values with `max()` floors and `np.clip()`. When clamping activated, the RL thought it applied `base + residual` but the simulator used the clamped value. The RL learned wrong gradients.

**Fix:** Moved bounding into the RL's action space:
1. Per-factor residual bounds defined in `config.py` (`residual_min`/`residual_max` arrays)
2. Policy network (`CouplingPolicyHead`) uses tanh to map to per-factor ranges
3. Env (`rl_env._rescale_action`) rescales [-1,1] to per-factor ranges
4. `apply_inflammatory_residuals` now does pure addition — no clipping

**Files changed:** `config.py`, `rl_env.py`, `models/attention_coupling.py`, `train_rl.py`, `tests/test_end_to_end.py`, `tests/test_message_scaling.py`, `cardiorenal_coupling.py`

**Details:** See code_workings.md Section 6.

---

### 4. Wrong Step Numbers in CircAdapt Wrapper Docstring

**Status:** FIXED

**What was wrong:** `CircAdaptWrapper` docstring (~line 94) had arbitrary "Step 1, Step 2, Step 3" labels that didn't match Algorithm 1. Confusing when trying to map code to paper.

**Fix:** Changed to descriptive names: "Apply disease: scales Sf_act (contractility)", "Kidney to Heart: sets blood volume and SVR", "CircAdapt solver: runs to steady state", etc.

**File changed:** `cardiorenal_coupling.py`
