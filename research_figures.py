"""
Publication-quality figures for the cardiorenal digital twin simulator.

Generates a multi-panel figure showing:
  - Healthy vs disease progression comparison
  - Bidirectional message passing (Heart ↔ Kidney)
  - Inflammatory mediator effects on both organs
  - PV loops at early vs late disease

Usage:
    python research_figures.py
"""

import json
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from cardiorenal_coupling import run_coupled_simulation


LOG_FILE = "logs/simulation_runs.jsonl"

def _log_run(source: str, params: dict, hist: dict):
    """Append per-step entries to simulation_runs.jsonl in the same format as agent tools."""
    os.makedirs("logs", exist_ok=True)
    n = len(hist["GFR"])
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        for i in range(n):
            entry = {
                "timestamp": ts,
                "source": source,
                "success": True,
                "params": params,
                "step": i + 1,
                "outputs": {
                    "EF":       round(float(hist["EF"][i]),       2),
                    "MAP":      round(float(hist["MAP"][i]),      2),
                    "CO":       round(float(hist["CO"][i]),       2),
                    "SV":       round(float(hist["SV"][i]),       2),
                    "V_blood":  round(float(hist["V_blood"][i]),  2),
                    "GFR":      round(float(hist["GFR"][i]),      2),
                    "Na_excr":  round(float(hist["Na_excr"][i]),  2),
                    "P_glom":   round(float(hist["P_glom"][i]),   2),
                },
            }
            f.write(json.dumps(entry) + "\n")
    print(f"  Logged {n} steps → {LOG_FILE}")

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

HEALTHY_COLOR = '#2ecc71'
DISEASE_COLOR = '#e74c3c'
INFLAM_COLOR  = '#9b59b6'
MSG_H2K_COLOR = '#3498db'
MSG_K2H_COLOR = '#e67e22'
N_STEPS         = 11520   # 8 years at 6h per coupling step (365.25*8*4 ≈ 11688, use 11520 = 96*120)
STEPS_PER_MONTH = 120     # 30 days × 4 steps/day

# ── Fixed disease parameters (ODE drives progression internally) ──────
# These are CONSTANT inputs — the ODE layer (use_ode=True) accumulates
# fibrosis, endothelial dysfunction, AGE, etc. over time and progressively
# modifies both organ functions on top of these baselines.
SF_DISEASE  = 0.85   # mild HFpEF contractility reduction
KF_DISEASE  = 0.75   # ~25% reduction in filtration coefficient
K1_DISEASE  = 1.30   # myocardial stiffening
INFL        = 0.20   # mild chronic inflammation
DIAB        = 0.40   # moderate diabetes

DISEASE_PARAMS = {
    "Sf_act_scale": SF_DISEASE, "Kf_scale": KF_DISEASE, "k1_scale": K1_DISEASE,
    "inflammation_scale": INFL, "diabetes_scale": DIAB,
}

# ── Run simulations ───────────────────────────────────────────────────
print(f"Running healthy baseline ({N_STEPS} steps = 8 years @ 6h) ...")
healthy = run_coupled_simulation(
    n_steps=N_STEPS, dt_renal_hours=6.0,
    cardiac_schedule=[1.0]*N_STEPS,
    kidney_schedule=[1.0]*N_STEPS,
    stiffness_schedule=[1.0]*N_STEPS,
    inflammation_schedule=[0.0]*N_STEPS,
    diabetes_schedule=[0.0]*N_STEPS,
)
_log_run("research_figures/healthy", {"Sf_act_scale": 1.0, "Kf_scale": 1.0, "k1_scale": 1.0, "inflammation_scale": 0.0, "diabetes_scale": 0.0}, healthy)

print(f"Running disease progression ({N_STEPS} steps = 8 years, ODE-driven) ...")
disease = run_coupled_simulation(
    n_steps=N_STEPS, dt_renal_hours=6.0,
    cardiac_schedule=[SF_DISEASE]*N_STEPS,
    kidney_schedule=[KF_DISEASE]*N_STEPS,
    stiffness_schedule=[K1_DISEASE]*N_STEPS,
    inflammation_schedule=[INFL]*N_STEPS,
    diabetes_schedule=[DIAB]*N_STEPS,
    use_ode=True,
)
_log_run("research_figures/disease", DISEASE_PARAMS, disease)

# x-axis in years
years = np.arange(1, N_STEPS + 1) / (STEPS_PER_MONTH * 12)
steps = years  # alias used throughout figure code below


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1: Healthy vs Disease — Clinical Progression
# ═══════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(14, 6))
fig1.suptitle('Figure 1: Healthy vs Progressive Cardiorenal Disease', fontweight='bold', fontsize=12)

panels = [
    ('EF',      'Ejection Fraction',    '%',       (30, 70),  'EF'),
    ('MAP',     'Mean Arterial Pressure','mmHg',    (70, 100), 'MAP'),
    ('CO',      'Cardiac Output',       'L/min',    (2, 6),    'CO'),
    ('SV',      'Stroke Volume',        'mL',       (30, 80),  'SV'),
    ('GFR',     'Glomerular Filtration', 'mL/min',  (0, 160),  'GFR'),
    ('V_blood', 'Blood Volume',         'mL',       (3500, 6500), 'V_blood'),
    ('Na_excr', 'Na Excretion',         'mEq/day',  (0, 200),  'Na_excr'),
    ('P_glom',  'Glomerular Pressure',  'mmHg',     (30, 70),  'P_glom'),
]

for ax, (key, title, unit, ylim, hist_key) in zip(axes1.flat, panels):
    h_vals = np.array([float(v) for v in healthy[hist_key]])
    d_vals = np.array([float(v) for v in disease[hist_key]])
    ax.plot(steps, h_vals, '-o', color=HEALTHY_COLOR, ms=3, lw=1.5, label='Healthy')
    ax.plot(steps, d_vals, '-s', color=DISEASE_COLOR, ms=3, lw=1.5, label='Disease')
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_xlabel('Time (years)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

fig1.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig('plots/fig1_healthy_vs_disease.png')
print("  Saved: plots/fig1_healthy_vs_disease.png")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2: Bidirectional Message Passing
# ═══════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(14, 8))
fig2.suptitle('Figure 2: Bidirectional Heart ↔ Kidney Message Passing (Disease Patient)',
              fontweight='bold', fontsize=12)

gs = gridspec.GridSpec(3, 4, figure=fig2, hspace=0.45, wspace=0.35)

# Row 0: Input schedules (what drives the disease)
ax_sf = fig2.add_subplot(gs[0, 0])
ax_sf.plot(steps, disease['Sf_scale'], '-o', color='#34495e', ms=3, lw=1.5, label='Sf (contractility)')
ax_sf.plot(steps, disease['effective_Sf'], '--', color=DISEASE_COLOR, lw=1.5, label='Effective Sf')
ax_sf.set_title('Cardiac Schedule')
ax_sf.set_ylabel('Scale factor')
ax_sf.legend(loc='best')
ax_sf.grid(True, alpha=0.3)

ax_kf = fig2.add_subplot(gs[0, 1])
ax_kf.plot(steps, disease['Kf_scale'], '-o', color='#34495e', ms=3, lw=1.5, label='Kf (filtration)')
ax_kf.plot(steps, disease['effective_Kf'], '--', color=DISEASE_COLOR, lw=1.5, label='Effective Kf')
ax_kf.set_title('Renal Schedule')
ax_kf.set_ylabel('Scale factor')
ax_kf.legend(loc='best')
ax_kf.grid(True, alpha=0.3)

ax_k1 = fig2.add_subplot(gs[0, 2])
ax_k1.plot(steps, disease['k1_scale'], '-o', color='#34495e', ms=3, lw=1.5, label='k1 (stiffness)')
ax_k1.plot(steps, disease['effective_k1'], '--', color=DISEASE_COLOR, lw=1.5, label='Effective k1')
ax_k1.set_title('Stiffness Schedule')
ax_k1.set_ylabel('Scale factor')
ax_k1.legend(loc='best')
ax_k1.grid(True, alpha=0.3)

ax_infl = fig2.add_subplot(gs[0, 3])
ax_infl.plot(steps, disease['inflammation_scale'], '-o', color=INFLAM_COLOR, ms=3, lw=1.5, label='Inflammation')
ax_infl.plot(steps, disease['diabetes_scale'], '-s', color='#f39c12', ms=3, lw=1.5, label='Diabetes')
ax_infl.set_title('Disease Drivers')
ax_infl.set_ylabel('Severity (0-1)')
ax_infl.legend(loc='best')
ax_infl.grid(True, alpha=0.3)

# Row 1: Heart → Kidney messages
ax_h2k_map = fig2.add_subplot(gs[1, 0])
ax_h2k_map.plot(steps, [float(v) for v in disease['h2k_MAP']], '-o', color=MSG_H2K_COLOR, ms=3, lw=1.5)
ax_h2k_map.axhline(86.4, color='gray', ls='--', lw=0.8, label='Baseline')
ax_h2k_map.set_title('H→K: MAP')
ax_h2k_map.set_ylabel('mmHg')
ax_h2k_map.legend(loc='best')
ax_h2k_map.grid(True, alpha=0.3)

ax_h2k_co = fig2.add_subplot(gs[1, 1])
ax_h2k_co.plot(steps, [float(v) for v in disease['h2k_CO']], '-o', color=MSG_H2K_COLOR, ms=3, lw=1.5)
ax_h2k_co.axhline(5.1, color='gray', ls='--', lw=0.8, label='Baseline')
ax_h2k_co.set_title('H→K: Cardiac Output')
ax_h2k_co.set_ylabel('L/min')
ax_h2k_co.legend(loc='best')
ax_h2k_co.grid(True, alpha=0.3)

ax_h2k_cvp = fig2.add_subplot(gs[1, 2])
ax_h2k_cvp.plot(steps, [float(v) for v in disease['h2k_Pven']], '-o', color=MSG_H2K_COLOR, ms=3, lw=1.5)
ax_h2k_cvp.set_title('H→K: Venous Pressure')
ax_h2k_cvp.set_ylabel('mmHg')
ax_h2k_cvp.grid(True, alpha=0.3)

# Row 1, col 3: Heart outputs
ax_heart = fig2.add_subplot(gs[1, 3])
ax_heart.plot(steps, [float(v) for v in disease['EF']], '-o', color=DISEASE_COLOR, ms=3, lw=1.5, label='EF (%)')
ax_heart.plot(steps, [float(v) for v in disease['SV']], '-s', color='#1abc9c', ms=3, lw=1.5, label='SV (mL)')
ax_heart.set_title('Cardiac Function')
ax_heart.set_ylabel('EF (%) / SV (mL)')
ax_heart.legend(loc='best')
ax_heart.grid(True, alpha=0.3)

# Row 2: Kidney → Heart messages
ax_k2h_vbl = fig2.add_subplot(gs[2, 0])
ax_k2h_vbl.plot(steps, [float(v) for v in disease['k2h_Vblood']], '-o', color=MSG_K2H_COLOR, ms=3, lw=1.5)
ax_k2h_vbl.axhline(5000, color='gray', ls='--', lw=0.8, label='Baseline')
ax_k2h_vbl.set_title('K→H: Blood Volume')
ax_k2h_vbl.set_ylabel('mL')
ax_k2h_vbl.legend(loc='best')
ax_k2h_vbl.grid(True, alpha=0.3)

ax_k2h_svr = fig2.add_subplot(gs[2, 1])
ax_k2h_svr.plot(steps, [float(v) for v in disease['k2h_SVR']], '-o', color=MSG_K2H_COLOR, ms=3, lw=1.5)
ax_k2h_svr.axhline(1.0, color='gray', ls='--', lw=0.8, label='Baseline')
ax_k2h_svr.set_title('K→H: SVR Ratio')
ax_k2h_svr.set_ylabel('Ratio')
ax_k2h_svr.legend(loc='best')
ax_k2h_svr.grid(True, alpha=0.3)

ax_k2h_gfr = fig2.add_subplot(gs[2, 2])
ax_k2h_gfr.plot(steps, [float(v) for v in disease['k2h_GFR']], '-o', color=MSG_K2H_COLOR, ms=3, lw=1.5)
ax_k2h_gfr.set_title('K→H: GFR')
ax_k2h_gfr.set_ylabel('mL/min')
ax_k2h_gfr.grid(True, alpha=0.3)

# Row 2, col 3: Kidney outputs
ax_kidney = fig2.add_subplot(gs[2, 3])
ax_kidney.plot(steps, [float(v) for v in disease['Na_excr']], '-o', color=MSG_K2H_COLOR, ms=3, lw=1.5, label='Na excr')
ax_kidney.axhline(142, color='gray', ls='--', lw=0.8, label='Na intake')
ax_kidney.set_title('Renal Function')
ax_kidney.set_ylabel('mEq/day')
ax_kidney.legend(loc='best')
ax_kidney.grid(True, alpha=0.3)

for ax in fig2.axes:
    ax.set_xlabel('Time (years)')

fig2.savefig('plots/fig2_message_passing.png')
print("  Saved: plots/fig2_message_passing.png")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3: Inflammatory Mediator Layer
# ═══════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(13, 7))
fig3.suptitle('Figure 3: Inflammatory Mediator Effects on Cardiac & Renal Parameters (Table 1)',
              fontweight='bold', fontsize=12)

# Top row: cardiac modifiers
ax = axes3[0, 0]
ax.plot(steps, disease['Sf_act_factor'], '-o', color='#c0392b', ms=3, lw=1.5, label='Sf_act_factor')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_title('Contractility Factor\n(TNF-α cardiodepression)')
ax.set_ylabel('Multiplicative factor')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes3[0, 1]
ax.plot(steps, disease['passive_k1_factor'], '-o', color='#8e44ad', ms=3, lw=1.5, label='k1_factor')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_title('Diastolic Stiffness Factor\n(AGE cross-linking)')
ax.set_ylabel('Multiplicative factor')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes3[0, 2]
ax.plot(steps, disease['p0_factor'], '-o', color='#2980b9', ms=3, lw=1.5, label='p0_factor (SVR)')
ax.plot(steps, disease['stiffness_factor'], '-s', color='#16a085', ms=3, lw=1.5, label='stiffness_factor')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_title('Vascular Modifiers\n(Endothelial dysfunction)')
ax.set_ylabel('Multiplicative factor')
ax.legend(); ax.grid(True, alpha=0.3)

# Bottom row: renal modifiers
ax = axes3[1, 0]
ax.plot(steps, disease['Kf_factor'], '-o', color='#d35400', ms=3, lw=1.5, label='Kf_factor')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_title('Filtration Coefficient Factor\n(Mesangial expansion)')
ax.set_ylabel('Multiplicative factor')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes3[1, 1]
ax.plot(steps, disease['R_AA_factor'], '-o', color='#27ae60', ms=3, lw=1.5, label='R_AA_factor')
ax.plot(steps, disease['R_EA_factor'], '-s', color='#f39c12', ms=3, lw=1.5, label='R_EA_factor')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_title('Arteriolar Resistance Factors\n(ET-1, AngII)')
ax.set_ylabel('Multiplicative factor')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes3[1, 2]
ax.plot(steps, disease['RAAS_gain_factor'], '-o', color='#2c3e50', ms=3, lw=1.5, label='RAAS gain')
ax.plot(steps, disease['eta_PT_offset'], '-s', color='#e74c3c', ms=3, lw=1.5, label='η_PT offset')
ax2 = ax.twinx()
ax2.plot(steps, disease['MAP_setpoint_offset'], '-^', color='#9b59b6', ms=3, lw=1.5, label='MAP_sp offset')
ax2.set_ylabel('MAP offset (mmHg)', color='#9b59b6')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.axhline(0.0, color='gray', ls=':', lw=0.8)
ax.set_title('RAAS & Tubular Modifiers\n(IL-6, SGLT2, NHE3)')
ax.set_ylabel('Factor / Offset')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='best')
ax.grid(True, alpha=0.3)

for ax in axes3.flat:
    ax.set_xlabel('Time (years)')

fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig('plots/fig3_inflammatory_mediators.png')
print("  Saved: plots/fig3_inflammatory_mediators.png")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 4: PV Loops — Early vs Late Disease
# ═══════════════════════════════════════════════════════════════════════
fig4, (ax_lv, ax_rv) = plt.subplots(1, 2, figsize=(10, 4.5))
fig4.suptitle('Figure 4: Pressure-Volume Loops — Early vs Late Cardiorenal Disease',
              fontweight='bold', fontsize=12)

yr4_idx = 4 * STEPS_PER_MONTH * 12   # step index at year 4
for idx, label, color, ls in [
    (0,        f'Year 0  (Sf={SF_DISEASE:.2f}, Kf={KF_DISEASE:.2f}, ODE start)', HEALTHY_COLOR, '-'),
    (yr4_idx,  f'Year 4  (Sf={SF_DISEASE:.2f}, Kf={KF_DISEASE:.2f}, ODE mid)',   '#f39c12',     '--'),
    (-1,       f'Year 8  (Sf={SF_DISEASE:.2f}, Kf={KF_DISEASE:.2f}, ODE end)',   DISEASE_COLOR, '-'),
]:
    v_lv, p_lv = disease['PV_LV'][idx]
    v_rv, p_rv = disease['PV_RV'][idx]
    ax_lv.plot(v_lv, p_lv, ls, color=color, lw=1.5, label=label)
    ax_rv.plot(v_rv, p_rv, ls, color=color, lw=1.5, label=label)

# Also plot healthy step 1 as reference
v_lv_h, p_lv_h = healthy['PV_LV'][0]
v_rv_h, p_rv_h = healthy['PV_RV'][0]
ax_lv.plot(v_lv_h, p_lv_h, ':', color='gray', lw=1.0, label='Healthy ref')
ax_rv.plot(v_rv_h, p_rv_h, ':', color='gray', lw=1.0, label='Healthy ref')

ax_lv.set_title('Left Ventricle')
ax_lv.set_xlabel('Volume (mL)')
ax_lv.set_ylabel('Pressure (mmHg)')
ax_lv.legend(loc='upper right', fontsize=7)
ax_lv.grid(True, alpha=0.3)

ax_rv.set_title('Right Ventricle')
ax_rv.set_xlabel('Volume (mL)')
ax_rv.set_ylabel('Pressure (mmHg)')
ax_rv.legend(loc='upper right', fontsize=7)
ax_rv.grid(True, alpha=0.3)

fig4.tight_layout(rect=[0, 0, 1, 0.93])
fig4.savefig('plots/fig4_pv_loops.png')
print("  Saved: plots/fig4_pv_loops.png")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 5: Coupling Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(1, 1, figsize=(10, 5))
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 6)
ax5.axis('off')
fig5.suptitle('Figure 5: Bidirectional Cardiorenal Coupling Architecture (Algorithm 1)',
              fontweight='bold', fontsize=12)

# Heart box
heart_box = plt.Rectangle((0.5, 1.5), 3, 3, fill=True, facecolor='#fadbd8',
                           edgecolor='#c0392b', lw=2, zorder=2)
ax5.add_patch(heart_box)
ax5.text(2.0, 4.0, 'HEART', fontsize=12, fontweight='bold', color='#c0392b',
         ha='center', va='center')
ax5.text(2.0, 3.3, 'CircAdapt VanOsta2024', fontsize=8, ha='center', color='#7f8c8d')
ax5.text(2.0, 2.7, 'Sarcomere mechanics\nTriSeg interaction\nClosed-loop circulation',
         fontsize=7, ha='center', color='#34495e')
ax5.text(2.0, 1.8, 'Outputs: MAP, CO, SV, EF, CVP', fontsize=7, ha='center',
         color='#2c3e50', style='italic')

# Kidney box
kidney_box = plt.Rectangle((6.5, 1.5), 3, 3, fill=True, facecolor='#d5f5e3',
                            edgecolor='#27ae60', lw=2, zorder=2)
ax5.add_patch(kidney_box)
ax5.text(8.0, 4.0, 'KIDNEY', fontsize=12, fontweight='bold', color='#27ae60',
         ha='center', va='center')
ax5.text(8.0, 3.3, 'Hallow et al. 2017', fontsize=8, ha='center', color='#7f8c8d')
ax5.text(8.0, 2.7, 'Glomerular hemodynamics\nTGF + RAAS feedback\nVasopressin PI controller',
         fontsize=7, ha='center', color='#34495e')
ax5.text(8.0, 1.8, 'Outputs: GFR, V_blood, Na_excr', fontsize=7, ha='center',
         color='#2c3e50', style='italic')

# Inflammatory layer (top center)
infl_box = plt.Rectangle((3.2, 4.8), 3.6, 1.0, fill=True, facecolor='#f5eef8',
                          edgecolor='#8e44ad', lw=2, zorder=2)
ax5.add_patch(infl_box)
ax5.text(5.0, 5.5, 'INFLAMMATORY LAYER', fontsize=10, fontweight='bold',
         color='#8e44ad', ha='center')
ax5.text(5.0, 5.0, 'inflammation (i) + diabetes (d) → modifier factors',
         fontsize=7, ha='center', color='#34495e')

# H→K arrow (top)
ax5.annotate('', xy=(6.4, 3.5), xytext=(3.6, 3.5),
            arrowprops=dict(arrowstyle='->', color=MSG_H2K_COLOR, lw=2.5))
ax5.text(5.0, 3.75, 'MAP, CO, CVP', fontsize=8, ha='center', fontweight='bold',
         color=MSG_H2K_COLOR)

# K→H arrow (bottom)
ax5.annotate('', xy=(3.6, 2.3), xytext=(6.4, 2.3),
            arrowprops=dict(arrowstyle='->', color=MSG_K2H_COLOR, lw=2.5))
ax5.text(5.0, 2.0, 'V_blood, SVR, GFR', fontsize=8, ha='center', fontweight='bold',
         color=MSG_K2H_COLOR)

# Inflammatory arrows down
ax5.annotate('', xy=(2.0, 4.6), xytext=(3.5, 4.8),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5, ls='--'))
ax5.annotate('', xy=(8.0, 4.6), xytext=(6.5, 4.8),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5, ls='--'))

# Labels on inflammatory arrows
ax5.text(1.8, 4.7, 'Sf↓ k1↑ p0↑', fontsize=6, color='#8e44ad', rotation=25)
ax5.text(7.2, 4.7, 'Kf↓ R_EA↑ η_PT↑', fontsize=6, color='#8e44ad', rotation=-25)

# Deterioration input
ax5.text(5.0, 0.8, 'Input Schedules: Sf(t), Kf(t), k1(t), inflammation(t), diabetes(t)',
         fontsize=8, ha='center', color='#7f8c8d', style='italic')
ax5.annotate('', xy=(5.0, 1.2), xytext=(5.0, 0.95),
            arrowprops=dict(arrowstyle='->', color='#bdc3c7', lw=1.5))

fig5.savefig('plots/fig5_architecture.png')
print("  Saved: plots/fig5_architecture.png")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 6: Input → Effective Parameter Comparison
# ═══════════════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(1, 3, figsize=(12, 3.5))
fig6.suptitle('Figure 6: Direct Deterioration vs Effective Parameters (with Inflammatory Modification)',
              fontweight='bold', fontsize=11)

# Sf
ax = axes6[0]
ax.fill_between(steps, [float(v) for v in disease['Sf_scale']],
                [float(v) for v in disease['effective_Sf']],
                alpha=0.3, color=INFLAM_COLOR, label='Inflammatory penalty')
ax.plot(steps, disease['Sf_scale'], '-o', color='#34495e', ms=3, lw=1.5, label='Input Sf schedule')
ax.plot(steps, disease['effective_Sf'], '-s', color=DISEASE_COLOR, ms=3, lw=1.5, label='Effective Sf (with infl.)')
ax.set_title('Contractility (Sf_act)')
ax.set_ylabel('Scale factor')
ax.set_xlabel('Coupling Step')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Kf
ax = axes6[1]
ax.fill_between(steps, [float(v) for v in disease['Kf_scale']],
                [float(v) for v in disease['effective_Kf']],
                alpha=0.3, color=INFLAM_COLOR, label='Inflammatory penalty')
ax.plot(steps, disease['Kf_scale'], '-o', color='#34495e', ms=3, lw=1.5, label='Input Kf schedule')
ax.plot(steps, disease['effective_Kf'], '-s', color=DISEASE_COLOR, ms=3, lw=1.5, label='Effective Kf (with infl.)')
ax.set_title('Filtration Coefficient (Kf)')
ax.set_ylabel('Scale factor')
ax.set_xlabel('Coupling Step')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# k1
ax = axes6[2]
ax.fill_between(steps, [float(v) for v in disease['k1_scale']],
                [float(v) for v in disease['effective_k1']],
                alpha=0.3, color=INFLAM_COLOR, label='Inflammatory addition')
ax.plot(steps, disease['k1_scale'], '-o', color='#34495e', ms=3, lw=1.5, label='Input k1 schedule')
ax.plot(steps, disease['effective_k1'], '-s', color=DISEASE_COLOR, ms=3, lw=1.5, label='Effective k1 (with infl.)')
ax.set_title('Myocardial Stiffness (k1)')
ax.set_ylabel('Scale factor')
ax.set_xlabel('Coupling Step')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig6.tight_layout(rect=[0, 0, 1, 0.92])
fig6.savefig('plots/fig6_effective_params.png')
print("  Saved: plots/fig6_effective_params.png")


print("\nAll figures saved to plots/")
plt.show()
