#!/usr/bin/env python3
"""
Agent Tools: LLM-callable functions wrapping the CircAdapt + Hallow model.
===========================================================================
Provides four tools for the agentic LLM framework to explore the cardiorenal
parameter space: run the model, compute error, get sensitivity, and compare
to clinical norms.

Each tool is defined as a function + an OpenAI function-calling schema
compatible with LiteLLM.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TUNABLE_PARAMS, ARIC_VARIABLES, NUMERIC_VAR_NAMES,
    CLINICAL_THRESHOLDS, N_FEATURES,
)
from synthetic_cohort import evaluate_patient_state


# ═══════════════════════════════════════════════════════════════════════════
# Tool 1: Run CircAdapt Model
# ═══════════════════════════════════════════════════════════════════════════

def run_circadapt_model(
    Sf_act_scale: float = 1.0,
    Kf_scale: float = 1.0,
    inflammation_scale: float = 0.0,
    diabetes_scale: float = 0.0,
    k1_scale: float = 1.0,
    RAAS_gain: float = 1.5,
    TGF_gain: float = 2.0,
    na_intake: float = 150.0,
    age: float = 75.0,
    sex: str = 'M',
    BSA: float = 1.9,
    height_m: float = 1.75,
) -> Dict:
    """
    Run the coupled CircAdapt + Hallow model and return 113 ARIC variables.
    ~0.5s per call.
    """
    params = {
        'Sf_act_scale': float(np.clip(Sf_act_scale, *TUNABLE_PARAMS['Sf_act_scale']['range'])),
        'Kf_scale': float(np.clip(Kf_scale, *TUNABLE_PARAMS['Kf_scale']['range'])),
        'inflammation_scale': float(np.clip(inflammation_scale, *TUNABLE_PARAMS['inflammation_scale']['range'])),
        'diabetes_scale': float(np.clip(diabetes_scale, *TUNABLE_PARAMS['diabetes_scale']['range'])),
        'k1_scale': float(np.clip(k1_scale, *TUNABLE_PARAMS['k1_scale']['range'])),
        'RAAS_gain': float(np.clip(RAAS_gain, *TUNABLE_PARAMS['RAAS_gain']['range'])),
        'TGF_gain': float(np.clip(TGF_gain, *TUNABLE_PARAMS['TGF_gain']['range'])),
        'na_intake': float(np.clip(na_intake, *TUNABLE_PARAMS['na_intake']['range'])),
    }
    demographics = {
        'age': float(age),
        'sex': str(sex),
        'BSA': float(BSA),
        'height_m': float(height_m),
    }

    result = evaluate_patient_state(params, demographics)
    if result is None:
        return {'error': 'Model evaluation failed', 'params': params}

    # Round for readability
    output = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            output[k] = round(float(v), 3)
        else:
            output[k] = v
    output['params_used'] = params
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Tool 2: Compute Error vs Target
# ═══════════════════════════════════════════════════════════════════════════

def compute_error(model_output: Dict, target: Dict) -> Dict:
    """
    Compute weighted per-variable error between model output and target.
    Returns aggregate error, per-variable errors, and worst variables.
    """
    errors = {}
    weighted_sq_sum = 0.0
    weight_sum = 0.0
    direction_mismatches = []

    for var_name in NUMERIC_VAR_NAMES:
        if var_name not in model_output or var_name not in target:
            continue

        model_val = float(model_output[var_name])
        target_val = float(target[var_name])
        normal = ARIC_VARIABLES.get(var_name, {}).get('normal', (0, 1))
        weight = ARIC_VARIABLES.get(var_name, {}).get('weight', 0.5)
        normal_range = max(abs(normal[1] - normal[0]), 1e-6)

        # Normalized absolute error
        abs_err = abs(model_val - target_val) / normal_range
        errors[var_name] = {
            'model': round(model_val, 3),
            'target': round(target_val, 3),
            'abs_error': round(abs_err, 4),
            'weight': weight,
        }
        weighted_sq_sum += abs_err ** 2 * weight
        weight_sum += weight

    aggregate = (weighted_sq_sum / max(weight_sum, 1e-6)) ** 0.5

    # Worst 5 variables by weighted error
    ranked = sorted(errors.items(), key=lambda x: x[1]['abs_error'] * x[1]['weight'], reverse=True)
    worst_5 = {k: v for k, v in ranked[:5]}

    return {
        'aggregate_error': round(float(aggregate), 4),
        'n_variables_compared': len(errors),
        'worst_5_variables': worst_5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool 3: Parameter Sensitivity (Jacobian)
# ═══════════════════════════════════════════════════════════════════════════

def get_sensitivity(
    base_params: Dict,
    param_name: str,
    delta: float = 0.05,
    age: float = 75.0,
    sex: str = 'M',
    BSA: float = 1.9,
    height_m: float = 1.75,
) -> Dict:
    """
    Finite-difference sensitivity: d(ARIC_var)/d(param).
    Runs the model twice (~1s total) with param ± delta.
    """
    if param_name not in TUNABLE_PARAMS:
        return {'error': f'Unknown parameter: {param_name}',
                'valid_params': list(TUNABLE_PARAMS.keys())}

    demographics = {'age': age, 'sex': sex, 'BSA': BSA, 'height_m': height_m}
    prange = TUNABLE_PARAMS[param_name]['range']
    base_val = base_params.get(param_name, TUNABLE_PARAMS[param_name]['default'])

    # Perturb
    val_lo = max(base_val - delta, prange[0])
    val_hi = min(base_val + delta, prange[1])
    actual_delta = val_hi - val_lo
    if actual_delta < 1e-8:
        return {'error': f'Cannot perturb {param_name} (at boundary)'}

    params_lo = dict(base_params)
    params_lo[param_name] = val_lo
    params_hi = dict(base_params)
    params_hi[param_name] = val_hi

    result_lo = evaluate_patient_state(params_lo, demographics)
    result_hi = evaluate_patient_state(params_hi, demographics)

    if result_lo is None or result_hi is None:
        return {'error': 'Model evaluation failed during sensitivity analysis'}

    # Compute sensitivities
    sensitivities = {}
    for var_name in NUMERIC_VAR_NAMES:
        if var_name in result_lo and var_name in result_hi:
            lo_val = float(result_lo[var_name])
            hi_val = float(result_hi[var_name])
            deriv = (hi_val - lo_val) / actual_delta
            sensitivities[var_name] = round(deriv, 4)

    # Rank by absolute sensitivity
    ranked = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    top_10 = {k: v for k, v in ranked[:10]}

    return {
        'parameter': param_name,
        'base_value': round(base_val, 4),
        'delta': round(actual_delta, 4),
        'top_10_sensitivities': top_10,
        'all_sensitivities': sensitivities,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool 4: Clinical Classification
# ═══════════════════════════════════════════════════════════════════════════

def compare_to_clinical_norms(variables: Dict) -> Dict:
    """
    Classify each variable as normal/borderline/abnormal per clinical thresholds.
    Also applies threshold-based disease staging.
    """
    classifications = {}

    for var_name, thresholds in CLINICAL_THRESHOLDS.items():
        if var_name not in variables:
            continue
        val = float(variables[var_name])
        label = 'unknown'
        for threshold_val, threshold_label in thresholds:
            if val >= threshold_val:
                label = threshold_label
                break
        classifications[var_name] = {
            'value': round(val, 2),
            'classification': label,
        }

    # Overall normal-range check for all variables
    out_of_range = {}
    for var_name in NUMERIC_VAR_NAMES:
        if var_name not in variables:
            continue
        normal = ARIC_VARIABLES.get(var_name, {}).get('normal')
        if normal is None:
            continue
        val = float(variables[var_name])
        if val < normal[0]:
            out_of_range[var_name] = {
                'value': round(val, 2),
                'normal_range': normal,
                'status': 'below_normal',
            }
        elif val > normal[1]:
            out_of_range[var_name] = {
                'value': round(val, 2),
                'normal_range': normal,
                'status': 'above_normal',
            }

    return {
        'disease_staging': classifications,
        'out_of_range_count': len(out_of_range),
        'out_of_range': out_of_range,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LiteLLM Tool Schemas (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_circadapt_model",
            "description": (
                "Run the coupled CircAdapt heart + Hallow renal model with given "
                "disease parameters. Returns 113 ARIC-compatible clinical variables "
                "including echocardiographic measurements, hemodynamics, and renal markers. "
                "Takes ~0.5 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "Sf_act_scale": {
                        "type": "number",
                        "description": "Active fiber stress scale (0.2-1.0). Lower = weaker contraction (HFrEF). Default 1.0.",
                    },
                    "Kf_scale": {
                        "type": "number",
                        "description": "Glomerular ultrafiltration coefficient scale (0.05-1.0). Lower = nephron loss (CKD). Default 1.0.",
                    },
                    "inflammation_scale": {
                        "type": "number",
                        "description": "Systemic inflammation (0.0-1.0). Increases SVR, arterial stiffness, reduces Sf_act and Kf. Default 0.0.",
                    },
                    "diabetes_scale": {
                        "type": "number",
                        "description": "Diabetes burden (0.0-1.0). Increases diastolic stiffness (k1→HFpEF), arterial stiffness, affects Kf. Default 0.0.",
                    },
                    "k1_scale": {
                        "type": "number",
                        "description": "Passive myocardial stiffness scale (1.0-3.0). Higher = more diastolic dysfunction (HFpEF). Default 1.0.",
                    },
                    "RAAS_gain": {
                        "type": "number",
                        "description": "RAAS sensitivity (0.5-3.0). Higher = more reactive to MAP drops. Default 1.5.",
                    },
                    "TGF_gain": {
                        "type": "number",
                        "description": "Tubuloglomerular feedback gain (1.0-4.0). Default 2.0.",
                    },
                    "na_intake": {
                        "type": "number",
                        "description": "Dietary sodium intake in mEq/day (50-300). Default 150.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_error",
            "description": (
                "Compute weighted error between model output and a target clinical state. "
                "Returns aggregate error (0=perfect match), worst 5 variables, and per-variable errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_output": {
                        "type": "object",
                        "description": "Dict of ARIC variable names → values from run_circadapt_model.",
                    },
                    "target": {
                        "type": "object",
                        "description": "Dict of ARIC variable names → target values to match.",
                    },
                },
                "required": ["model_output", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensitivity",
            "description": (
                "Compute finite-difference sensitivity of all ARIC variables to a single "
                "disease parameter. Returns d(variable)/d(parameter) for the top 10 most "
                "sensitive variables. Takes ~1 second (two model runs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_params": {
                        "type": "object",
                        "description": "Current disease parameter values.",
                    },
                    "param_name": {
                        "type": "string",
                        "description": "Name of parameter to perturb. One of: Sf_act_scale, Kf_scale, inflammation_scale, diabetes_scale, k1_scale, RAAS_gain, TGF_gain, na_intake.",
                    },
                    "delta": {
                        "type": "number",
                        "description": "Perturbation size. Default 0.05.",
                    },
                },
                "required": ["base_params", "param_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_clinical_norms",
            "description": (
                "Classify model output against clinical thresholds. Returns disease staging "
                "(HFpEF/HFrEF, CKD stage, filling pressure grade, albuminuria) and a list "
                "of variables outside their normal range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "object",
                        "description": "Dict of ARIC variable names → values.",
                    },
                },
                "required": ["variables"],
            },
        },
    },
]

# Map function names → callables
TOOL_FUNCTIONS = {
    'run_circadapt_model': run_circadapt_model,
    'compute_error': compute_error,
    'get_sensitivity': get_sensitivity,
    'compare_to_clinical_norms': compare_to_clinical_norms,
}


def execute_tool(name: str, arguments: Dict) -> str:
    """Execute a tool by name with given arguments, return JSON string."""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({'error': f'Unknown tool: {name}'})
    try:
        result = TOOL_FUNCTIONS[name](**arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({'error': str(e)})
