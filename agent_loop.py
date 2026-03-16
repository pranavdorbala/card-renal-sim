#!/usr/bin/env python3
"""
Agentic LLM Framework: Cardiorenal Parameter Optimization
===========================================================
Uses LiteLLM (multi-backend: GPT, Gemini, Ollama, Anthropic) to iteratively
find mechanistic model parameters that reproduce a predicted V7 clinical target.

The agent uses tool-calling to run the CircAdapt model, compute errors,
analyze sensitivities, and compare to clinical norms. It produces:
- Optimal disease parameters
- A parameter policy (what changed and why)
- A mechanistic explanation of the disease trajectory

Usage:
    from agent_loop import CardiorenalAgent
    agent = CardiorenalAgent(model="gpt-4o")
    result = agent.solve(v5_data, v7_target, demographics)
"""

import os
import sys
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TUNABLE_PARAMS, LLM_CONFIG, NUMERIC_VAR_NAMES
from agent_tools import TOOL_SCHEMAS, execute_tool


# ═══════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """Result from the agentic optimization."""
    optimal_params: Dict
    parameter_policy: str
    mechanistic_explanation: str
    model_v7_output: Dict
    final_error: float
    n_iterations: int
    n_model_runs: int
    elapsed_seconds: float
    converged: bool
    error_history: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a cardiorenal physiology expert optimizing a coupled heart-kidney simulation model.

## Your Task
Given a patient's Visit 5 (baseline) clinical data and a predicted Visit 7 (6-year follow-up) target,
find the disease progression parameters that make the model reproduce the V7 target.

## Available Disease Parameters
{param_descriptions}

## Key Mechanistic Relationships
- **HFrEF pathway**: Low Sf_act_scale → reduced contractility → low LVEF, low CO → renal hypoperfusion → low GFR
- **HFpEF pathway**: High diabetes_scale or k1_scale → increased diastolic stiffness → preserved EF but elevated E/e', high LVEDP
- **CKD pathway**: Low Kf_scale → reduced nephrons → low GFR, high creatinine, elevated UACR
- **Inflammation**: High inflammation_scale → ↑SVR, ↑arterial stiffness, ↓Sf_act, ↓Kf (multi-organ effect)
- **Cardiorenal syndrome (CRS)**: Heart failure → renal hypoperfusion → fluid retention → worsens heart failure
- **RAAS**: High RAAS_gain → more reactive to MAP changes → affects efferent arteriole tone and CD reabsorption

## Strategy
1. First, analyze the V7 target to identify the dominant phenotype (HFrEF, HFpEF, CKD, CRS, etc.)
2. Set initial parameters based on the phenotype pattern
3. Run the model and compute error
4. Use sensitivity analysis on the worst-matching variables to identify which parameters to adjust
5. Iteratively refine parameters, focusing on the largest error contributors
6. When error is below threshold, provide your final explanation

## Important
- Start with the V5 baseline parameters if provided, then adjust toward V7
- Focus on the clinically important variables first (LVEF, eGFR, E/e', MAP, NTproBNP)
- Consider bidirectional cardiorenal feedback when adjusting parameters
- Explain your reasoning at each step
"""


def _build_system_prompt() -> str:
    """Build system prompt with parameter descriptions."""
    param_lines = []
    for name, info in TUNABLE_PARAMS.items():
        param_lines.append(
            f"- **{name}** (range {info['range'][0]}-{info['range'][1]}, "
            f"default {info['default']}): {info['desc']}"
        )
    return SYSTEM_PROMPT.format(param_descriptions='\n'.join(param_lines))


def _build_initial_prompt(v5_data: Dict, v7_target: Dict, demographics: Dict) -> str:
    """Build the initial user prompt with V5 data and V7 target."""
    # Select key variables for display
    key_vars = [
        'LVEF_pct', 'MAP_mmHg', 'SBP_mmHg', 'DBP_mmHg', 'CO_Lmin',
        'GFR_mL_min', 'eGFR_mL_min_173m2', 'serum_creatinine_mg_dL',
        'E_e_prime_avg', 'LVEDV_mL', 'LVESV_mL', 'GLS_pct',
        'NTproBNP_pg_mL', 'UACR_mg_g', 'RBF_mL_min',
        'LV_mass_g', 'LA_volume_mL', 'PASP_mmHg',
    ]

    v5_lines = []
    v7_lines = []
    for vn in key_vars:
        if vn in v5_data:
            v5_lines.append(f"  {vn}: {v5_data[vn]:.2f}")
        if vn in v7_target:
            v7_lines.append(f"  {vn}: {v7_target[vn]:.2f}")

    return f"""## Patient Demographics
- Age at V5: {demographics.get('age', 75):.0f}, Sex: {demographics.get('sex', 'M')}
- BSA: {demographics.get('BSA', 1.9):.2f} m², Height: {demographics.get('height_m', 1.75):.2f} m

## Visit 5 (Baseline) Key Variables
{chr(10).join(v5_lines)}

## Visit 7 (Target) Key Variables
{chr(10).join(v7_lines)}

## Your Goal
Find disease parameters that make the model output match the V7 target.
Start by analyzing the phenotype, then use the tools to run the model and optimize.
Begin with run_circadapt_model using your best initial guess, then compute_error to see how far off you are.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Nelder-Mead Fallback
# ═══════════════════════════════════════════════════════════════════════════

def _nelder_mead_fallback(
    v7_target: Dict,
    demographics: Dict,
    initial_params: Dict,
    max_evals: int = 100,
) -> Dict:
    """
    Scipy Nelder-Mead optimization as fallback when LLM doesn't converge.
    Returns optimized parameter dict.
    """
    from scipy.optimize import minimize
    from agent_tools import run_circadapt_model, compute_error

    param_names = list(TUNABLE_PARAMS.keys())
    x0 = np.array([initial_params.get(p, TUNABLE_PARAMS[p]['default']) for p in param_names])
    bounds = [TUNABLE_PARAMS[p]['range'] for p in param_names]

    def objective(x):
        params = {p: float(np.clip(x[i], *bounds[i])) for i, p in enumerate(param_names)}
        result = run_circadapt_model(**params, **demographics)
        if 'error' in result:
            return 1e6
        err = compute_error(result, v7_target)
        return err['aggregate_error']

    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxfev': max_evals, 'xatol': 0.01, 'fatol': 0.001},
    )

    return {p: float(np.clip(result.x[i], *bounds[i])) for i, p in enumerate(param_names)}


# ═══════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════

class CardiorenalAgent:
    """
    Agentic LLM framework for cardiorenal parameter optimization.
    Uses LiteLLM for multi-backend support (OpenAI, Gemini, Ollama, Anthropic).
    """

    def __init__(
        self,
        model: str = LLM_CONFIG['model'],
        max_iterations: int = LLM_CONFIG['max_iterations'],
        convergence_threshold: float = LLM_CONFIG['convergence_threshold'],
        temperature: float = LLM_CONFIG['temperature'],
        verbose: bool = True,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temperature = temperature
        self.verbose = verbose

    def solve(
        self,
        v5_data: Dict,
        v7_target: Dict,
        demographics: Dict,
    ) -> AgentResult:
        """
        Run the agentic optimization loop.

        Parameters
        ----------
        v5_data : dict
            V5 ARIC variables (from NN input or model evaluation)
        v7_target : dict
            V7 ARIC variables (from NN prediction or real data)
        demographics : dict
            age, sex, BSA, height_m

        Returns
        -------
        AgentResult with optimal params, policy, explanation, etc.
        """
        import litellm

        t0 = time.time()

        # Build messages
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_initial_prompt(v5_data, v7_target, demographics)},
        ]

        best_error = float('inf')
        best_params = {p: TUNABLE_PARAMS[p]['default'] for p in TUNABLE_PARAMS}
        best_output = {}
        error_history = []
        n_model_runs = 0
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"  Agent iteration {iteration}/{self.max_iterations}...")

            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=LLM_CONFIG.get('max_tokens', 4096),
                )
            except Exception as e:
                if self.verbose:
                    print(f"  LLM API error: {e}")
                break

            msg = response.choices[0].message
            # Append assistant message
            messages.append(msg.model_dump())

            # Check for tool calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    if self.verbose:
                        print(f"    Tool: {fn_name}({', '.join(f'{k}={v}' for k, v in list(fn_args.items())[:3])}...)")

                    # Execute tool
                    tool_result = execute_tool(fn_name, fn_args)
                    n_model_runs += 1 if fn_name in ('run_circadapt_model', 'get_sensitivity') else 0

                    # Track error
                    if fn_name == 'compute_error':
                        try:
                            err_data = json.loads(tool_result)
                            agg_err = err_data.get('aggregate_error', float('inf'))
                            error_history.append(agg_err)
                            if agg_err < best_error:
                                best_error = agg_err
                                # Extract params from the most recent model run
                                if 'model_output' in fn_args:
                                    mo = fn_args['model_output']
                                    if 'params_used' in mo:
                                        best_params = mo['params_used']
                                    best_output = mo
                        except (json.JSONDecodeError, KeyError):
                            pass

                    if fn_name == 'run_circadapt_model':
                        try:
                            run_data = json.loads(tool_result)
                            if 'params_used' in run_data:
                                # Store as candidate best
                                best_output = run_data
                                best_params = run_data['params_used']
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Truncate long results
                    if len(tool_result) > 4000:
                        tool_result_msg = tool_result[:4000] + '... (truncated)'
                    else:
                        tool_result_msg = tool_result

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_msg,
                    })

            # Check convergence
            if best_error <= self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged! Error={best_error:.4f}")
                converged = True

                # Ask for final explanation
                messages.append({
                    "role": "user",
                    "content": (
                        f"The model has converged with error={best_error:.4f}. "
                        f"Please provide:\n"
                        f"1. A PARAMETER POLICY: which parameters changed from baseline and by how much\n"
                        f"2. A MECHANISTIC EXPLANATION: what disease processes explain the V5→V7 transition\n"
                        f"Format your response with clear sections."
                    ),
                })

                try:
                    final_response = litellm.completion(
                        model=self.model, messages=messages,
                        temperature=0.3, max_tokens=2048,
                    )
                    explanation_text = final_response.choices[0].message.content or ""
                except Exception:
                    explanation_text = "Converged but failed to generate explanation."

                # Parse policy and explanation
                policy, explanation = _parse_explanation(explanation_text)
                break

            # Check for stagnation
            if len(error_history) >= 3 and iteration >= 10:
                recent = error_history[-3:]
                if max(recent) - min(recent) < 0.001:
                    if self.verbose:
                        print(f"  Stagnated at error={best_error:.4f}, falling back to Nelder-Mead...")
                    best_params = _nelder_mead_fallback(
                        v7_target, demographics, best_params, max_evals=80
                    )
                    # Run final evaluation with optimized params
                    from agent_tools import run_circadapt_model, compute_error
                    best_output = run_circadapt_model(**best_params, **demographics)
                    err = compute_error(best_output, v7_target)
                    best_error = err['aggregate_error']
                    converged = best_error <= self.convergence_threshold
                    break

            # If no tool calls (pure text response), check if LLM is done
            if not msg.tool_calls:
                content = msg.content or ""
                if 'parameter policy' in content.lower() or 'mechanistic explanation' in content.lower():
                    policy, explanation = _parse_explanation(content)
                    break
        else:
            # Max iterations reached
            if self.verbose:
                print(f"  Max iterations reached. Best error={best_error:.4f}")
            policy = f"Optimization did not converge (error={best_error:.4f})"
            explanation = "Max iterations reached without convergence."

        elapsed = time.time() - t0

        # Ensure we have policy/explanation if not set
        if 'policy' not in dir():
            policy = f"Parameters: {json.dumps(best_params, indent=2)}"
        if 'explanation' not in dir():
            explanation = "No mechanistic explanation generated."

        return AgentResult(
            optimal_params=best_params,
            parameter_policy=policy,
            mechanistic_explanation=explanation,
            model_v7_output=best_output,
            final_error=best_error,
            n_iterations=min(iteration, self.max_iterations) if 'iteration' in dir() else 0,
            n_model_runs=n_model_runs,
            elapsed_seconds=elapsed,
            converged=converged,
            error_history=error_history,
        )


def _parse_explanation(text: str):
    """Parse LLM text into parameter policy and mechanistic explanation."""
    text = text.strip()

    policy = ""
    explanation = ""

    # Try to split by sections
    lower = text.lower()
    policy_idx = lower.find('parameter policy')
    mech_idx = lower.find('mechanistic explanation')

    if policy_idx >= 0 and mech_idx >= 0:
        if policy_idx < mech_idx:
            policy = text[policy_idx:mech_idx].strip()
            explanation = text[mech_idx:].strip()
        else:
            explanation = text[mech_idx:policy_idx].strip()
            policy = text[policy_idx:].strip()
    elif policy_idx >= 0:
        policy = text[policy_idx:].strip()
        explanation = text[:policy_idx].strip()
    elif mech_idx >= 0:
        explanation = text[mech_idx:].strip()
        policy = text[:mech_idx].strip()
    else:
        # No clear sections, use the whole text
        policy = text
        explanation = text

    return policy, explanation


# ═══════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Quick test with a synthetic patient."""
    import argparse
    parser = argparse.ArgumentParser(description='Test agentic cardiorenal optimizer')
    parser.add_argument('--model', type=str, default=LLM_CONFIG['model'],
                        help='LiteLLM model string (e.g., gpt-4o, ollama/llama3.1)')
    parser.add_argument('--max_iter', type=int, default=LLM_CONFIG['max_iterations'])
    args = parser.parse_args()

    from synthetic_cohort import evaluate_patient_state

    # Generate a test patient (mild HFrEF + early CKD)
    v5_params = {
        'Sf_act_scale': 0.85, 'Kf_scale': 0.9, 'inflammation_scale': 0.1,
        'diabetes_scale': 0.0, 'k1_scale': 1.0, 'RAAS_gain': 1.5,
        'TGF_gain': 2.0, 'na_intake': 150.0,
    }
    v7_params = {
        'Sf_act_scale': 0.65, 'Kf_scale': 0.6, 'inflammation_scale': 0.25,
        'diabetes_scale': 0.1, 'k1_scale': 1.2, 'RAAS_gain': 1.8,
        'TGF_gain': 2.0, 'na_intake': 160.0,
    }
    demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}

    print("Generating V5 and V7 data...")
    v5_data = evaluate_patient_state(v5_params, demographics)
    v7_target = evaluate_patient_state(v7_params, {'age': 78, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75})

    if v5_data is None or v7_target is None:
        print("ERROR: Model evaluation failed")
        return

    print(f"V5: LVEF={v5_data['LVEF_pct']:.1f}%, GFR={v5_data['GFR_mL_min']:.1f}")
    print(f"V7 target: LVEF={v7_target['LVEF_pct']:.1f}%, GFR={v7_target['GFR_mL_min']:.1f}")

    agent = CardiorenalAgent(model=args.model, max_iterations=args.max_iter)
    result = agent.solve(v5_data, v7_target, demographics)

    print(f"\n{'='*60}")
    print(f"  Result: converged={result.converged}, error={result.final_error:.4f}")
    print(f"  Iterations: {result.n_iterations}, Model runs: {result.n_model_runs}")
    print(f"  Time: {result.elapsed_seconds:.1f}s")
    print(f"{'='*60}")
    print(f"\nParameter Policy:\n{result.parameter_policy[:500]}")
    print(f"\nMechanistic Explanation:\n{result.mechanistic_explanation[:500]}")


if __name__ == '__main__':
    main()
