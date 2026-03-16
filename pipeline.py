#!/usr/bin/env python3
"""
End-to-End Pipeline: V5 → NN Prediction → Agent Optimization → Result
=======================================================================
Integrates the neural network (V5→V7 prediction) with the agentic LLM
framework to produce optimal disease parameters, a parameter policy,
and a mechanistic explanation.

Usage:
    # Single patient (JSON)
    python pipeline.py --v5 '{"LVEF_pct": 58, "GFR_mL_min": 90, ...}'

    # Single patient from model params
    python pipeline.py --params '{"Sf_act_scale": 0.85, "Kf_scale": 0.9}'

    # Batch from CSV
    python pipeline.py --v5_csv patients.csv --model gpt-4o --workers 4
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TUNABLE_PARAMS, NUMERIC_VAR_NAMES, LLM_CONFIG, NN_DEFAULTS


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class CardiorenalPipeline:
    """
    End-to-end pipeline: V5 input → NN prediction → Agent optimization → result.

    Components:
    1. Neural network: predicts V7 clinical state from V5 baseline (~1ms)
    2. Agent: finds mechanistic parameters reproducing V7 target (~30-60s)
    """

    def __init__(
        self,
        nn_model_path: str = 'models/v5_to_v7_best.pt',
        llm_model: str = LLM_CONFIG['model'],
        max_iterations: int = LLM_CONFIG['max_iterations'],
        verbose: bool = True,
    ):
        base = os.path.dirname(os.path.abspath(__file__))
        nn_path = os.path.join(base, nn_model_path) if not os.path.isabs(nn_model_path) else nn_model_path

        # Load NN
        if os.path.exists(nn_path):
            from train_nn import load_trained_model, predict
            self.nn_model, self.nn_ckpt = load_trained_model(nn_path)
            self.nn_var_names = self.nn_ckpt['var_names']
            self.predict_fn = predict
            self.has_nn = True
            if verbose:
                print(f"Loaded NN from {nn_path} ({len(self.nn_var_names)} features)")
        else:
            self.has_nn = False
            if verbose:
                print(f"NN model not found at {nn_path}. "
                      f"Pipeline will work without NN (requires explicit V7 target).")

        # Agent
        from agent_loop import CardiorenalAgent
        self.agent = CardiorenalAgent(
            model=llm_model,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self.verbose = verbose

    def predict_v7(self, v5_data: Dict) -> Dict:
        """Use the NN to predict V7 from V5 data."""
        if not self.has_nn:
            raise RuntimeError("No NN model loaded. Train one first with train_nn.py.")

        v5_vec = np.array(
            [float(v5_data.get(k, 0.0)) for k in self.nn_var_names],
            dtype=np.float32,
        )
        v7_vec = self.predict_fn(self.nn_model, v5_vec)

        return {k: float(v7_vec[i]) for i, k in enumerate(self.nn_var_names)}

    def predict_and_explain(
        self,
        v5_data: Dict,
        demographics: Dict,
        v7_target: Optional[Dict] = None,
    ) -> Dict:
        """
        Full pipeline: V5 → NN prediction → Agent optimization → result.

        Parameters
        ----------
        v5_data : dict
            V5 ARIC variables
        demographics : dict
            age, sex, BSA, height_m
        v7_target : dict, optional
            If provided, use this instead of NN prediction

        Returns
        -------
        dict with keys:
            v5_input, v7_nn_prediction, v7_target_used,
            optimal_params, parameter_policy, mechanistic_explanation,
            model_v7_output, prediction_error, agent_converged,
            timing
        """
        t0 = time.time()

        # Step 1: NN prediction (if no explicit target)
        if v7_target is None:
            v7_nn = self.predict_v7(v5_data)
            v7_target_used = v7_nn
        else:
            v7_nn = None
            v7_target_used = v7_target

        t_nn = time.time() - t0

        # Step 2: Agent optimization
        t_agent_start = time.time()
        agent_result = self.agent.solve(v5_data, v7_target_used, demographics)
        t_agent = time.time() - t_agent_start

        return {
            'v5_input': v5_data,
            'v7_nn_prediction': v7_nn,
            'v7_target_used': v7_target_used,
            'optimal_params': agent_result.optimal_params,
            'parameter_policy': agent_result.parameter_policy,
            'mechanistic_explanation': agent_result.mechanistic_explanation,
            'model_v7_output': agent_result.model_v7_output,
            'prediction_error': agent_result.final_error,
            'agent_converged': agent_result.converged,
            'error_history': agent_result.error_history,
            'timing': {
                'nn_seconds': round(t_nn, 3),
                'agent_seconds': round(t_agent, 1),
                'total_seconds': round(time.time() - t0, 1),
                'n_model_runs': agent_result.n_model_runs,
                'n_iterations': agent_result.n_iterations,
            },
        }

    def batch_predict(
        self,
        v5_batch: List[Dict],
        demographics_batch: List[Dict],
        v7_targets: Optional[List[Dict]] = None,
        n_workers: int = 4,
    ) -> List[Dict]:
        """
        Batch processing. NN is batched; agent is parallelized with threads.
        """
        n = len(v5_batch)
        if v7_targets is None:
            v7_targets = [None] * n

        def process_one(i):
            return self.predict_and_explain(
                v5_batch[i], demographics_batch[i], v7_targets[i]
            )

        if n_workers <= 1:
            results = [process_one(i) for i in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(process_one, range(n)))

        return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Cardiorenal V5→V7 Pipeline')
    parser.add_argument('--v5', type=str, default=None,
                        help='V5 data as JSON string')
    parser.add_argument('--params', type=str, default=None,
                        help='Disease params as JSON (will run model to get V5 data)')
    parser.add_argument('--v7_target', type=str, default=None,
                        help='Optional V7 target as JSON (skips NN prediction)')
    parser.add_argument('--v5_csv', type=str, default=None,
                        help='CSV file with V5 data for batch processing')
    parser.add_argument('--model', type=str, default=LLM_CONFIG['model'],
                        help='LiteLLM model string')
    parser.add_argument('--nn_path', type=str, default='models/v5_to_v7_best.pt')
    parser.add_argument('--max_iter', type=int, default=LLM_CONFIG['max_iterations'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    # Single patient from disease params
    if args.params:
        from synthetic_cohort import evaluate_patient_state
        params = json.loads(args.params)
        demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}
        print("Running CircAdapt model with given params to generate V5 data...")
        v5_data = evaluate_patient_state(params, demographics)
        if v5_data is None:
            print("ERROR: Model evaluation failed")
            return

        pipeline = CardiorenalPipeline(
            nn_model_path=args.nn_path, llm_model=args.model,
            max_iterations=args.max_iter,
        )

        v7_target = json.loads(args.v7_target) if args.v7_target else None
        result = pipeline.predict_and_explain(v5_data, demographics, v7_target)
        _print_result(result)

        if args.output:
            _save_result(result, args.output)
        return

    # Single patient from V5 JSON
    if args.v5:
        v5_data = json.loads(args.v5)
        demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}

        pipeline = CardiorenalPipeline(
            nn_model_path=args.nn_path, llm_model=args.model,
            max_iterations=args.max_iter,
        )

        v7_target = json.loads(args.v7_target) if args.v7_target else None
        result = pipeline.predict_and_explain(v5_data, demographics, v7_target)
        _print_result(result)

        if args.output:
            _save_result(result, args.output)
        return

    # Batch from CSV
    if args.v5_csv:
        import pandas as pd
        df = pd.read_csv(args.v5_csv)
        print(f"Loaded {len(df)} patients from {args.v5_csv}")

        v5_batch = [dict(row) for _, row in df.iterrows()]
        demographics_batch = [
            {'age': row.get('age', 72), 'sex': row.get('sex', 'M'),
             'BSA': row.get('BSA', 1.9), 'height_m': row.get('height_m', 1.75)}
            for _, row in df.iterrows()
        ]

        pipeline = CardiorenalPipeline(
            nn_model_path=args.nn_path, llm_model=args.model,
            max_iterations=args.max_iter,
        )

        results = pipeline.batch_predict(
            v5_batch, demographics_batch, n_workers=args.workers
        )

        for i, r in enumerate(results):
            print(f"\nPatient {i+1}:")
            _print_result(r, brief=True)

        if args.output:
            _save_result(results, args.output)
        return

    # Demo mode: generate a test patient
    print("No input provided. Running demo with a synthetic patient...")
    from synthetic_cohort import evaluate_patient_state

    v5_params = {
        'Sf_act_scale': 0.8, 'Kf_scale': 0.85, 'inflammation_scale': 0.15,
        'diabetes_scale': 0.1, 'k1_scale': 1.0, 'RAAS_gain': 1.5,
        'TGF_gain': 2.0, 'na_intake': 150.0,
    }
    v7_params = {
        'Sf_act_scale': 0.6, 'Kf_scale': 0.55, 'inflammation_scale': 0.3,
        'diabetes_scale': 0.2, 'k1_scale': 1.3, 'RAAS_gain': 1.8,
        'TGF_gain': 2.0, 'na_intake': 170.0,
    }
    demographics = {'age': 72, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75}

    print("Generating V5 and V7 ground truth from model...")
    v5_data = evaluate_patient_state(v5_params, demographics)
    v7_target = evaluate_patient_state(v7_params, {'age': 78, 'sex': 'M', 'BSA': 1.9, 'height_m': 1.75})

    if v5_data is None or v7_target is None:
        print("ERROR: Model evaluation failed")
        return

    print(f"V5: LVEF={v5_data['LVEF_pct']:.1f}%, GFR={v5_data['GFR_mL_min']:.1f}")
    print(f"V7: LVEF={v7_target['LVEF_pct']:.1f}%, GFR={v7_target['GFR_mL_min']:.1f}")

    pipeline = CardiorenalPipeline(
        nn_model_path=args.nn_path, llm_model=args.model,
        max_iterations=args.max_iter,
    )

    result = pipeline.predict_and_explain(v5_data, demographics, v7_target)
    _print_result(result)

    if args.output:
        _save_result(result, args.output)


def _print_result(result: Dict, brief: bool = False):
    """Print pipeline result."""
    print(f"\n{'='*60}")
    t = result.get('timing', {})
    print(f"  Converged: {result.get('agent_converged', '?')}  "
          f"Error: {result.get('prediction_error', '?'):.4f}  "
          f"Time: {t.get('total_seconds', '?')}s  "
          f"Model runs: {t.get('n_model_runs', '?')}")

    if not brief:
        print(f"\nOptimal Parameters:")
        for k, v in result.get('optimal_params', {}).items():
            default = TUNABLE_PARAMS.get(k, {}).get('default', '?')
            print(f"  {k:25s} = {v:.4f}  (default: {default})")

        print(f"\nParameter Policy:")
        print(f"  {result.get('parameter_policy', 'N/A')[:400]}")

        print(f"\nMechanistic Explanation:")
        print(f"  {result.get('mechanistic_explanation', 'N/A')[:400]}")
    print(f"{'='*60}")


def _save_result(result, path: str):
    """Save result to JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=convert)
    print(f"Result saved to {path}")


if __name__ == '__main__':
    main()
