# Codebase Architecture

## Core Simulator
| File | Description |
|---|---|
| `cardiorenal_coupling.py` | CircAdapt heart model wrapper, Hallow renal model (dataclass), inflammatory mediator layer, bidirectional coupling loop (Algorithm 1), and RL-enhanced coupling variant. Everything depends on this. |
| `emission_functions.py` | Converts raw CircAdapt waveforms + renal state into 114 ARIC clinical variables. 17 `emit_*` functions across 8 categories (LV structure, systolic function, Doppler, tissue Doppler, filling pressures, LA, hemodynamics, renal). |
| `config.py` | Single source of truth for all constants. The 8 tunable parameter definitions with ranges, 113 ARIC variable metadata with weights, clinical thresholds, LLM config, cohort generation defaults, NN hyperparameters, RL config. Every other file imports from here. |

## Data Generation
| File | Description |
|---|---|
| `synthetic_cohort.py` | Generates fake patients. Two modes: (1) Paired V5/V7 — samples disease params, runs simulator twice per patient (baseline + 6yr worsened), outputs paired clinical vectors for NN training. (2) Monthly — generates month-by-month trajectories for RL. Also contains `evaluate_patient_state` which is the single-patient forward model used everywhere. |

## Neural Network (V5 → V7 prediction)
| File | Description |
|---|---|
| `train_nn.py` | Residual MLP: `V7 = W_skip * V5 + g_phi(V5)`. Linear skip connection captures "things mostly stay the same over 6 years," nonlinear branch learns the disease-specific changes. Trains on synthetic cohort data. |

## RL System (learned coupling equation)
| File | Description |
|---|---|
| `rl_env.py` | Gymnasium environment wrapping the coupled simulator. Each episode = one patient trajectory over 72-120 monthly steps. Observation = 32-dim state vector. Action = 15-dim (5 alphas + 10 residuals). Reward = physiological plausibility + V7 match. |
| `models/attention_coupling.py` | The RL policy network. Attention-based architecture where cardiac features cross-attend to renal features (and vice versa). Outputs the 5 coupling alphas and 10 inflammatory residuals. Cross-attention weights are interpretable — they show which cardiac features depend on which renal features. |
| `train_rl.py` | PPO training script. Two-stage: (1) synthetic pre-training with dense per-step reward, (2) fine-tuning with terminal V7-only reward. Uses the gym env + attention policy. |

## LLM Agent (parameter recovery)
| File | Description |
|---|---|
| `agent_loop.py` | The LLM agent loop. Given V5 data + V7 target, an LLM (GPT-4o/Gemini/Claude via LiteLLM) iteratively calls tools to find disease parameters that reproduce the V7 target. Has Nelder-Mead fallback if LLM stagnates. Outputs optimal params + mechanistic explanation. |
| `agent_tools.py` | The 4 tools the LLM agent can call: `run_simulation` (run model), `compute_error` (compare to target), `get_sensitivity` (finite-difference Jacobian), `get_clinical_norms` (guideline ranges). |
| `pipeline.py` | End-to-end: V5 input → NN predicts V7 → agent finds params → returns results. Has batch mode for multiple patients with thread parallelism. CLI entry point. |

## Logging
| File | Description |
|---|---|
| `sim_logging.py` | Structured JSON-line logger. Logs 8 key outputs (EF, MAP, CO, SV, GFR, V_blood, Na_excr, P_glom) + params + success/failure per simulation step to `logs/simulation_runs.jsonl`. RL runs also log policy actions (alphas + residuals). |

## Tests
| File | Description |
|---|---|
| `tests/test_parameter_ranges.py` | 58 tests validating the 8 input parameter ranges — boundary, monotonicity, physiological direction, combined extremes, config consistency. |
| `tests/test_end_to_end.py` | RL integration tests — runs coupled simulation with identity coupling and random policy. |
| `tests/test_attention_policy.py` | Tests the attention policy network shapes, forward pass, action generation. |
| `tests/test_backward_compat.py` | Tests that config structures and function signatures haven't broken. |
| `tests/test_message_scaling.py` | Tests the alpha message scaling math (identity at alpha=1, amplification, dampening). |
| `tests/test_ppo_training.py` | Tests PPO training loop runs without crashing (1 iteration). |
| `tests/test_rl_env.py` | Tests gym environment reset, step, observation/action spaces. |

## Other
| File | Description |
|---|---|
| `P_ref_VanOsta2024.npy` | Reference pressure waveform from the VanOsta2024 CircAdapt model. |
| `requirements.txt` | Dependencies: circadapt, numpy, scipy, torch, litellm, gymnasium. |
| `manuscript/` | Paper source files (paper.tex, paper.pdf, refs, style files). |
