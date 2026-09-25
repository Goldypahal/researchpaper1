# Provider Benchmark & Comparative Evaluation Report

**Generated:** 2026-09-25 10:29:59

## 1. Executive Summary

This benchmark measures the live execution characteristics of candidate reasoning providers under standardized prompt conditions for autonomous sequence modeling.

| Provider | Model | Successful Calls | Mean Latency (s) | JSON Validity | Proposal Validity | Reasoning (0-10) | Usefulness (0-10) | Status |
|---|---|---|---|---|---|---|---|---|
| **Nvidia** | `nvidia/llama-3.1-nemotron-70b-instruct` | 0/1 | N/A | N/A | N/A | N/A | N/A | MISSING_CREDENTIALS |
| **Groq** | `openai/gpt-oss-120b` | 0/1 | N/A | N/A | N/A | N/A | N/A | MISSING_CREDENTIALS |
| **Mistral** | `mistral-small-latest` | 0/1 | N/A | N/A | N/A | N/A | N/A | MISSING_CREDENTIALS |

## 2. Experimental Rigor & Variable Freezing Protocol

- **Frozen Experimental Variable:** The primary LLM search arm must lock `provider` and `model` across all seeds (`[42, 101, 202, 303, 404]`).
- **Prohibition of Seed-Provider Mixing:** Alternating providers across seeds within the same arm (e.g., seed 42 on Gemini, seed 101 on Groq) introduces severe confounding variance.
- **Model Dependence Study:** If investigating provider differences, instantiate distinct arms (`LLM-Groq`, `LLM-Mistral`, `LLM-Gemini`) with identical seed allocations.
- **Accurate Quota Documentation:** Cite exact observed account/model quotas on specific test dates rather than generic marketing figures.
- **Gateway Reproducibility:** When using OpenRouter, explicitly pin the underlying model identifier.
