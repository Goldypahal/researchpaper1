"""
Top-level entry point for LLM Qualification & Pre-Flight Provider Benchmark Suite.
Candidate Pool Architecture:
                 LLM Qualification
                         │
         ┌───────────────┼────────────────┐
         ↓               ↓                ↓
      NVIDIA            Groq            Mistral
      Nemotron          GPT-OSS         candidate
         │                │
      Super/Ultra       20B/120B

Usage:
  python benchmark_providers.py --calls 3
  python benchmark_providers.py --providers nvidia groq mistral
  python benchmark_providers.py --nvidia-model nvidia/llama-3.1-nemotron-70b-instruct --groq-model openai/gpt-oss-120b
"""

import os
import sys

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.benchmark_providers import run_provider_benchmark_suite
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Qualification & Pre-flight Provider Benchmark Suite.")
    parser.add_argument("--calls", type=int, default=3, help="Calls per candidate (default: 3)")
    parser.add_argument("--providers", nargs="+", default=["nvidia", "groq", "mistral"], help="List of providers to test (e.g. nvidia groq mistral)")
    parser.add_argument("--nvidia-model", type=str, default="nvidia/llama-3.1-nemotron-70b-instruct", help="NVIDIA Nemotron model")
    parser.add_argument("--groq-model", type=str, default="openai/gpt-oss-120b", help="Groq model (e.g. openai/gpt-oss-120b, openai/gpt-oss-20b, llama-3.3-70b-versatile)")
    parser.add_argument("--mistral-model", type=str, default="mistral-small-latest", help="Mistral model (e.g. mistral-small-latest, codestral-latest)")
    parser.add_argument("--gemini-model", type=str, default="gemini-3.6-flash", help="Gemini model")
    parser.add_argument("--openrouter-model", type=str, default="nvidia/nemotron-3-ultra-550b-a55b", help="OpenRouter pinned model")
    args = parser.parse_args()

    models_config = {
        "nvidia": args.nvidia_model,
        "groq": args.groq_model,
        "mistral": args.mistral_model,
        "gemini": args.gemini_model,
        "openrouter": args.openrouter_model
    }

    run_provider_benchmark_suite(
        providers=args.providers,
        models=models_config,
        calls_per_provider=args.calls
    )
