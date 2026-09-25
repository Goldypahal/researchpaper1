"""
Quick Diagnostic Tool: Verifies Live Connection to Real LLM Reasoning Provider.
Run locally or in Kaggle before starting any large experimental sweep.

Usage:
  python test_real_llm_connection.py
"""

import os
import sys
import json
import time

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_ROOT)
from agent_scaffold.llm_agent import LLMResearchAgent, SimulationDisallowedError


def main():
    print("=" * 75)
    print("LIVE REAL LLM ADAPTER DIAGNOSTIC")
    print("=" * 75)

    print("\n[1] Environment & Secrets Scan:")
    keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY"),
        "LOCAL_LLM_MODEL": os.environ.get("LOCAL_LLM_MODEL")
    }

    # Also check Kaggle UserSecretsClient if in Kaggle
    in_kaggle = os.path.exists("/kaggle")
    print(f"  Kaggle Environment Detected: {in_kaggle}")
    if in_kaggle:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            for k in list(keys.keys()):
                if not keys[k]:
                    try:
                        val = user_secrets.get_secret(k)
                        if val:
                            keys[k] = val
                            os.environ[k] = val
                            print(f"  [Kaggle Secrets] Loaded {k} successfully.")
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [Kaggle Secrets] Note: {e}")

    for k, v in keys.items():
        masked = f"{v[:4]}...{v[-4:]}" if (v and len(v) > 8) else ("SET" if v else "NOT SET")
        status = "[FOUND]" if v else "[MISSING]"
        print(f"  {status:9s} {k:22s} : {masked}")

    print("\n[2] Initializing LLMResearchAgent (STRICT MODE)...")
    try:
        agent = LLMResearchAgent(strict=True, allow_simulation=False)
        print(f"  [OK] Agent initialized successfully with provider='{agent.provider}' (model='{agent.model}')")
    except SimulationDisallowedError as e:
        print("\n" + "!" * 75)
        print("  [HEALTH CHECK FAILED] No real LLM provider is available.")
        print("  STRICT MODE prevented falling back to scripted simulation.")
        print("!" * 75)
        print("\nTo enable real LLM reasoning:")
        print("  1. For Kaggle: Add your secret under Notebook -> 'Add-ons' -> 'Secrets'")
        print("     Supported secret names: GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        print("  2. For Local: export GEMINI_API_KEY='your-key' (or OPENAI_API_KEY / GROQ_API_KEY)")
        print("  3. For Local GPU Model: export LOCAL_LLM_MODEL='Qwen/Qwen2.5-3B-Instruct'")
        print("=" * 75)
        sys.exit(1)

    print("\n[3] Testing Live 1-Shot Health Ping...")
    t0 = time.time()
    ok, resp = agent.verify_connection()
    latency = time.time() - t0

    if not ok:
        print(f"  [ERROR] Live ping failed: {resp}")
        sys.exit(1)

    print(f"  [SUCCESS] Live ping succeeded in {latency:.2f}s! Response: {resp}")

    print("\n[4] Generating 1 Test Research Proposal (Dyck-4 baseline)...")
    t0 = time.time()
    proposal = agent.propose_experiment(
        iteration_idx=1,
        history=[],
        incumbent_best_loss=4.1800,
        base_loss=4.1800,
        task="dyck"
    )
    latency = time.time() - t0

    print(f"  [SUCCESS] Proposal generated in {latency:.2f}s:")
    print(f"    - Provider:           {proposal.get('provider')}")
    print(f"    - Model:              {proposal.get('model')}")
    print(f"    - Prompt Hash:        {proposal.get('prompt_hash')}")
    print(f"    - Hypothesis:         {proposal.get('hypothesis')}")
    print(f"    - Target Component:   {proposal.get('target_component')}")
    print(f"    - Predicted Delta:    {proposal.get('predicted_delta_loss')}")
    print(f"    - Modifications:      {json.dumps(proposal.get('modifications', {}), indent=6)}")

    print("\n" + "=" * 75)
    print("ALL HEALTH CHECKS PASSED: Environment is 100% ready for real LLM sweeps!")
    print("=" * 75)


if __name__ == "__main__":
    main()
