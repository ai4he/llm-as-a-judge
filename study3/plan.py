"""Study 3 full-run plan (shared by run_full.py and progress.py)."""
# Judge panel: 5 families, 9B -> 754B (Active/Active-LTS preferred).
PANEL = ["glm-5.1-fp8","gptoss-120b","deepseek-v4-pro","gemma-4-31b","qwen3.6-27b-fp8","qwen3.5-9b"]
RUN_ID = "full"
N = 500
# (dataset_key, n, contamination_tier, sensitive)  -- loaders added incrementally in harness.py
FULL_PLAN = [
    ("chaosnli_snli",       N, "HIGH", False),
    ("chaosnli_mnli",       N, "HIGH", False),
    ("hatexplain",          N, "HIGH", True),
    ("go_emotions",         N, "HIGH", False),
    ("social_bias_frames",  N, "MED",  True),
    # queued (data-access pending): gaqcorpus, lewidi2025 (LOW), election2026 (LOW)
]
