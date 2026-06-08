"""Merge-evaluation pass against the Claude ceiling-test outputs.

Imports merge_evaluation.py and overrides the three path constants so the
same metrics code runs against data/claude_test_* instead of the main
gemini-flash-lite outputs.

Run after claude_ceiling_test_v2.py:
    python merge_evaluation_claude.py

Edit MODEL below if you changed it in claude_ceiling_test_v2.py.
"""
import os
import merge_evaluation as me

MODEL = "claude-opus-4-8"  # must match claude_ceiling_test_v2.py
_slug = MODEL.replace("-", "_").replace(".", "_")

me.PRED_DIR = f"data/claude_test_predictions_{_slug}"
me.EVAL_DIR = f"data/claude_test_evaluation_results_{_slug}"
me.OUT_DIR  = f"data/claude_test_evaluation_summary_{_slug}"

os.makedirs(me.OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 70)
    print(f"MERGE EVAL — Claude ceiling test ({MODEL})")
    print(f"  Reading from: {me.PRED_DIR}")
    print(f"  Writing to:   {me.OUT_DIR}")
    print("=" * 70)
    me.main()
