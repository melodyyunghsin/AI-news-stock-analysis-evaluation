"""Merge-evaluation pass against the gemini-pro test outputs.

Imports merge_evaluation.py and overrides the three path constants so the
same metrics code runs against data/pro_test_* instead of the main
gemini-flash-lite outputs.

Run after gemini_predict_pro.py:
    python merge_evaluation_pro.py
"""
import os
import merge_evaluation as me

me.PRED_DIR = "data/pro_test_predictions_gemini_pro"
me.EVAL_DIR = "data/pro_test_evaluation_results_gemini_pro"
me.OUT_DIR  = "data/pro_test_evaluation_summary_gemini_pro"

os.makedirs(me.OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 70)
    print(f"MERGE EVAL — gemini-2.5-pro outputs")
    print(f"  Reading from: {me.PRED_DIR}")
    print(f"  Writing to:   {me.OUT_DIR}")
    print("=" * 70)
    me.main()
