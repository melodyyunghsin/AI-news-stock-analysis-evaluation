"""
Article Quality Audit & Cleanup
================================
Run this BEFORE any prediction pipeline to understand data quality
and filter out broken/unusable articles.

Usage:
  python audit_articles.py              # audit only (no changes)
  python audit_articles.py --clean      # audit + write cleaned parquet files

Reads from: data/return_batches/*.parquet
Writes to:  data/return_batches_clean/*.parquet  (only with --clean flag)
"""

import os
import sys
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

RETURN_DIR = "data/return_batches"
CLEAN_DIR = "data/return_batches_clean"

# Minimum character count for an article to be considered real content.
# Most real news articles are 500+ chars; 200 is a generous floor.
MIN_ARTICLE_LENGTH = 200

# Patterns that indicate a scraper hit an error page, paywall, or placeholder
# instead of actual article content. Case-insensitive matching.
JUNK_PATTERNS = [
    "try using other words",
    "page not found",
    "404 error",
    "404 not found",
    "access denied",
    "subscribe to continue",
    "subscribe to read",
    "javascript is disabled",
    "enable javascript",
    "enable cookies",
    "cookie settings",
    "this page is not available",
    "the page you requested",
    "we couldn't find",
    "we could not find",
    "this content is available to subscribers",
    "you need to be a subscriber",
    "sign in to read",
    "log in to continue",
    "create a free account",
    "your session has expired",
    "too many requests",
    "rate limit",
    "robot",
    "captcha",
    "are you a human",
    "please verify",
    "browser is not supported",
    "unsupported browser",
    "we are working diligently",
    "working diligently to resolve",
    "we apologize for the inconvenience",
    "this article is no longer available",
    "content has been removed",
    "this story has been removed",
]


def classify_article(text):
    """Classify an article as 'valid', 'empty', 'too_short', or 'junk'.

    Returns (label, reason) tuple.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "empty", "no text"

    stripped = text.strip()

    if len(stripped) < MIN_ARTICLE_LENGTH:
        return "too_short", f"only {len(stripped)} chars"

    lower = stripped.lower()
    for pattern in JUNK_PATTERNS:
        if pattern in lower:
            return "junk", f"matched: '{pattern}'"

    return "valid", ""


def audit_file(filepath):
    """Audit a single parquet file and return classification results."""
    df = pd.read_parquet(filepath)
    results = []

    for idx, row in df.iterrows():
        text = row.get("Article_text", "")
        ticker = str(row.get("Stock_symbol", "")).strip()
        article_id = row.get("Article_id", "")
        label, reason = classify_article(text)

        results.append({
            "idx": idx,
            "article_id": article_id,
            "ticker": ticker,
            "label": label,
            "reason": reason,
            "text_length": len(text) if isinstance(text, str) else 0,
            "text_preview": (text[:120] + "...") if isinstance(text, str) and len(text) > 120 else text,
        })

    return df, pd.DataFrame(results)


def print_audit_report(audit_df, filename):
    """Print a summary report for one file."""
    total = len(audit_df)
    counts = audit_df["label"].value_counts()

    valid = counts.get("valid", 0)
    empty = counts.get("empty", 0)
    too_short = counts.get("too_short", 0)
    junk = counts.get("junk", 0)
    unusable = empty + too_short + junk

    print(f"\n{'─' * 60}")
    print(f"File: {filename}")
    print(f"{'─' * 60}")
    print(f"  Total rows:     {total}")
    print(f"  Valid articles:  {valid} ({valid/total:.1%})")
    print(f"  Empty/missing:   {empty} ({empty/total:.1%})")
    print(f"  Too short:       {too_short} ({too_short/total:.1%})")
    print(f"  Junk/error page: {junk} ({junk/total:.1%})")
    print(f"  ─────────────────")
    print(f"  Unusable total:  {unusable} ({unusable/total:.1%})")

    # Per-ticker breakdown
    ticker_stats = audit_df.groupby("ticker")["label"].value_counts().unstack(fill_value=0)
    ticker_totals = audit_df.groupby("ticker").size()

    if "valid" in ticker_stats.columns:
        ticker_stats["valid_pct"] = (ticker_stats["valid"] / ticker_totals * 100).round(1)
        ticker_stats = ticker_stats.sort_values("valid_pct")

        print(f"\n  Per-ticker valid article rate:")
        for ticker in ticker_stats.index:
            v = ticker_stats.loc[ticker].get("valid", 0)
            t = ticker_totals[ticker]
            pct = v / t * 100
            flag = " ⚠️" if pct < 50 else ""
            print(f"    {ticker:8s}: {int(v):4d}/{int(t):4d} valid ({pct:5.1f}%){flag}")

    # Show sample junk articles
    junk_rows = audit_df[audit_df["label"] == "junk"].head(5)
    if len(junk_rows) > 0:
        print(f"\n  Sample junk articles:")
        for _, row in junk_rows.iterrows():
            print(f"    [{row['ticker']}] {row['reason']}")
            print(f"      Preview: {row['text_preview'][:100]}")

    # Show sample too_short articles
    short_rows = audit_df[audit_df["label"] == "too_short"].head(3)
    if len(short_rows) > 0:
        print(f"\n  Sample too-short articles:")
        for _, row in short_rows.iterrows():
            print(f"    [{row['ticker']}] {row['text_length']} chars: {row['text_preview'][:100]}")

    return valid, unusable, total


def clean_file(df, audit_df, output_path):
    """Write a cleaned parquet file keeping only valid articles."""
    valid_indices = audit_df[audit_df["label"] == "valid"]["idx"].values
    clean_df = df.loc[valid_indices].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(clean_df), output_path)
    return len(clean_df)


def main():
    do_clean = "--clean" in sys.argv

    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No parquet files found in", RETURN_DIR)
        return

    if do_clean:
        os.makedirs(CLEAN_DIR, exist_ok=True)

    print("=" * 60)
    print("ARTICLE QUALITY AUDIT")
    print("=" * 60)

    grand_valid = 0
    grand_unusable = 0
    grand_total = 0

    for f in files:
        filepath = os.path.join(RETURN_DIR, f)
        df, audit_df = audit_file(filepath)
        valid, unusable, total = print_audit_report(audit_df, f)

        grand_valid += valid
        grand_unusable += unusable
        grand_total += total

        if do_clean:
            clean_path = os.path.join(CLEAN_DIR, f)
            kept = clean_file(df, audit_df, clean_path)
            print(f"\n  ✅ Cleaned file saved: {clean_path} ({kept} rows)")

    # Grand summary
    print(f"\n{'=' * 60}")
    print(f"GRAND TOTAL")
    print(f"{'=' * 60}")
    print(f"  Total rows across all files: {grand_total}")
    print(f"  Valid:    {grand_valid} ({grand_valid/grand_total:.1%})")
    print(f"  Unusable: {grand_unusable} ({grand_unusable/grand_total:.1%})")

    if grand_unusable / grand_total > 0.1:
        print(f"\n  ⚠️  Over {grand_unusable/grand_total:.0%} of your data is unusable.")
        print(f"     This is severely impacting your accuracy numbers.")
        print(f"     Run with --clean flag to generate filtered parquet files:")
        print(f"       python audit_articles.py --clean")
        print(f"     Then update RETURN_DIR in your prediction script to '{CLEAN_DIR}'")

    if do_clean:
        print(f"\n  Cleaned files written to: {CLEAN_DIR}/")
        print(f"  To use them, update your prediction script:")
        print(f'    RETURN_DIR = "{CLEAN_DIR}"')


if __name__ == "__main__":
    main()
