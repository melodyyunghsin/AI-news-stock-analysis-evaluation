"""
Clean Article Text in Parquet Files
=====================================
Strips Nasdaq scraper error prefixes and trailing boilerplate from article text,
then writes cleaned files to a new directory.

Usage:
  python clean_articles.py

Reads from:  data/return_batches/*.parquet
Writes to:   data/return_batches_clean/*.parquet
"""

import os
import re
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

RETURN_DIR = "data/return_batches"
CLEAN_DIR = "data/return_batches_clean"
MIN_CLEAN_LENGTH = 150  # after cleaning, articles shorter than this are marked empty

os.makedirs(CLEAN_DIR, exist_ok=True)


def clean_article_text(text):
    """Strip Nasdaq scraper junk prefix and trailing boilerplate.

    The FNSPID dataset articles scraped from nasdaq.com typically start with:
      "Please try using other words for your search..."
      "Our team is working diligently..."
    followed by a date line, then "Written by X for Y->"
    and only THEN does the actual article begin.

    The end of each article has Nasdaq boilerplate about symbols,
    TipRanks, disclaimers, etc.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    cleaned = text

    # --- Strip the leading junk prefix ---
    # Strategy: find the "->" marker that ends the "Written by...for...Source->" line
    # Everything before that is scraper noise + metadata
    if "try using other words" in cleaned[:400] or "working diligently" in cleaned[:400]:
        arrow_idx = cleaned.find("->")
        if arrow_idx != -1 and arrow_idx < 600:
            cleaned = cleaned[arrow_idx + 2:]
        else:
            # Fallback: try to find the content after the date/author block
            # Look for the pattern after "understanding.\n"
            understanding_idx = cleaned.find("understanding.\n")
            if understanding_idx != -1 and understanding_idx < 400:
                cleaned = cleaned[understanding_idx + len("understanding.\n"):]

    # Some articles have multiple "->" if there are nested source attributions
    # e.g. "Written by X for InvestorPlace->\nInvestorPlace - Stock Market News..."
    # Try to skip past the secondary source line too
    if cleaned.startswith("\n"):
        cleaned = cleaned.lstrip("\n")

    # Skip lines like "InvestorPlace - Stock Market News, Stock Advice & Trading Tips"
    source_prefixes = [
        "InvestorPlace - Stock Market News",
        "MarketBeat -",
        "ETF Trends -",
    ]
    for prefix in source_prefixes:
        if cleaned.startswith(prefix):
            newline_idx = cleaned.find("\n")
            if newline_idx != -1 and newline_idx < 200:
                cleaned = cleaned[newline_idx + 1:]

    # --- Strip trailing boilerplate ---
    cut_markers = [
        "The views and opinions expressed herein",
        "This data feed is not available",
        "\u00a9 2025, Nasdaq, Inc.",  # © 2025
        "\u00a9 2024, Nasdaq, Inc.",
        "\u00a9 2023, Nasdaq, Inc.",
        "To add symbols:",
        "To add instruments:",
        "Smart Portfolio is supported by our partner TipRanks",
        "This story originally appeared on",
    ]
    for marker in cut_markers:
        pos = cleaned.find(marker)
        if pos != -1:
            cleaned = cleaned[:pos]

    # Also strip common promotional footers
    promo_patterns = [
        r"\d+ stocks we like better than .+?When our analyst team",
        r"Want the latest recommendations from Zacks",
        r"Click to get this free report",
        r"Zacks Names .Single Best Pick to Double.",
        r"Why Haven.t You Looked at Zacks",
        r"See Stocks Free >>",
    ]
    for pattern in promo_patterns:
        match = re.search(pattern, cleaned)
        if match:
            cleaned = cleaned[:match.start()]

    # Strip Motley Fool disclosure blocks
    fool_markers = [
        "The Motley Fool has a disclosure policy.",
        "The Motley Fool has positions in and recommends",
        "The Motley Fool recommends",
    ]
    for marker in fool_markers:
        pos = cleaned.find(marker)
        if pos != -1:
            # Keep the text before the disclosure
            cleaned = cleaned[:pos]

    # Strip "See the 10 stocks" + surrounding promo
    see_stocks = cleaned.find("See the 10 stocks")
    if see_stocks != -1:
        cleaned = cleaned[:see_stocks]

    # Final cleanup
    cleaned = cleaned.strip()

    # Remove articles that are too short after cleaning (just metadata remnants)
    if len(cleaned) < MIN_CLEAN_LENGTH:
        return ""

    return cleaned


def process_file(filepath, output_path):
    """Clean all articles in a parquet file and save."""
    df = pd.read_parquet(filepath)

    if "Article_text" not in df.columns:
        print(f"  ⚠️ No Article_text column, skipping")
        return 0, 0, 0

    original_texts = df["Article_text"].fillna("")
    cleaned_texts = original_texts.apply(clean_article_text)

    # Stats
    total = len(df)
    had_content_before = (original_texts.str.len() > 0).sum()
    has_content_after = (cleaned_texts.str.len() > 0).sum()
    was_empty = (original_texts.str.len() == 0).sum()
    lost_in_cleaning = had_content_before - has_content_after

    # Show how much shorter articles got (text was stripped)
    len_before = original_texts.str.len()
    len_after = cleaned_texts.str.len()
    valid_mask = (len_before > 0) & (len_after > 0)
    if valid_mask.sum() > 0:
        avg_reduction = (1 - len_after[valid_mask].mean() / len_before[valid_mask].mean()) * 100
    else:
        avg_reduction = 0

    df["Article_text"] = cleaned_texts
    pq.write_table(pa.Table.from_pandas(df), output_path)

    return total, has_content_after, avg_reduction


def main():
    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No parquet files found in", RETURN_DIR)
        return

    print("=" * 60)
    print("CLEANING ARTICLE TEXT")
    print(f"Reading from: {RETURN_DIR}")
    print(f"Writing to:   {CLEAN_DIR}")
    print("=" * 60)

    grand_total = 0
    grand_valid = 0

    for f in files:
        filepath = os.path.join(RETURN_DIR, f)
        output_path = os.path.join(CLEAN_DIR, f)

        total, valid, avg_reduction = process_file(filepath, output_path)
        grand_total += total
        grand_valid += valid

        empty_pct = (1 - valid / total) * 100 if total > 0 else 0
        print(f"  {f}: {valid}/{total} valid after cleaning "
              f"({empty_pct:.0f}% empty, ~{avg_reduction:.0f}% boilerplate removed)")

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  Total rows:    {grand_total}")
    print(f"  Valid articles: {grand_valid} ({grand_valid/grand_total:.1%})")
    print(f"  Empty/removed:  {grand_total - grand_valid} ({(grand_total - grand_valid)/grand_total:.1%})")
    print(f"\n  Cleaned files saved to: {CLEAN_DIR}/")
    print(f"  Update your prediction script:")
    print(f'    RETURN_DIR = "{CLEAN_DIR}"')
    print(f"\n  Also clear data/prediction_batches/ before re-running predictions")
    print(f"  so the resume logic doesn't skip rows.")


if __name__ == "__main__":
    main()
