import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

PRED_DIR  = "data/prediction_batches"
RETURN_DIR = "data/return_batches"
PRICE_DIR = "data/full_history"

# Horizons in TRADING DAYS. "1w" = 5 trading days.
HORIZONS = {
    "1d": 1,
    "3d": 3,
    "5d": 5,     # 1 week (trading)
    # Optional:
    "10d": 10,  # ~2 weeks trading
    "21d": 21,  # ~1 month trading
}

OVERWRITE = True  # set True if you want to recompute existing columns


def get_return_from_csv(ticker: str, article_date: str, window: int):
    """
    % change from first trading day >= article_date to (window) trading days later.
    If no data on article_date, finds most recent date before article_date.
    Requires CSV columns: date, close
    """
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "date" not in df.columns or "close" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    article_date = pd.to_datetime(article_date)

    # First, try to find a date >= article_date
    mask = df["date"] >= article_date
    if mask.any():
        idx0 = df.index[mask][0]
    else:
        # If no date >= article_date, find the most recent date before it
        mask_before = df["date"] < article_date
        if mask_before.any():
            idx0 = df.index[mask_before][-1]  # Get the last (most recent) date before article_date
        else:
            return None

    idxN = idx0 + window
    if idxN >= len(df):
        return None

    p0 = df.loc[idx0, "close"]
    pN = df.loc[idxN, "close"]

    if pd.isna(p0) or pd.isna(pN) or p0 == 0:
        return None

    return (pN - p0) / p0 * 100


def classify_return(pct):
    if pct is None:
        return "NO_DATA", "none"

    ap = abs(pct)
    if ap <= 1:
        return ("UP" if pct > 0 else "DOWN"), "weak"
    if ap <= 3:
        return ("UP" if pct > 0 else "DOWN"), "moderate"
    return ("UP" if pct > 0 else "DOWN"), "strong"


def get_majority_direction_from_csv(ticker: str, article_date: str, window: int):
    """
    Day-over-day majority direction over `window` trading days starting at the
    first close on/before article_date.

    For window=N we look at N day-over-day moves (closes p0..pN) and count UP
    vs DOWN. Example:
        prices: 100 105 106 110 111 99
        moves:   UP  UP  UP  UP DOWN
        window=5 -> 4 UP, 1 DOWN -> "UP"

    Returns (majority, up_days, down_days):
        majority: "UP", "DOWN", "FLAT" (tie), or "NO_DATA"
    """
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return "NO_DATA", 0, 0

    try:
        df = pd.read_csv(path)
    except Exception:
        return "NO_DATA", 0, 0

    if "date" not in df.columns or "close" not in df.columns:
        return "NO_DATA", 0, 0

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    article_date = pd.to_datetime(article_date)

    mask = df["date"] >= article_date
    if mask.any():
        idx0 = df.index[mask][0]
    else:
        mask_before = df["date"] < article_date
        if mask_before.any():
            idx0 = df.index[mask_before][-1]
        else:
            return "NO_DATA", 0, 0

    idxN = idx0 + window
    if idxN >= len(df):
        return "NO_DATA", 0, 0

    closes = df.loc[idx0:idxN, "close"].astype(float).values
    up_days = 0
    down_days = 0
    for i in range(1, len(closes)):
        if pd.isna(closes[i]) or pd.isna(closes[i - 1]):
            continue
        if closes[i] > closes[i - 1]:
            up_days += 1
        elif closes[i] < closes[i - 1]:
            down_days += 1

    if up_days > down_days:
        majority = "UP"
    elif down_days > up_days:
        majority = "DOWN"
    else:
        # Tie on day counts — break with the overall price change p_N - p_0.
        diff = closes[-1] - closes[0]
        if diff > 0:
            majority = "UP"
        elif diff < 0:
            majority = "DOWN"
        else:
            majority = "FLAT"  # genuine flat: equal day counts AND equal start/end

    return majority, up_days, down_days


def process_file(path: str):
    print("📈 Processing:", os.path.basename(path))
    df = pd.read_parquet(path)

    # Ensure needed columns exist
    if "Stock_symbol" not in df.columns or "Date" not in df.columns:
        print("  ⚠️ Missing Stock_symbol/Date, skipping")
        return

    for label, window in HORIZONS.items():
        ret_col = f"return_{label}"
        dir_col = f"actual_direction_{label}"
        str_col = f"actual_strength_{label}"
        maj_col = f"actual_majority_direction_{label}"
        up_col = f"actual_up_days_{label}"
        down_col = f"actual_down_days_{label}"

        all_present = all(c in df.columns for c in
                          (ret_col, dir_col, str_col, maj_col, up_col, down_col))
        if (not OVERWRITE) and all_present:
            print(f"  - {label}: already exists, skipping")
            continue

        rets, dirs, strs = [], [], []
        majs, ups, downs = [], [], []

        for _, row in df.iterrows():
            ticker = str(row["Stock_symbol"]).strip()
            # Your Date looks like epoch-ms. Converting robustly:
            d = row["Date"]
            if isinstance(d, (int, float)):
                date_str = pd.to_datetime(d, unit="ms").strftime("%Y-%m-%d")
            else:
                date_str = str(d)[:10]

            pct = get_return_from_csv(ticker, date_str, window)
            dlab, slab = classify_return(pct)

            maj, u, dn = get_majority_direction_from_csv(ticker, date_str, window)

            rets.append(pct)
            dirs.append(dlab)
            strs.append(slab)
            majs.append(maj)
            ups.append(u)
            downs.append(dn)

        df[ret_col] = rets
        df[dir_col] = dirs
        df[str_col] = strs
        df[maj_col] = majs
        df[up_col] = ups
        df[down_col] = downs

        print(f"  ✅ computed {label}")

    # Write back to the same file (append new horizons)
    pq.write_table(pa.Table.from_pandas(df), path)
    print("  💾 updated:", path)


def main():
    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No return batch parquet files found.")
        return

    for f in files:
        process_file(os.path.join(RETURN_DIR, f))


if __name__ == "__main__":
    main()
