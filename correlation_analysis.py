import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 0. PATH SAFETY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# =========================
# 1. LOAD ALL CLOSE CSVs
# =========================
files = glob.glob("*_1year_close.csv")
series = []

for f in files:
    try:
        ticker = f.split("_")[0]
        df = pd.read_csv(f, index_col=0, parse_dates=True)

        if "Close" not in df.columns:
            print(f"Skipping {ticker}: No 'Close' column")
            continue

        s = df["Close"].astype(float)
        s.name = ticker
        series.append(s)
    except Exception as e:
        print(f"Error loading {f}: {e}")
        continue

print(f"Loaded {len(series)} tickers with price data")

if len(series) == 0:
    print("ERROR: No data loaded. Check your CSV files.")
    raise SystemExit(1)

# =========================
# 2. BUILD PRICE MATRIX
# =========================
prices = pd.concat(series, axis=1).sort_index()
print(f"Price matrix shape: {prices.shape}")

# =========================
# 3. COMPUTE LOG RETURNS
# =========================
log_returns = np.log(prices / prices.shift(1))

# =========================
# 4. FILTER BAD / INCOMPLETE SERIES
# =========================
# Drop first row (all NaN from shift)
log_returns = log_returns.iloc[1:]

# Analyze data quality
print("\nData Quality Analysis:")
valid_counts = log_returns.notna().sum()
print(
    "Valid observations per ticker - "
    f"Min: {valid_counts.min()}, Max: {valid_counts.max()}, Mean: {valid_counts.mean():.0f}"
)

# Use a more lenient threshold - require at least 30 valid observations
MIN_OBS = 30
valid_tickers = valid_counts[valid_counts >= MIN_OBS].index
log_returns = log_returns[valid_tickers]

print(f"Tickers after filtering (>={MIN_OBS} observations): {log_returns.shape[1]}")

if log_returns.shape[1] < 2:
    print("ERROR: Not enough valid tickers for correlation analysis")
    raise SystemExit(1)

# Preserve raw log returns for overlap checks before filling
log_returns_raw = log_returns.copy()

# For correlation, we need overlapping data
# Fill NaN with forward fill, then backward fill as last resort
log_returns = log_returns.ffill().bfill()

# Check if any columns still have all NaN
all_nan_cols = log_returns.columns[log_returns.isna().all()]
if len(all_nan_cols) > 0:
    print(f"Dropping {len(all_nan_cols)} tickers with no valid data after filling")
    log_returns = log_returns.drop(columns=all_nan_cols)

print(f"Final tickers for correlation: {log_returns.shape[1]}")
print(f"Date range: {log_returns.index.min()} to {log_returns.index.max()}")

# =========================
# 5. SAVE LOG RETURNS
# =========================
log_returns.to_csv("EGX_LogReturns.csv")
print("Saved EGX_LogReturns.csv")

# =========================
# 6. PEARSON CORRELATION
# =========================
corr = log_returns.corr(method="pearson")
corr.to_csv("EGX_Pearson_Correlation_LogReturns.csv")
print("Saved EGX_Pearson_Correlation_LogReturns.csv")

# =========================
# 7. HEATMAP
# =========================
plt.figure(figsize=(18, 14))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    linewidths=0,
    cbar_kws={"label": "Pearson correlation"},
)
plt.title(
    f"EGX Pearson Correlation Matrix - {corr.shape[0]} Stocks (Log Returns)",
    fontsize=16,
    pad=20,
)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("EGX_Pearson_Correlation_LogReturns.png", dpi=300, bbox_inches="tight")
print("Saved heatmap")
plt.show()

# =========================
# 8. TOP 100 CORRELATED PAIRS
# =========================
# Create upper triangle mask to avoid duplicates
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# Extract upper triangle
corr_upper = corr.where(mask)

# Stack and remove NaN
pairs = (
    corr_upper.stack()
    .reset_index()
    .rename(
        columns={
            "level_0": "Ticker_1",
            "level_1": "Ticker_2",
            0: "Correlation",
        }
    )
)

# Add overlap counts per pair from raw (unfilled) log returns
overlap_counts = log_returns_raw.notna().T.dot(log_returns_raw.notna())
overlap_pairs = (
    overlap_counts.where(mask)
    .stack()
    .reset_index()
    .rename(
        columns={
            "level_0": "Ticker_1",
            "level_1": "Ticker_2",
            0: "Overlap_Count",
        }
    )
)
pairs = pairs.merge(overlap_pairs, on=["Ticker_1", "Ticker_2"], how="left")

# Sort by absolute correlation
pairs["Abs_Correlation"] = pairs["Correlation"].abs()
pairs_sorted = pairs.sort_values("Abs_Correlation", ascending=False)

TOP_ABS = 100
TOP_POS = 50
TOP_NEG = 50
MIN_PAIR_OVERLAP = 80

pairs_filtered = pairs_sorted[pairs_sorted["Overlap_Count"] >= MIN_PAIR_OVERLAP]

# Top 100 overall (by absolute value)
top_abs = pairs_filtered.head(TOP_ABS).copy()

# Top 50 positive correlations
top_positive = pairs_filtered[pairs_filtered["Correlation"] > 0].nlargest(
    TOP_POS, "Correlation"
)

# Top 50 negative correlations
top_negative = pairs_filtered[pairs_filtered["Correlation"] < 0].nsmallest(
    TOP_NEG, "Correlation"
)

# =========================
# 9. OUTPUT RESULTS
# =========================
print("\n" + "=" * 70)
print(f"TOP {TOP_ABS} STOCK PAIRS BY ABSOLUTE CORRELATION")
print("=" * 70)
for _, row in top_abs.iterrows():
    print(f"{row['Ticker_1']:>8} <-> {row['Ticker_2']:<8} : {row['Correlation']:>7.4f}")

print("\n" + "=" * 70)
print(f"TOP {TOP_POS} POSITIVELY CORRELATED PAIRS")
print("=" * 70)
for _, row in top_positive.iterrows():
    print(f"{row['Ticker_1']:>8} <-> {row['Ticker_2']:<8} : {row['Correlation']:>7.4f}")

print("\n" + "=" * 70)
print(f"TOP {TOP_NEG} NEGATIVELY CORRELATED PAIRS")
print("=" * 70)
for _, row in top_negative.iterrows():
    print(f"{row['Ticker_1']:>8} <-> {row['Ticker_2']:<8} : {row['Correlation']:>7.4f}")

# Save to CSV
top_abs.to_csv("EGX_Top100_Correlated_Pairs.csv", index=False)
top_positive.to_csv("EGX_Top50_Positive_Correlations.csv", index=False)
top_negative.to_csv("EGX_Top50_Negative_Correlations.csv", index=False)

print("\n✅ DONE - Results saved to CSV files")
print(f"   - Total pairs analyzed: {len(pairs):,}")
print(f"   - Stocks in analysis: {corr.shape[0]}")
