import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# ----------------------------
# FETCH UNG DATA
# ----------------------------
ticker = "UNG"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}...")
ung_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

# Check if data was downloaded successfully
if ung_data.empty:
    print("No data downloaded. Please check the ticker symbol and date range.")
    raise SystemExit(1)

# Use adjusted close prices and remove any NaN values
adj_close = ung_data["Adj Close"].dropna()
prices = adj_close.to_numpy().astype(float).reshape(-1)
dates = adj_close.index

N = len(prices)
print(f"Downloaded {N} data points (after removing NaN values)")

if N < 50:
    print("Not enough data points for meaningful FFT analysis.")
    raise SystemExit(1)

# ----------------------------
# CALCULATE LOG RETURNS
# ----------------------------
print(f"\nPrices array info:")
print(f"  Type: {type(prices)}")
print(f"  Shape: {prices.shape}")
print(f"  Dtype: {prices.dtype}")
print(f"  First 5 values: {prices[:5]}")

log_prices = np.log(prices)
print(f"\nLog prices array info:")
print(f"  Type: {type(log_prices)}")
print(f"  Shape: {log_prices.shape}")
print(f"  Dtype: {log_prices.dtype}")
print(f"  First 5 values: {log_prices[:5]}")

log_returns = np.diff(log_prices)

print(f"\nLog returns array info:")
print(f"  Type: {type(log_returns)}")
print(f"  Shape: {log_returns.shape}")
print(f"  Dtype: {log_returns.dtype}")
print(f"  Length: {len(log_returns)}")
print(f"  First 5 values: {log_returns[:5]}")
print(f"  Contains NaN: {np.any(np.isnan(log_returns))}")
print(f"  Contains Inf: {np.any(np.isinf(log_returns))}")
print(f"  Is empty: {log_returns.size == 0}")

dates_returns = dates[1:]
print(f"\nDates returns length: {len(dates_returns)}")

if len(log_returns) < 50:
    print("Not enough log returns for FFT analysis.")
    raise SystemExit(1)

# Estimate dt (assuming ~252 trading days per year)
dt = 1 / 252  # in years

# ----------------------------
# FFT ON LOG RETURNS
# ----------------------------
print(f"\nAttempting FFT on array of length: {len(log_returns)}")
R = np.fft.fft(log_returns)
freqs = np.fft.fftfreq(len(log_returns), d=dt)
power = np.abs(R) ** 2

# Exclude DC component (0 frequency)
dc_idx = np.argmin(np.abs(freqs))  # Find index closest to 0
power_no_dc = power.copy()
power_no_dc[dc_idx] = 0.0

# Select top 25 frequency components by power
k = 25
top_idx = np.argpartition(power_no_dc, -k)[-k:]
top_idx = top_idx[np.argsort(power_no_dc[top_idx])[::-1]]  # sort descending

# ----------------------------
# RECONSTRUCT DENOISED SIGNAL FROM TOP 25 COMPONENTS
# ----------------------------
R_filtered = np.zeros_like(R)
R_filtered[top_idx] = R[top_idx]
log_returns_denoised = np.fft.ifft(R_filtered).real

# Reconstruct price from denoised returns
log_prices_denoised = np.zeros(len(log_returns) + 1)
log_prices_denoised[0] = log_prices[0]
log_prices_denoised[1:] = log_prices_denoised[0] + np.cumsum(log_returns_denoised)
prices_denoised = np.exp(log_prices_denoised)

# ----------------------------
# MAIN PLOT: ORIGINAL vs FFT RECONSTRUCTED
# ----------------------------
plt.figure(figsize=(14, 7))
plt.plot(
    dates,
    prices,
    label="Original UNG Price",
    linewidth=1.5,
    color="tab:blue",
    alpha=0.7,
)
plt.plot(
    dates,
    prices_denoised,
    label="FFT Denoised (top 25 frequencies)",
    linewidth=2.5,
    color="tab:orange",
    alpha=0.9,
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.title(f"{ticker} Price: Original vs FFT Denoised Signal (1 Year)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# LOG RETURNS COMPARISON
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(
    dates_returns,
    log_returns,
    alpha=0.5,
    label="Original log returns",
    linewidth=1,
    color="tab:blue",
)
plt.plot(
    dates_returns,
    log_returns_denoised,
    linewidth=2,
    label="FFT denoised returns (top 25)",
    color="tab:red",
)
plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Log Return", fontsize=12)
plt.title(f"{ticker} Log Returns: Original vs FFT Denoised", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# POWER SPECTRUM
# ----------------------------
# Focus on positive frequencies for cleaner visualization
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
power_pos = power[pos_mask]

plt.figure(figsize=(12, 6))
plt.semilogy(freqs_pos, power_pos, linewidth=1.5, color="tab:blue")
plt.xlabel("Frequency (cycles per year)", fontsize=12)
plt.ylabel("Power |FFT|²", fontsize=12)
plt.title(f"{ticker} Power Spectrum (Positive Frequencies)", fontsize=14)
plt.grid(True, alpha=0.3, which="both")

# Highlight top 25 components
top_freqs = freqs[top_idx]
top_powers = power[top_idx]
top_pos_mask = top_freqs > 0

plt.scatter(
    top_freqs[top_pos_mask],
    top_powers[top_pos_mask],
    s=80,
    color="red",
    zorder=5,
    label="Top 25 components",
    edgecolors="darkred",
    linewidth=1.5,
)

# Annotate top 5 for clarity
if np.sum(top_pos_mask) > 0:
    top_5_count = min(5, np.sum(top_pos_mask))
    top_5_indices = np.argsort(top_powers[top_pos_mask])[::-1][:top_5_count]
    top_5_pos = top_freqs[top_pos_mask][top_5_indices]
    top_5_pow = top_powers[top_pos_mask][top_5_indices]

    for f, p in zip(top_5_pos, top_5_pow):
        plt.annotate(
            f"{f:.1f}",
            (f, p),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=9,
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
        )

plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# ----------------------------
# STATISTICS
# ----------------------------
print("\n" + "=" * 60)
print("UNG FFT DENOISING ANALYSIS")
print("=" * 60)
print(f"Total data points: {N}")
print(f"Log returns: {len(log_returns)}")
print(f"Original price range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"Denoised price range: ${prices_denoised.min():.2f} - ${prices_denoised.max():.2f}")
print(f"\nOriginal volatility (std of log returns): {np.std(log_returns):.4f}")
print(f"Denoised volatility (std of log returns): {np.std(log_returns_denoised):.4f}")
print(
    f"Noise reduction: {(1 - np.std(log_returns_denoised) / np.std(log_returns)) * 100:.1f}%"
)

print(f"\n{'=' * 60}")
print("TOP 25 FREQUENCY COMPONENTS (excluding DC)")
print(f"{'=' * 60}")
print(f"{'Rank':<6}{'Index':<8}{'Frequency':<15}{'Period (days)':<18}{'Power':<12}")
print("-" * 60)

for i, idx in enumerate(top_idx[:25], 1):
    freq = freqs[idx]
    period_days = (1 / abs(freq)) * 252 if freq != 0 else np.inf
    print(f"{i:<6}{idx:<8}{freq:>10.2f}{period_days:>15.1f}{power[idx]:>15.2e}")

print(f"\n{'=' * 60}")
print("DOMINANT CYCLES DETECTED:")
print(f"{'=' * 60}")
# Show cycles with period between 5 and 252 days (meaningful trading cycles)
meaningful_cycles = []
for idx in top_idx:
    freq = freqs[idx]
    if freq != 0:
        period_days = abs((1 / freq) * 252)
        if 5 <= period_days <= 252:
            meaningful_cycles.append((period_days, power[idx]))

meaningful_cycles.sort(key=lambda x: x[1], reverse=True)
if len(meaningful_cycles) > 0:
    for i, (period, pow_val) in enumerate(meaningful_cycles[:10], 1):
        print(f"{i}. Period: {period:.1f} days (~{period / 21:.1f} months)")
else:
    print("No meaningful cycles detected in the 5-252 day range.")
