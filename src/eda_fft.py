"""FFT analysis on cybersecurity attack protocol traffic over time."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cybersecurity_attacks.csv"


def load_timestamp_protocol(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load CSV and return a DataFrame with Timestamp and Protocol only."""
    df = pd.read_csv(path, usecols=["Timestamp", "Protocol"], parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def build_protocol_timeseries(
    df: pd.DataFrame, freq: str = "1h"
) -> pd.DataFrame:
    """Resample protocol counts into a regular time series.

    Args:
        df: DataFrame with Timestamp and Protocol columns.
        freq: Resampling frequency (e.g. '1h', '30min', '1D').

    Returns:
        DataFrame indexed by time with one column per protocol (count).
    """
    df = df.set_index("Timestamp")
    ts = (
        df.groupby("Protocol")
        .resample(freq)
        .size()
        .unstack(level=0, fill_value=0)
    )
    return ts


def fft_analysis(signal: np.ndarray, sampling_period_hours: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sided FFT magnitude spectrum.

    Args:
        signal: 1-D real-valued time series.
        sampling_period_hours: Time step between samples in hours.

    Returns:
        (frequencies, magnitudes, raw_fft_coefficients) – one-sided spectrum.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal - signal.mean())  # remove DC offset
    magnitudes = (2.0 / n) * np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=sampling_period_hours)
    return freqs, magnitudes, fft_vals


def noise_threshold(magnitudes: np.ndarray, sigma: float = 3.0) -> float:
    """Compute noise floor as mean + sigma * std of the magnitude spectrum.

    Args:
        magnitudes: One-sided magnitude spectrum (excl. DC at index 0).
        sigma: Number of standard deviations above the mean.

    Returns:
        Threshold value; peaks above this are considered signal.
    """
    mags = magnitudes[1:]  # exclude DC
    return float(mags.mean() + sigma * mags.std())


def filter_and_reconstruct(
    signal: np.ndarray, sampling_period_hours: float, sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Zero out FFT bins below the noise threshold and reconstruct.

    Args:
        signal: Original time series.
        sampling_period_hours: Sampling interval in hours.
        sigma: Noise threshold = mean + sigma * std.

    Returns:
        (freqs, magnitudes, filtered_signal, threshold).
    """
    freqs, mags, fft_vals = fft_analysis(signal, sampling_period_hours)
    thresh = noise_threshold(mags, sigma=sigma)

    # Zero out bins whose magnitude is below the threshold (keep DC)
    mask = mags >= thresh
    mask[0] = True  # always keep DC
    fft_filtered = fft_vals * mask

    # Reconstruct time-domain signal
    filtered_signal = np.fft.irfft(fft_filtered, n=len(signal))
    # Add back the original mean (was removed before FFT)
    filtered_signal += signal.mean()

    return freqs, mags, filtered_signal, thresh


def plot_fft_results(
    ts: pd.DataFrame, freq_hours: float = 1.0, sigma: float = 3.0
) -> None:
    """Run FFT on each protocol column and plot raw vs filtered signal + spectrum."""
    protocols = ts.columns.tolist()
    fig, axes = plt.subplots(len(protocols), 3, figsize=(20, 4 * len(protocols)))

    for i, proto in enumerate(protocols):
        signal = np.asarray(ts[proto].values, dtype=float)
        freqs, mags, filtered, thresh = filter_and_reconstruct(
            signal, sampling_period_hours=freq_hours, sigma=sigma
        )

        # --- Raw time-domain ---
        axes[i, 0].plot(ts.index, signal, linewidth=0.4, alpha=0.5, label="Raw")
        axes[i, 0].plot(ts.index, filtered, linewidth=0.8, color="red", label="Filtered")
        axes[i, 0].set_title(f"{proto} – raw vs filtered signal")
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Count")
        axes[i, 0].legend()

        # --- Spectrum with threshold line ---
        axes[i, 1].stem(freqs[1:], mags[1:], markerfmt=" ", basefmt=" ")
        axes[i, 1].axhline(thresh, color="red", linestyle="--", label=f"Threshold ({sigma}sigma)")
        axes[i, 1].set_title(f"{proto} – FFT spectrum + noise threshold")
        axes[i, 1].set_xlabel("Frequency (cycles / hour)")
        axes[i, 1].set_ylabel("Magnitude")
        axes[i, 1].legend()

        # --- Peaks above threshold ---
        peak_mask = mags[1:] >= thresh
        peak_freqs = freqs[1:][peak_mask]
        peak_mags = mags[1:][peak_mask]
        axes[i, 2].bar(peak_freqs, peak_mags, width=freqs[1] * 0.8, color="green")
        axes[i, 2].set_title(f"{proto} – significant peaks only")
        axes[i, 2].set_xlabel("Frequency (cycles / hour)")
        axes[i, 2].set_ylabel("Magnitude")

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent.parent / "fft_protocol_analysis.png", dpi=150)
    plt.show()


def plot_frequency_by_period(df: pd.DataFrame) -> None:
    """Plot protocol frequency aggregated by day, week, and month."""
    df = df.copy()
    df = df.set_index("Timestamp")

    resample_configs = [
        ("Day", "1D"),
        ("Week", "1W"),
        ("Month", "1ME"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    for ax, (label, freq) in zip(axes, resample_configs):
        ts = (
            df.groupby("Protocol", group_keys=False)
            .resample(freq)
            .size()
            .unstack(level=0, fill_value=0)
        )
        for proto in ts.columns:
            ax.plot(ts.index, ts[proto], label=proto, linewidth=0.8)
        ax.set_title(f"Protocol frequency by {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Packet count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent / "protocol_frequency_day_week_month.png",
        dpi=150,
    )
    plt.show()


def main() -> None:
    """Entry point: load data, build time series, run FFT, and plot."""
    df = load_timestamp_protocol()
    print(f"Loaded {len(df)} rows  |  Protocols: {df['Protocol'].unique().tolist()}")
    print(df.head())

    ts = build_protocol_timeseries(df, freq="1h")
    print(f"\nResampled time series shape: {ts.shape}")
    print(ts.head(10))

    # Print dominant frequencies per protocol (with noise filtering)
    sigma = 3.0
    for proto in ts.columns:
        signal = np.asarray(ts[proto].values, dtype=float)
        freqs, mags, _, thresh = filter_and_reconstruct(
            signal, sampling_period_hours=1.0, sigma=sigma
        )
        n_peaks = int((mags[1:] >= thresh).sum())
        print(f"\n{proto} – noise threshold ({sigma}sigma): {thresh:.4f}  |  {n_peaks} significant peaks")

        # Show only peaks above threshold
        peak_indices = np.where(mags[1:] >= thresh)[0] + 1
        for idx in peak_indices[:10]:  # top 10 at most
            period_h = 1.0 / freqs[idx] if freqs[idx] != 0 else float("inf")
            print(f"  freq={freqs[idx]:.6f}  mag={mags[idx]:.4f}  period={period_h:.1f}h")

    plot_fft_results(ts, freq_hours=1.0, sigma=sigma)
    plot_frequency_by_period(df)


if __name__ == "__main__":
    main()
